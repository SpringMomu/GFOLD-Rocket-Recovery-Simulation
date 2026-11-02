"""Finite-burn numerical predictor-corrector for return-to-launch-site.

The shooting variable is the remaining *propulsive* velocity integral (m/s),
not an instantaneous change to inertial velocity. Each trial propagates the
powered arc, shutdown transient, coast and landing-burn handover. A small
trust-region conic problem corrects the shooting variable. Only trials that
improve the nonlinear merit are accepted; a failed correction never replaces
a good iterate. The powered arc uses the 6-DOF model, with a reduced steering
parameterisation; the coast and landing handover use point-mass predictions.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from common.conic import ConstraintRows

from .attitude import attitude_control
from .vehicle import G0

# Propagation steps of the cooperative (browser / server work-slice) path.
# The exact synchronous shooting keeps the fine steps below; the cooperative
# task must finish inside its execution lead even on a WASM CPU, so it trades
# some propagation accuracy for wall time.  The shooting re-solves during the
# burn, so the aim only has to start close.
FAST_PREDICT_DT = 0.2
FAST_COAST_DT = 0.25
FAST_BURN_DT = 0.2


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def turn_towards(a, b, angle):
    """Shortest rotation, including antiparallel attitudes."""
    c = float(np.clip(a @ b, -1.0, 1.0))
    theta = math.acos(c)
    if theta <= angle:
        return b.copy()
    side = b - c * a
    if np.linalg.norm(side) < 1e-8:
        side = np.cross(a, [0.0, 1.0, 0.0])
        if np.linalg.norm(side) < 1e-8:
            side = np.cross(a, [1.0, 0.0, 0.0])
    return a * math.cos(angle) + unit(side) * math.sin(angle)


def steering(ap, requested, velocity, wind):
    """The same nose-first flow cone in prediction and execution."""
    d = unit(requested)
    vr = velocity - wind
    speed = float(np.linalg.norm(vr))
    if speed > ap.nose_first_speed:
        flow = vr / speed
        if d @ flow > 0.0:
            d = turn_towards(flow, d, math.radians(ap.nose_first_aoa_deg))
    return d


def throttle_command(ap, axis, direction, mass, thrust, hop):
    """Alignment interlock; a low hop retains vertical support while turning."""
    if axis @ direction >= math.cos(math.radians(25.0)):
        return 1.0
    if hop and axis[2] > 0.3:
        return float(np.clip(1.1 * mass * G0 / (thrust * axis[2]), 0.0, 1.0))
    return 0.0


def braking_handover(ap, impulse, axis, velocity, wind):
    """End the nose-first acceleration when it would become a braking burn."""
    vr = velocity - wind
    speed = float(np.linalg.norm(vr))
    if speed <= ap.nose_first_speed:
        return False
    flow = vr / speed
    if float(unit(impulse) @ flow) > 0.0:
        return bool(impulse[:2] @ flow[:2] < 0.0 or axis @ flow < 0.0)
    return bool(ap._hop)


@dataclass
class Prediction:
    impulse: np.ndarray
    miss: np.ndarray
    fuel: float
    reserve: float
    duration: float
    coast_time: float
    valid: bool
    burnout_position: np.ndarray | None = None
    burnout_velocity: np.ndarray | None = None
    free_accel: np.ndarray | None = None

    @property
    def return_ready(self):
        return self.valid and float(np.linalg.norm(self.miss)) <= 40.0

    @property
    def merit(self):
        # Exact L1 terminal penalty, in kg/m. No flight-path-angle or loft
        # target: among trajectories reaching the pad, fuel selects the arc.
        return self.fuel + self.reserve + 20.0 * float(np.abs(self.miss).sum())


def finish(work):
    """Drive the same resumable computation synchronously for headless runs."""
    while True:
        try:
            next(work)
        except StopIteration as done:
            return done.value


def predict(ap, veh, r, v, m, wind_now, impulse, dt=0.05):
    return finish(predict_steps(ap, veh, r, v, m, wind_now, impulse, dt))


def execution_state_steps(ap, veh, impulse, lead, waiting, dt=0.05):
    """Predict the known control during solver latency with the same 6-DOF model."""
    probe = copy.copy(veh)
    for key, value in vars(veh).items():
        if isinstance(value, np.ndarray):
            setattr(probe, key, value.copy())
    probe.controls = copy.copy(veh.controls)
    probe.last = copy.deepcopy(veh.last)
    probe.fin_model_cache = None
    probe.structure = None
    aero = copy.copy(ap.aero)
    if aero is not None:
        aero.cfd = None
        aero.last = {}

    class ForecastWind:
        def at(self, position, time):
            return ap._wind(float(position[2]) - probe.rest_height - ap.h_off)

    desired = veh.axis.copy() if ap._bb_axis is None else ap._bb_axis.copy()
    remaining = impulse.copy()
    delivered, t = 0.0, 0.0
    while t < lead - 1e-9:
        step = min(dt, lead - t)
        r, v, mass = ap._state(probe)
        wind = np.zeros(3) if not ap.boostback_wind else ap._wind(r[2])
        direction = steering(ap, remaining, v, wind)
        thrust = ap._thrust_available(probe, r[2]) * ap.thrust_scale
        if waiting:
            throttle = 0.0
            if ap._hop and probe.axis[2] > 0.3:
                throttle = float(np.clip(1.1 * mass * G0 / (thrust * probe.axis[2]), 0.0, 1.0))
        else:
            throttle = throttle_command(ap, probe.axis, direction, mass, thrust, ap._hop)
        desired = turn_towards(desired, direction, ap.max_rate * step)
        probe.controls.throttle = throttle
        probe.controls.fins_deploy = False
        attitude_control(probe, desired, ap.k_angle, ap.k_rate, ap.max_rate, True, hold_roll=True)
        probe.step(step, aero, ForecastWind())
        dv = float(np.linalg.norm(probe.last["thrust"])) / mass * step
        delivered += dv
        if not waiting:
            mag = float(np.linalg.norm(remaining))
            remaining *= max(0.0, mag - dv) / max(1e-9, mag)
        t += step
        yield
    return probe, desired, remaining, delivered


def predict_steps(ap, veh, r, v, m, wind_now, impulse, dt=0.05, fast=False):
    """Propagate the powered arc through the actual 6-DOF/actuator models.

    The prediction owns its vehicle, controls, aerodynamic scratch state and
    wind adapter. No trial can change the flying vehicle or live CFD data.
    The subsequent coast uses the much cheaper point-mass model.
    """
    s = veh.spec
    probe = copy.copy(veh)
    for key, value in vars(veh).items():
        if isinstance(value, np.ndarray):
            setattr(probe, key, value.copy())
    probe.controls = copy.copy(veh.controls)
    probe.last = copy.deepcopy(veh.last)
    probe.fin_model_cache = None
    probe.structure = None
    aero = copy.copy(ap.aero)
    if aero is not None:
        aero.cfd = None
        aero.last = {}
    blind = not ap.boostback_wind

    class ForecastWind:
        def at(self, position, time):
            return np.zeros(3) if blind else ap._wind(float(position[2]) - probe.rest_height - ap.h_off)

    forecast_wind = ForecastWind()
    if ap.thrust_scale != 1.0:
        performance = probe.engine_performance

        def scaled_performance(pressure):
            thrust, mdot, isp = performance(pressure)
            return thrust * ap.thrust_scale, mdot, isp
        probe.engine_performance = scaled_performance
    desired = veh.axis.copy() if ap._bb_axis is None else ap._bb_axis.copy()
    remaining = float(np.linalg.norm(impulse))
    elapsed = 0.0
    shutting_down = False
    max_time = min(60.0, max(0.0, veh.propellant - 150.0) / s.max_mass_flow + 12.0)
    valid = True
    while elapsed < max_time:
        p, u, mass = ap._state(probe)
        wind = np.zeros(3) if blind else ap._wind(p[2])
        direction = steering(ap, impulse, u, wind)
        thrust = ap._thrust_available(probe, p[2] + ap.h_off) * ap.thrust_scale
        tail = thrust / mass * probe.throttle ** 2 / (2.0 * s.throttle_rate)
        shutting_down = (shutting_down or remaining <= max(ap.boostback_done_dv, tail)
                         or braking_handover(ap, impulse, probe.axis, u, wind))
        command = 0.0 if shutting_down else throttle_command(ap, probe.axis, direction, mass, thrust, ap._hop)
        probe.controls.throttle = command
        probe.controls.fins_deploy = False
        desired = turn_towards(desired, direction, ap.max_rate * dt)
        attitude_control(probe, desired, ap.k_angle, ap.k_rate, ap.max_rate, True, hold_roll=True)
        probe.step(dt, aero, forecast_wind)
        remaining -= float(np.linalg.norm(probe.last["thrust"])) / mass * dt
        elapsed += dt
        yield
        position = ap._state(probe)[0]
        collision = any(np.all(np.abs(box.axes.T @ (position - box.center)) < box.half) for box in ap.obstacles)
        if ap._height(probe) < ap._bb_floor or probe.propellant < 150 or probe.state != "FLYING" or collision:
            valid = False
            break
        if shutting_down and probe.throttle < 1e-6:
            break
    else:
        valid = False
    p, u, mass = ap._state(probe)
    # Priming the landing-authority tables is also resumable: rebuilding all
    # fin/angle probes at once would otherwise create a WASM frame spike.
    if ap.aero is not None and (ap._auth_cache[1] is None or ap.elapsed - ap._auth_cache[0] >= ap.aoa_refresh):
        yield from ap._trim_tables_steps(veh)
        yield from ap._aoa_table_steps(veh)
    coast = yield from ap._coast_steps(veh, p, u, mass, wind_now,
                                      dt=FAST_COAST_DT if fast else 0.25,
                                      blind=blind, interpolate=True)
    if coast is None:
        return Prediction(impulse.copy(), p[:2].copy(), m - mass, veh.propellant + 1000.0, elapsed, 0.0, False)
    miss, _ = yield from ap._burn_miss_steps(veh, coast[1], coast[2], mass, wind_now,
                                             dt=FAST_BURN_DT if fast else 0.2, blind=blind, interpolate=True)
    # Rocket-equation landing budget, including gravity loss over the landing
    # burn, with dispersions. It is a reserve, not fuel available to boost-back.
    landing_dv = float(np.linalg.norm(coast[2])) + 0.5 * G0 * coast[3] + 15.0
    reserve = 1.2 * mass * (1.0 - math.exp(-landing_dv / (s.isp_sl * G0))) + 150.0
    valid = (valid and math.isfinite(reserve) and reserve < veh.propellant - (m - mass)
             and np.all(np.isfinite(miss)))
    free_accel = np.asarray(probe.last["aero"], float) / mass + np.array([0.0, 0.0, -G0])
    return Prediction(impulse.copy(), miss, m - mass, reserve, elapsed, coast[0], bool(valid),
                      p.copy(), u.copy(), free_accel)


def terminal_probe_steps(ap, veh, m, wind_now, reference, impulse, fast=False):
    """Local translational sensitivity about a fully propagated powered arc.

    A thrust-integral perturbation changes burnout velocity, duration, position
    and mass. The atmospheric coast and landing handover are then integrated
    nonlinearly. Only the derivative uses this approximation: every proposed
    command is accepted against a fresh full 6-DOF propagation.
    """
    delta = impulse - reference.impulse
    mass = m - reference.fuel
    dnorm = float(np.linalg.norm(impulse) - np.linalg.norm(reference.impulse))
    thrust_accel = ap._thrust_available(veh, reference.burnout_position[2] + ap.h_off) * ap.thrust_scale / mass
    extra_time = dnorm / max(1.0, thrust_accel)
    p = (reference.burnout_position + 0.5 * reference.duration * delta
         + (reference.burnout_velocity - 0.5 * reference.impulse) * extra_time)
    v = reference.burnout_velocity + delta + reference.free_accel * extra_time
    mass *= math.exp(-dnorm / (veh.spec.isp_sl * G0))
    blind = not ap.boostback_wind
    coast = yield from ap._coast_steps(veh, p, v, mass, wind_now,
                                       dt=FAST_COAST_DT if fast else 0.25,
                                       blind=blind, interpolate=True)
    if coast is None:
        return reference.miss.copy(), reference.fuel + reference.reserve
    miss, _ = yield from ap._burn_miss_steps(veh, coast[1], coast[2], mass, wind_now,
                                           dt=FAST_BURN_DT if fast else 0.2, blind=blind, interpolate=True)
    landing_dv = float(np.linalg.norm(coast[2])) + 0.5 * G0 * coast[3] + 15.0
    reserve = 1.2 * mass * (1.0 - math.exp(-landing_dv / (veh.spec.isp_sl * G0))) + 150.0
    return miss, m - mass + reserve


def correction(solver, x, prediction, jacobian, fuel_gradient, radius, dv_limit):
    """Trust-region SOCP: fuel gradient + exact L1 endpoint error.

    Variables: delta(3), positive/negative endpoint slacks(4). Physical fuel
    bounds and the trust region are second-order cones, with no matrix inverse
    or determinant test on a nearly singular endpoint sensitivity.
    """
    rows = ConstraintRows()
    for i in range(2):
        rows([(j, jacobian[i, j]) for j in range(3)] + [(3 + i, -1.0), (5 + i, 1.0)], -prediction.miss[i])
    for i in range(3, 7):
        rows([(i, -1.0)], 0.0)
    rows([], radius)
    for j in range(3):
        rows([(j, -1.0)], 0.0)
    rows([], dv_limit)
    for j in range(3):
        rows([(j, -1.0)], x[j])
    P = sparse.diags([0.05] * 3 + [0.0] * 4, format="csc")
    q = np.r_[fuel_gradient, np.full(4, 20.0)]
    step, status = solver.solve(P, q, rows.matrix(7), rows.rhs, 2, 4, [4, 4])
    return (None if step is None else step[:3]), status


def solve(ap, veh, r, v, m, wind_now, seed, iterations=4):
    return finish(solve_steps(ap, veh, r, v, m, wind_now, seed, iterations))


def solve_steps(ap, veh, r, v, m, wind_now, seed, iterations=4, progress=False, fast=False, alternatives=()):
    """Successive local convex corrections with nonlinear accept/reject."""
    x = np.asarray(seed, float).copy()
    dt = FAST_PREDICT_DT if fast else 0.05
    best = yield from predict_steps(ap, veh, r, v, m, wind_now, x, dt=dt, fast=fast)
    if progress and best.valid:
        yield best
    if fast and best.return_ready:
        return best, "verified", 1
    radius = 40.0
    evaluations = 1
    status = "prediction"
    # Distinct initial powered arcs can lie in different aerodynamic basins.
    # Compare their full nonlinear predictions before taking local derivatives.
    for alternative in alternatives:
        trial = yield from predict_steps(ap, veh, r, v, m, wind_now, np.asarray(alternative, float),
                                         dt=dt, fast=fast)
        evaluations += 1
        if ((trial.valid and not best.valid)
                or (trial.valid == best.valid and trial.merit < best.merit)):
            x, best = trial.impulse.copy(), trial
            if progress and best.valid:
                yield best
            if fast and best.return_ready:
                return best, "verified", evaluations
    dv_limit = veh.spec.isp_sl * G0 * math.log(m / (veh.spec.dry_mass + 150.0))
    if not math.isfinite(dv_limit) or dv_limit <= 0:
        return None, "fuel", evaluations
    for _ in range(iterations):
        jac = np.zeros((2, 3))
        grad = np.zeros(3)
        for j in range(3):
            probe = x.copy()
            probe[j] += 2.0
            if fast and best.burnout_position is not None:
                miss, cost = yield from terminal_probe_steps(ap, veh, m, wind_now, best, probe, fast=fast)
            else:
                trial = yield from predict_steps(ap, veh, r, v, m, wind_now, probe, dt=dt, fast=fast)
                miss, cost = trial.miss, trial.fuel + trial.reserve
            evaluations += 1
            jac[:, j] = (miss - best.miss) / 2.0
            grad[j] = (cost - best.fuel - best.reserve) / 2.0
        if not np.all(np.isfinite(jac)) or not np.all(np.isfinite(grad)):
            status = "nonfinite"
            break
        step, status = correction(ap.planner.solver, x, best, jac, grad, radius, dv_limit)
        if step is None:
            break
        accepted = False
        for scale in (1.0, 0.5, 0.25):
            trial = yield from predict_steps(ap, veh, r, v, m, wind_now, x + scale * step, dt=dt, fast=fast)
            evaluations += 1
            # Infeasible seeds may improve, but cannot displace a valid trial.
            if (trial.valid or not best.valid) and trial.merit < best.merit:
                x, best = trial.impulse.copy(), trial
                accepted = True
                if progress:
                    yield best
                if fast and best.return_ready:
                    return best, status, evaluations
                break
        radius = min(80.0, radius * 1.4) if accepted else radius * 0.3
        if radius < 1.0 or (accepted and np.linalg.norm(step) < 0.5):
            break
    return best if best.valid else None, status, evaluations
