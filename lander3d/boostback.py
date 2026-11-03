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
from itertools import pairwise

import numpy as np
from scipy import sparse

from common.conic import ConstraintRows

from .attitude import attitude_control
from .vehicle import G0, atmosphere

# Propagation steps of the cooperative (browser / server work-slice) path.
# The exact synchronous shooting keeps the fine steps below; the cooperative
# task must finish inside its execution lead even on a WASM CPU, so it trades
# some propagation accuracy for wall time.  The shooting re-solves during the
# burn, so the aim only has to start close.
FAST_PREDICT_DT = 0.2
FAST_COAST_DT = 0.25
FAST_BURN_DT = 0.2
# Engines-first flip after the powered arc, before the flow-aligned coast
# takes over (see predict_steps).
FLIP_MAX_TIME = 8.0
FLIP_DONE_COS = math.cos(math.radians(15.0))
FLIP_DT = 0.1
# Cost of one step of a resumable computation, in work units (~25 us of
# CPython each), yielded by the step so the cooperative path can give the
# solve a fixed amount of computation per simulated second
# (Autopilot3D.boostback_work_rate).  A bare ``yield`` is one unit (a
# point-mass step).  Measured: a 6-DOF prediction step ~250 us, a coast step
# ~40 us, one angle of the attitude/trim tables ~1.5 ms.
WORK_6DOF = 10
WORK_COAST = 2
WORK_TABLE = 60


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def prediction_matches_state(prediction, age, position, velocity):
    """Whether the live vehicle still follows a delayed predicted powered arc."""
    trajectory = prediction.trajectory
    if not trajectory or len(trajectory) < 2 or age < 0.0 or age > prediction.duration:
        return False
    for (t0, p0), (t1, p1) in pairwise(trajectory):
        if t0 <= age <= t1 and t1 > t0:
            u = (age - t0) / (t1 - t0)
            expected = np.asarray(p0) + u * (np.asarray(p1) - p0)
            predicted_v = (np.asarray(p1) - p0) / (t1 - t0)
            return (np.linalg.norm(position - expected) < 30.0
                    and np.linalg.norm(velocity - predicted_v) < 15.0)
    return False


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


# Engines-first at high dynamic pressure the air pins the booster near the
# flow (grid fins stowed for the boost-back; the RCS cannot hold it far off,
# and the throttle waits for the alignment): 250 m/s out at 1 km (20-35 kPa)
# it held a thrust direction 40 deg off the flow at zero throttle until it
# fell out of the sky.  The thrust may leave the flow by this much, from no
# limit at RETRO_FREE_Q to nose_first_aoa_deg at RETRO_PINNED_Q.
NOSE_FIRST_HANDOVER_COS = math.cos(math.radians(70.0))   # the nose-first handover only for a change clearly along the flow
RETRO_FREE_Q = 5000.0
RETRO_PINNED_Q = 12000.0


def steering(ap, requested, velocity, wind, height=None):
    """The same flow cones in prediction and execution: nose-first above
    ``nose_first_speed``; engines-first at high dynamic pressure (needs the
    ``height``)."""
    d = unit(requested)
    vr = velocity - wind
    speed = float(np.linalg.norm(vr))
    if speed > ap.nose_first_speed:
        flow = vr / speed
        if d @ flow > 0.0:
            d = turn_towards(flow, d, math.radians(ap.nose_first_aoa_deg))
        elif height is not None:
            q = 0.5 * atmosphere(max(0.0, float(height)))[0] * speed * speed
            if q > RETRO_FREE_Q:
                limit = float(np.interp(q, (RETRO_FREE_Q, RETRO_PINNED_Q), (math.pi, math.radians(ap.nose_first_aoa_deg))))
                d = turn_towards(-flow, d, limit)
    return d


def throttle_command(ap, axis, direction, mass, thrust, hop):
    """Alignment interlock; a low hop retains vertical support while turning."""
    if axis @ direction >= math.cos(math.radians(25.0)):
        return 1.0
    if hop and axis[2] > 0.3:
        return float(np.clip(1.1 * mass * G0 / (thrust * axis[2]), 0.0, 1.0))
    return 0.0


def braking_handover(ap, impulse, axis, velocity, wind, position=None):
    """End the nose-first acceleration when it would become a braking burn.

    A hop also hands over once the change wanted is a braking one -- but only
    braking of the *return*: its horizontal part points away from the pad
    (``position`` is relative to it).  A booster still flying away from the
    pad wants a retrograde (engines-first, aerodynamically stable) burn to
    turn round; ending the hop there left the whole divert to the landing
    burn (e.g. 3 km off at 800 m receding at 50 m/s: ~110 s, out of fuel)."""
    vr = velocity - wind
    speed = float(np.linalg.norm(vr))
    if speed <= ap.nose_first_speed:
        return False
    flow = vr / speed
    along = float(unit(impulse) @ flow)
    if along > NOSE_FIRST_HANDOVER_COS:
        return bool(impulse[:2] @ flow[:2] < 0.0 or axis @ flow < 0.0)
    if along > 0.0 or not ap._hop:
        # (Across the flow -- a crosswind of 21 m/s with the booster slowed to
        # ~50 m/s airspeed -- "against the horizontal flow" is not a braking
        # of the return: the sign of a near-zero product ended the return
        # 2.3 km out, depending on the timing.)
        return False
    return position is None or float(impulse[:2] @ position[:2]) > 0.0


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
    trajectory: list | None = None  # (seconds from the prediction epoch, position relative to target)
    coast_position: np.ndarray | None = None   # after the engines-first flip, where the point-mass coast starts
    coast_velocity: np.ndarray | None = None
    flip_time: float = 0.0

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
        direction = steering(ap, remaining, v, wind, r[2])
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
        yield WORK_6DOF
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
    trajectory = [(0.0, r.copy())]
    next_point = 0.5
    remaining = float(np.linalg.norm(impulse))
    elapsed = 0.0
    shutting_down = False
    max_time = min(60.0, max(0.0, veh.propellant - 150.0) / s.max_mass_flow + 12.0)
    valid = True
    up = np.array([0.0, 0.0, 1.0])
    while elapsed < max_time:
        p, u, mass = ap._state(probe)
        wind = np.zeros(3) if blind else ap._wind(p[2])
        direction = steering(ap, impulse, u, wind, p[2])
        thrust = ap._thrust_available(probe, p[2] + ap.h_off) * ap.thrust_scale
        tail = thrust / mass * probe.throttle ** 2 / (2.0 * s.throttle_rate)
        shutting_down = (shutting_down or remaining <= max(ap.boostback_done_dv, tail)
                         or braking_handover(ap, impulse, probe.axis, u, wind, p))
        command = 0.0 if shutting_down else throttle_command(ap, probe.axis, direction, mass, thrust, ap._hop)
        probe.controls.throttle = command
        probe.controls.fins_deploy = False
        desired = turn_towards(desired, direction, ap.max_rate * dt)
        attitude_control(probe, desired, ap.k_angle, ap.k_rate, ap.max_rate, True, hold_roll=True)
        probe.step(dt, aero, forecast_wind)
        remaining -= float(np.linalg.norm(probe.last["thrust"])) / mass * dt
        elapsed += dt
        yield WORK_6DOF
        position = ap._state(probe)[0]
        if elapsed >= next_point:
            trajectory.append((elapsed, position.copy()))
            next_point = elapsed + 0.5
        collision = any(np.all(np.abs(box.axes.T @ (position - box.center)) < box.half) for box in ap.obstacles)
        stop_lost = False
        if ap._bb_relaxed and not shutting_down:
            # A relaxed return has no fixed floor; it ends where execution
            # would end it, once a vertical stop is about to become impossible.
            vel = probe.vel
            f_v, t_v, _ = ap._burn_need(probe, position * up, vel * up, mass,
                                        max(0.0, float(probe.last["aero"][2]) / mass))
            stop_lost = ap._return_vertical_due(f_v, t_v)
        if (ap._height(probe) < ap._bb_floor or probe.propellant < 150 or probe.state != "FLYING" or collision
                or stop_lost):
            valid = False
            break
        if shutting_down and probe.throttle < 1e-6:
            break
    else:
        valid = False
    p, u, mass = ap._state(probe)
    burnout = (elapsed, p.copy(), u.copy(), np.asarray(probe.last["aero"], float) / mass)
    if valid and ap.aero is not None:
        # The coast flies engines-first.  A return that ended nose-first (it
        # turned round and accelerated towards the pad) first flips through
        # broadside, at full dynamic pressure and with the grid fins coming
        # out: ~20 m/s^2 of drag for a couple of seconds, which the
        # flow-aligned point-mass coast does not see -- the booster then came
        # down hundreds of metres short.  The flip is flown point-mass with
        # the axis turning at the attitude loop's rate limit and the fins
        # deploying at their rate (a 6-DOF flip made each prediction so much
        # dearer that the cooperative solver stopped converging); only the
        # drag of the flip is taken (see below).
        flip = 0.0
        axis = probe.axis.copy()
        fins = float(probe.fins_deploy)
        trim = ap._lift_trim(veh)
        # Only for hop-style returns (a fast descent, or low: they burn out
        # low and fast, where the flip costs hundreds of metres).  A classic
        # boost-back burns out high and slow; there, modelling the flip made
        # the cooperative shooting stall in a 21 m/s wind (scenario 8) and
        # the small difference is left to the glide.
        while ap._hop and flip < FLIP_MAX_TIME:
            vr = u - (np.zeros(3) if blind else ap._wind(p[2]))
            speed = float(np.linalg.norm(vr))
            if speed < 25.0 or float(axis @ (-vr / speed)) >= FLIP_DONE_COS:
                break
            axis = turn_towards(axis, -vr / speed, ap.max_rate * FLIP_DT)
            fins = min(1.0, fins + FLIP_DT / s.fin_deploy_time)
            rho, _, _, a_snd = atmosphere(max(0.0, float(p[2])))
            force = ap.aero.quick_force(vr, axis, rho, a_snd, 0.0, fins=fins, trim=trim)
            # Drag only: the sideways lift depends on the exact attitude path
            # through broadside (fins still deploying, RCS) and made the
            # shooting map too rough to converge in a 21 m/s wind.
            flow = vr / speed
            force = float(force @ flow) * flow
            u = u + (force / mass + np.array([0.0, 0.0, -G0])) * FLIP_DT
            p = p + u * FLIP_DT
            flip += FLIP_DT
            yield
            if p[2] <= 0.0:
                valid = False
                break
            if elapsed + flip >= next_point:
                trajectory.append((elapsed + flip, p.copy()))
                next_point = elapsed + flip + 0.5
        elapsed += flip
    if elapsed > trajectory[-1][0]:
        trajectory.append((elapsed, p.copy()))
    # Priming the landing-authority tables is also resumable: rebuilding all
    # fin/angle probes at once would otherwise create a WASM frame spike.
    if ap.aero is not None and (ap._auth_cache[1] is None or ap.elapsed - ap._auth_cache[0] >= ap.aoa_refresh):
        yield from ap._trim_tables_steps(veh)
        yield from ap._aoa_table_steps(veh)
    coast = yield from ap._coast_steps(veh, p, u, mass, wind_now,
                                      dt=FAST_COAST_DT if fast else 0.25,
                                      blind=blind, interpolate=True,
                                      trace=trajectory, trace_offset=elapsed)
    if coast is None:
        return Prediction(impulse.copy(), p[:2].copy(), m - mass, veh.propellant + 1000.0, burnout[0], 0.0, False)
    ignition_time = elapsed + coast[0]
    if ignition_time > trajectory[-1][0] + 1e-9:
        trajectory.append((ignition_time, coast[1].copy()))
    else:
        trajectory[-1] = (ignition_time, coast[1].copy())
    miss, _ = yield from ap._burn_miss_steps(veh, coast[1], coast[2], mass, wind_now,
                                             dt=FAST_BURN_DT if fast else 0.2, blind=blind, interpolate=True,
                                             trace=trajectory, trace_offset=ignition_time)
    # Rocket-equation landing budget, including gravity loss over the landing
    # burn, with dispersions. It is a reserve, not fuel available to boost-back.
    landing_dv = float(np.linalg.norm(coast[2])) + 0.5 * G0 * coast[3] + 15.0
    reserve = 1.2 * mass * (1.0 - math.exp(-landing_dv / (s.isp_sl * G0))) + 150.0
    # The handover must leave a landing burn that can still stop the fall: a
    # return that only gets back by diving (a hop out of a fast descent)
    # lights the burn past its latest point, and the coast then hands over
    # at once whatever the thrust it would need.
    vertical, _, _ = ap._burn_need(veh, coast[1] * up, coast[2] * up, mass, 0.0)
    valid = (valid and math.isfinite(reserve) and reserve < veh.propellant - (m - mass)
             and np.all(np.isfinite(miss)) and vertical <= ap.boostback_handover_cap)
    # The burnout state (end of the powered arc) seeds the local sensitivity
    # of terminal_probe_steps; coast_time runs from there to ignition.
    free_accel = burnout[3] + np.array([0.0, 0.0, -G0])
    return Prediction(impulse.copy(), miss, m - mass, reserve, burnout[0], ignition_time - burnout[0], bool(valid),
                      burnout[1], burnout[2], free_accel, trajectory, p.copy(), u.copy(), elapsed - burnout[0])


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
    if reference.coast_position is not None:
        # The flip to engines-first is carried over from the reference (its
        # 6-DOF drag), with the perturbed burnout velocity over its duration.
        p = p + (reference.coast_position - reference.burnout_position) + delta * reference.flip_time
        v = v + (reference.coast_velocity - reference.burnout_velocity)
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
    evaluations = 0
    best = yield from predict_steps(ap, veh, r, v, m, wind_now, x, dt=dt, fast=fast)
    evaluations += 1
    if progress and best.valid:
        yield best
    if fast and best.return_ready:
        return best, "verified", evaluations
    radius = 40.0
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
            # The derivative comes from the local translational model about
            # the accepted arc (cheap point-mass); every step it proposes is
            # still accepted only against a full propagation.  Three full
            # 6-DOF probes per iteration were about half of a solve.
            if best.burnout_position is not None:
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
