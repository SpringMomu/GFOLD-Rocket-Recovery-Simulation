"""Falcon-9-style recovery autopilot for the 3D booster.

Coast -> (boost-back) -> one landing burn straight to leg contact.  The burn is
planned by the G-FOLD planner (``planner.py``); attitude control and actuator
allocation are in ``attitude.py``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from common.conic import golden_section_search

from .aero import grid_fin_forces
from .attitude import attitude_control
from .planner import (
    KeepOutBox,
    Plan3D,
    PlanConfig3D,
    Planner3D,
    along_path,
    detour_path,
    detour_paths,
    separating_planes,
)
from .vehicle import G0, atmosphere

# Landing-burn tilt budget vs height (m -> deg): upright at contact.  It is
# measured from the wind trim (Autopilot3D._trim_xy: in a cross wind the
# booster holds its place leaning into the wind), which fades out over the
# last TRIM_FADE metres.  (Holding the planner to this budget too was tried:
# it then gives up the last-seconds corrections the tracker can still
# partly fly, and lands further off in wind.)
TILT_BUDGET = ([0.3, 1.5, 3.0, 8.0, 15.0, 40.0], [0.6, 2.0, 5.0, 12.0, 20.0, 75.0])
TRIM_FADE = (0.5, 5.5)          # m: upright below the first height, full trim above the second
SLEW_DEG = ([2.0, 30.0, 300.0], [8.0, 15.0, 25.0])   # thrust-direction slew-rate limit (deg/s) by height
# Angle of attack (thrust vs the oncoming flow) vs dynamic pressure (Pa ->
# deg): only a fallback without an aerodynamic model.  With one, the limit is
# physical -- the attitude the TVC can hold against the aerodynamic moment
# (Autopilot3D._aoa_table).
Q_AOA_BUDGET = ([300.0, 800.0, 1500.0, 2500.0, 8000.0], [115.0, 90.0, 60.0, 30.0, 12.0])
AOA_Q_GRID = (150.0, 300.0, 500.0, 800.0, 1200.0, 2000.0, 3500.0, 6000.0, 10000.0, 20000.0)


class _AttitudeProbe:
    """The booster at another attitude, for the aerodynamic model only."""

    def __init__(self, veh, R):
        self.R = R
        self._mp = veh.mass_properties()
        self.legs = veh.legs
        self.omega = np.zeros(3)
        self.fin_defl = np.zeros(4)
        self.fins_deploy = 1.0

    def mass_properties(self):
        return self._mp


class _Scored:
    """A plan whose objective, for the time search, is ``value(plan)``
    (unreachable plans rank behind every reaching one)."""

    def __init__(self, plan, value):
        self.plan = plan
        self.objective = value(plan)


class Autopilot3D:
    """Recovery by one convex optimiser from the moment the autopilot is on.

    From the current state to the gate (the pad, or the catch point above the
    arms) a single G-FOLD SOCP (Planner3D) plans the whole flight: thrust may
    drop to zero (the engine is off, coasting), so whether and when to boost
    back, coast and brake is the optimiser's decision -- there is no coast
    guidance, boost-back or ignition logic of its own any more.  The final
    time is the shortest one whose plan reaches the gate clear of the
    obstacles and the ground (hard limits); propellant only breaks ties.  It
    is re-planned every 0.2-0.6 s from the measured state, and every second a
    shorter time is tried; when the plan in hand no longer reaches, longer
    times are searched until one does.  The tracker flies the plan's thrust
    plus PD feedback; where the plan coasts, the engine is off and the body
    points engines-first into the flow, turning to the next burn's thrust
    direction shortly before it.  A tower catch ends in a hold above the
    arms (_catch_hold).
    """

    burn_replan_period = 0.2
    burn_fraction_nominal = 0.85
    minimum_throttle = 0.20
    divert_tilt_deg = 75.0        # thrust tilt the plan may use far from the end (the tracker's budget up high)
    touchdown_speed = 1.2
    gate_height = 0.0             # G-FOLD plan ends on the pad (base height 0) ...
    gate_speed = 1.2              # ... vertical, at the touchdown sink rate
    low_rcs_height = 15.0         # RCS assists TVC close to the ground
    sink_gain = 0.5               # final approach: sink <= 1.2 + 0.5 h (m/s): 6.2 at 10 m, 2.2 at 2 m
    sink_track_height = 15.0      # the tracker enforces the envelope below this height
    legs_deploy_height = 450.0
    lift_gradient_cap = 3.0       # body lift per radian of tilt, in units of thrust accel (drag model)
    search_backoff = 1.0          # s between final-time searches (doubled after failures, up to x4)
    # Free final time: J = propellant + time_weight t_f among plans that reach
    # the gate (_search_time: golden section, longer until one reaches); every
    # search_backoff a step either side (retime_step of the time to go).
    time_weight = 2.0             # propellant units (m/s) per second of flight: time counts double
    retime_step = 0.08
    time_search_evals = 9         # golden-section evaluations
    time_grow = 1.5               # longer times tried by this factor ...
    time_grow_steps = 6           # ... at most this often
    engage_refine = 2             # re-solves about the first plan (successive convexification)
    trust_weight = 0.01           # per (m/s)^2 s: trust region round the reference plan's velocities
    ignite_soon = 1.0             # s: a landing burn due this soon is checked against ...
    ignite_delay = 1.5            # ... the same plan lit this much later (still reaching: wait)
    burn_gap_merge = 3.0          # s: free-plan burns closer than this are one burn
    burn_sigma_frac = 0.5         # free-plan thrust above this share of the minimum throttle counts as a burn
    nose_first_speed = 50.0       # m/s: above this a nose-first burn keeps its thrust ...
    nose_first_aoa_deg = 20.0     # ... within this of the motion (planner: PlanConfig3D.nose_first_aoa_deg)
    glide_aoa_deg = 15.0          # largest angle of attack of the glide (body + grid-fin lift)
    coast_accel = 1.0             # m/s^2: planned thrust below this is a coast (engine off)
    burn_lookahead = 3.0          # s: turn to the next burn's direction this early
    # Tower catch: the burn ends with the collar this high above the arm tops,
    # sinking this slowly; there the booster holds, centred, until the arms
    # have closed round the barrel, and then settles onto them.
    catch_gate_height = 1.2
    catch_gate_speed = 0.5
    catch_settle_speed = 0.6
    catch_hold_max = 25.0         # s; then settle anyway
    catch_hold_tilt_deg = 4.0     # tilt budget about the wind trim while holding
    catch_integral_radius = 1.0   # m: the hold's integral only builds up this close to the axis
    # Wind profile for the predictions: the wind measured now, extrapolated
    # to other heights with a log-law profile whose shear strength k (as in
    # vehicle.Wind: 0 = the same at every height, 1 = the standard log law)
    # is fitted to the winds measured on the way down; with little height
    # spread in the samples the prior k = 1 wins.
    wind_sample_period = 0.25     # s
    wind_memory = 60.0            # s, e-folding age of the samples
    wind_shear_prior_weight = 0.003   # (shapes of nearby k differ little: a stronger prior hides real shear)
    # Wind trim: the sideways thrust acceleration that cancels the aerodynamic
    # force on the upright booster (lean into the wind).  The tracker's tilt
    # budget and the planner's tilt cone are both centred on it (fading to
    # upright over the last TRIM_FADE metres).
    trim_max_deg = 12.0
    trim_filter = 0.5             # s
    hold_bias_filter = 1.0        # s (_update_hold_bias)
    aoa_moment_margin = 0.8       # share of the gimbal's moment the aerodynamic moment may take (_aoa_table)
    aoa_refresh = 2.0             # s between recomputations of the angle-of-attack table
    q_filter = 1.0                # s, smoothing of the dynamic pressure for that limit
    # attitude loop
    k_angle = 2.0
    k_rate = 4.0
    max_rate = math.radians(30.0)

    def __init__(self, aero_model=None) -> None:
        self.planner = Planner3D()
        self.aero = aero_model
        self.set_target()
        self.reset()

    def set_target(self, base_z=None, rest_height=0.0, legs=True, obstacles=(), catch_link=None) -> None:
        """Where the burn ends.  Default: standing on the pad (legs).  For a
        tower catch, ``base_z`` is the world height of the booster's base when
        its catch collar rests on the arms; the whole guidance then works in
        heights above that point (``h_off`` above the pad altitude) and the
        legs stay stowed.  ``obstacles`` are keep-out boxes for the base point
        in world coordinates (``(centre, axes, half)``, e.g. the catch tower);
        the landing burn plans around them.  ``catch_link`` (a tower catch)
        reports whether the arms are closed round the barrel: the burn then
        ends in a hold just above the arms (``_catch_hold``)."""
        self.h_off = 0.0 if base_z is None else float(base_z) - float(rest_height)
        self.use_legs = legs
        self.catch_link = catch_link
        cls = type(self)
        self.gate_height = cls.catch_gate_height if catch_link is not None else cls.gate_height
        self.gate_speed = cls.catch_gate_speed if catch_link is not None else cls.gate_speed
        z0 = 0.0 if base_z is None else float(base_z)
        self._z_ground = -z0                 # the ground, in planner heights (base point)
        self.obstacles = [KeepOutBox(np.asarray(c, float) - np.array([0.0, 0.0, z0]), np.asarray(ax, float), np.asarray(hf, float))
                          for c, ax, hf in obstacles]

    def reset(self) -> None:
        self.elapsed = 0.0
        self.plan: Plan3D | None = None
        self.next_replan = 0.0
        self.next_predict = 0.0
        self.burn_locked = False
        self.touchdown_time: float | None = None
        self.burn_fraction = self.burn_fraction_nominal
        self.thrust_scale = 1.0       # measured / modelled engine thrust (accelerometer estimate)
        self._wind_ref = (np.zeros(3), 10.0)
        self._wind_samples: list = []   # (time, height, wind x, wind y)
        self._next_wind_sample = 0.0
        self._next_shear_fit = 0.0
        self.wind_shear_est = 1.0
        self._trim_xy = np.zeros(2)
        self._aoa_cache = (-1e9, Q_AOA_BUDGET)
        self._q_filt = 0.0
        self.solver_fail = None       # (time, status) of the last re-plan that found no solution at all
        self._hold_bias = np.zeros(2)
        self._route = None            # reference path round the obstacles while no plan gets through
        self._touched = False
        self.relax = 0                # (no relaxation levels any more; kept for the front end)
        self._next_search = 0.0
        self._search_fails = 0
        self._holding = False
        self._hold_h = 0.0
        self._hold_i = np.zeros(2)
        self._hold_t = 0.0
        self._secured = False
        self._dir_prev = None
        self._gate_xy = np.zeros(2)
        self._dir_rate = np.zeros(3)
        self._dt = 1.0 / 240.0
        self._ign_state = None
        self._brake_time = 0.0
        self.impact = None
        self.ignition_in = math.inf
        self.phase = "STANDBY"
        self.last_status = "idle"
        self.time_to_go = math.nan
        self.cmd_accel = np.zeros(3)
        self.cmd_throttle = 0.0
        self.cmd_axis = np.array([0.0, 0.0, 1.0])
        self.predicted_points: list = []
        self.visited: list = []

    # ------------------------------------------------------------ helpers
    def _state(self, veh):
        r = np.array([veh.pos[0], veh.pos[1], veh.altitude - self.h_off])
        return r, veh.vel.copy(), veh.mass

    def _height(self, veh) -> float:
        """Height of the base above the end of the burn (the pad or the catch point)."""
        return float(veh.altitude) - self.h_off

    def _drag_model(self, veh, wind_now, t_guess):
        """Aerodynamics along the previous plan (or a decelerating guess).

        Returns ``(d0, G)``: d0 is the drag acceleration with the booster
        aligned with the flow; ``G @ u`` adds the body/grid-fin normal force
        produced by tilting the thrust vector (and the vehicle) by
        ``u_perp / sigma_ref`` against the flow.  For an engines-first
        booster this force points opposite to the tilt, which is why a plan
        that ignored it would overshoot laterally at high dynamic pressure.
        """
        prev = self.plan if self.burn_locked else None
        now = self.elapsed
        r0, v0, m0 = self._state(veh)
        legs = max(veh.legs, 1.0 if veh.controls.legs_down else 0.0)
        s = veh.spec
        sig_now = max(3.0, veh.throttle * s.thrust_sl / m0)
        v_now2 = float(np.dot(v0, v0))

        def model(times):
            n = len(times)
            t_end = float(times[-1]) + (0.5 * float(times[1] - times[0]) if n > 1 else 0.0)
            t0 = now - prev.created_at if prev is not None else 0.0
            span = max(0.0, prev.t_f - t0) if prev is not None else 0.0
            out = np.zeros((n, 3))
            G = np.zeros((n, 3, 3))
            qd = np.zeros(n)
            wd = np.zeros((n, 3))
            dq = np.zeros(n)                # d(dynamic pressure)/d(height) along the reference
            Jv = np.zeros((n, 3, 3))        # d(drag accel)/d(velocity) about the reference
            vref = np.zeros((n, 3))         # the reference velocity it is linearised about
            lmax = np.zeros(n)              # glide lift available (engine off, glide_aoa_deg off engines-first)

            def damping(i, f, sp, u):
                # drag ~ -D u (u: unit airflow velocity), D ~ |v|^2: its
                # derivative in the velocity, so a plan flying faster or slower
                # than its reference sees the drag change (without it the drag
                # is that of the reference speed, and successive plans swing)
                D = max(0.0, -float(np.dot(f, u))) / m
                Jv[i] = -(D / sp) * (np.eye(3) + np.outer(u, u))
            if self.aero is None:
                return out, G, qd, wd, dq, Jv, vref, lmax
            for i, t in enumerate(times):
                u_ref = None
                coast = False
                if prev is None or t < 0.3:
                    T = max(t_guess, 0.5)
                    f = max(0.0, 1.0 - t / T)
                    # height consistent with that velocity (its integral),
                    # not below the gate: the old guess r0 + v0 t / 2 ran far
                    # underground on a long horizon
                    tt = min(float(t), T)
                    h = max(float(self.gate_height), float(r0[2] + v0[2] * (tt - 0.5 * tt * tt / T)))
                    v, m, sig = v0 * f, m0, max(sig_now, 0.6 * s.thrust_sl / m0)
                    if t < 0.3 and veh.throttle > 0.05:
                        u_ref = veh.last["thrust"] / m0
                else:
                    # the previous plan's remainder stretched over this
                    # horizon (a retimed plan follows the same path; at the
                    # same clock times a shorter one would see the old,
                    # slower, still-high part of it)
                    tau = t0 + span * t / t_end
                    pos, vel, acc = prev.sample(tau)
                    h, v = pos[2], vel
                    k = min(int(max(tau, 0.0) / prev.dt), prev.knots)
                    m = float(math.exp(prev.z[k]))
                    sig_ref = prev.sigma_at(tau)
                    sig = max(3.0, sig_ref)
                    u_ref = acc
                    if sig_ref < self.coast_accel:
                        coast = True
                # No faster than falling there from here without drag: a
                # reference from a plan that missed can carry absurd speeds,
                # and the drag about them blows the next plan up.
                v = np.asarray(v, float)
                v_cap = math.sqrt(v_now2 + 2.0 * G0 * max(0.0, float(r0[2]) - float(h))) + 5.0
                vn = float(np.linalg.norm(v))
                if vn > v_cap:
                    v = v * (v_cap / vn)
                rho, _, _, a = atmosphere(max(0.0, h + self.h_off))
                vr = v - self._wind(h)          # the wind where the plan will be, not the wind here
                sp = float(np.linalg.norm(vr))
                qd[i] = 0.5 * rho * sp * sp
                # lower down the wind is weaker (and the air a little denser):
                # how the dynamic pressure would change with the height
                qs = []
                for dh in (5.0, -5.0):
                    rho_d = atmosphere(max(0.0, h + dh + self.h_off))[0]
                    qs.append(0.5 * rho_d * float(np.linalg.norm(v - self._wind(h + dh))) ** 2)
                dq[i] = (qs[0] - qs[1]) / 10.0
                vref[i] = v
                if sp < 3.0:
                    continue
                u = vr / sp
                wd[i] = -u
                base = -u
                # glide lift: the body tilted glide_aoa_deg off engines-first
                f_b = self.aero.quick_force(vr, base, rho, a, legs)
                e_g = np.cross(u, [0.0, 0.0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0.0, 0.0])
                e_g /= np.linalg.norm(e_g)
                al = math.radians(self.glide_aoa_deg)
                df = self.aero.quick_force(vr, math.cos(al) * base + math.sin(al) * e_g, rho, a, legs) - f_b
                lmax[i] = float(np.linalg.norm(df - u * float(np.dot(df, u)))) / m
                if coast:
                    # the reference coasts: engine off, engines-first into the
                    # flow (as the tracker flies it) -- aligned drag, and no
                    # body lift from tilting a thrust that is not there
                    f0 = f_b
                    out[i] = f0 / m
                    damping(i, f0, sp, u)
                    continue
                # Linearise around the reference attitude (successive
                # linearisation: each replan uses the previous plan's tilt).
                tilt = np.zeros(3)
                broadside = None
                if u_ref is not None and float(np.linalg.norm(u_ref)) > 2.0:
                    d_ref = u_ref / np.linalg.norm(u_ref)
                    tilt = d_ref - base * float(np.dot(d_ref, base))
                    if float(np.dot(d_ref, base)) < 0.2:
                        # Flying (nearly) across the flow, e.g. a low divert:
                        # the body is broadside and its drag is large.  Take
                        # the force at that attitude, without a lift gradient
                        # (no useful linearisation that far from the flow).
                        tilt = np.zeros(3)
                        broadside = d_ref
                axis0 = base + tilt
                axis0 /= np.linalg.norm(axis0)
                if broadside is not None:
                    f0 = self.aero.quick_force(vr, broadside, rho, a, legs)
                    out[i] = f0 / m
                    damping(i, f0, sp, u)
                    continue
                f0 = self.aero.quick_force(vr, axis0, rho, a, legs)
                e1 = np.cross(u, [0.0, 0.0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0.0, 0.0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(u, e1)
                d = math.radians(2.0)
                Gi = np.zeros((3, 3))
                for e in (e1, e2):
                    ax1 = axis0 + d * e
                    f1 = self.aero.quick_force(vr, ax1 / np.linalg.norm(ax1), rho, a, legs)
                    grad = (f1 - f0) / d / m                  # accel per radian of tilt along e
                    # Keep the net effect of a tilt in the thrust direction
                    # (limits over-reliance on a local linearisation).
                    gn = float(np.linalg.norm(grad))
                    if gn > self.lift_gradient_cap * sig:
                        grad *= self.lift_gradient_cap * sig / gn
                    Gi += np.outer(grad, e) / sig             # tilt along e ~ (e . u) / sigma
                out[i] = f0 / m - (Gi @ u_ref if u_ref is not None and np.any(tilt) else 0.0)
                damping(i, f0, sp, u)
                G[i] = Gi
            return out, G, qd, wd, dq, Jv, vref, lmax

        return model

    def _keep_out(self, veh, path=None):
        """Keep-out half-spaces for the planner, linearised along the previous
        plan if that one made it to the gate (successive convexification),
        otherwise along a reference path: ``path``, or the route chosen by
        ``_routes`` (first plan, or one stuck behind an obstacle, which would
        only reproduce itself)."""
        if not self.obstacles:
            return None
        now = self.elapsed
        boxes = self.obstacles
        prev = self.plan if self.burn_locked else None
        if path is not None or (prev is not None and not prev.reaches_pad):
            prev = None
        if prev is None and path is None:
            path = self._route if self._route is not None else detour_path(
                boxes, self._state(veh)[0], np.zeros(3), z_min=self._z_ground + 1.0)

        def planes(t_knots):
            t_end = max(float(t_knots[-1]), 1e-6)
            if prev is not None:
                # the previous path stretched over this horizon (see _drag_model)
                t0 = now - prev.created_at
                span = max(0.0, prev.t_f - t0)
                pts = [prev.sample(t0 + span * float(t) / t_end)[0] for t in t_knots]
            else:
                pts = along_path(path, [float(t) / t_end for t in t_knots])
            return separating_planes(boxes, pts, target=np.zeros(3))

        return planes

    def _height_ref(self, veh):
        """Reference heights at the knot times (for the tilt budget): the
        previous plan stretched over the new horizon, else a straight descent."""
        prev = self.plan if self.burn_locked else None
        now = self.elapsed
        h0 = float(self._state(veh)[0][2])

        def heights(t_knots):
            t_end = max(float(t_knots[-1]), 1e-6)
            if prev is None:
                return [h0 + (self.gate_height - h0) * float(t) / t_end for t in t_knots]
            t0 = now - prev.created_at
            span = max(0.0, prev.t_f - t0)
            return [float(prev.sample(t0 + span * float(t) / t_end)[0][2]) for t in t_knots]

        return heights

    def _dir_ref(self, veh):
        """Reference thrust direction at the knot times (for the planner's
        slew limit): the previous plan's, stretched over the new horizon like
        the heights; at ignition (no plan yet) the body axis."""
        prev = self.plan if self.burn_locked else None
        now = self.elapsed
        axis = np.asarray(veh.axis, float)

        def dirs(t_knots):
            if prev is None:
                return np.tile(axis, (len(t_knots), 1))
            t_end = max(float(t_knots[-1]), 1e-6)
            t0 = now - prev.created_at
            span = max(0.0, prev.t_f - t0)
            out = []
            for t in t_knots:
                u = np.asarray(prev.sample(t0 + span * float(t) / t_end)[2], float)
                n = float(np.linalg.norm(u))
                out.append(u / n if n > 1e-6 else axis)
            return np.asarray(out)

        return dirs

    def _vel_ref(self, veh):
        """Reference velocities at the knot times (trust region of the
        successive convexification): the plan in hand, stretched over the new
        horizon like the drag reference.  None before there is a plan."""
        prev = self.plan if self.burn_locked else None
        if prev is None:
            return None
        now = self.elapsed

        def vels(t_knots):
            t_end = max(float(t_knots[-1]), 1e-6)
            t0 = now - prev.created_at
            span = max(0.0, prev.t_f - t0)
            return np.asarray([prev.sample(t0 + span * float(t) / t_end)[1] for t in t_knots])

        return vels

    def _config(self, veh, frac, keep_out=None):
        s = veh.spec
        return PlanConfig3D(
            thrust_min=0.0,                  # the engine may be off: coasting is the optimiser's choice
            thrust_max=frac * s.thrust_sl * self.thrust_scale, isp=s.isp_sl,
            dry_mass=s.dry_mass + 150.0, sigma_now=veh.throttle * s.thrust_sl / veh.mass,
            u_now=tuple(veh.last["thrust"] / veh.mass), engine_on=True,
            u_cmd=tuple(self.plan.sample(self.elapsed - self.plan.created_at)[2]) if self.plan is not None else None,
            max_tilt_deg=self.divert_tilt_deg,
            touchdown_speed=self.gate_speed, target_alt=self.gate_height, target_xy=tuple(self._gate_xy),
            sink_gain=self.sink_gain, keep_out=keep_out, q_aoa=self._aoa_table(veh), ground_alt=self._z_ground,
            body_length=s.length,
            tilt_trim=tuple(self._gate_trim(veh)), trim_fade=self._trim_fade(), tilt_budget=TILT_BUDGET,
            height_ref=self._height_ref(veh), slew_deg=SLEW_DEG, dir_ref=self._dir_ref(veh), time_weight=self.time_weight,
            vel_ref=self._vel_ref(veh), trust_weight=self.trust_weight,
        )

    def _solver(self, veh, wind_now, frac, keep_out):
        """t_f -> plan from the current state (drag predicted for that horizon)."""
        r, v, m = self._state(veh)
        base = self._config(veh, frac, keep_out)

        def solve(t_f):
            if not t_f > 0.6:
                return None
            cfg = dataclasses.replace(base, drag_model=self._drag_model(veh, wind_now, t_f))
            # Two passes.  Free thrust first (0 .. max): where the optimiser
            # wants the engine on.  Then the burns it can really have with
            # the engine starts left (_windows) -- coasts at zero thrust,
            # burns at least at the minimum throttle -- and that plan is flown.
            free = self.planner.solve(r, v, m, t_f, cfg)
            if free is None:
                return None
            windows = self._windows(veh, free)
            t_min = self.minimum_throttle * veh.spec.thrust_sl * self.thrust_scale

            def constrained(win):
                q = self.planner.solve(r, v, m, t_f, dataclasses.replace(cfg, burn_windows=win, thrust_min=t_min))
                if q is not None:
                    q.windows = win
                return q

            p = constrained(windows)
            left = veh.spec.max_ignitions - veh.ignitions
            if (p is not None and not veh.engine_lit and left == 1 and len(windows) == 1
                    and windows[0][0] < self.ignite_soon and t_f - windows[0][0] > 2.0 * self.ignite_delay):
                # The last start (the landing burn) is about to be used: only
                # if waiting would lose the gate (latest safe ignition) -- a
                # burn lit early must run to the end at no less than the
                # minimum throttle, and that is long and costly.
                later = constrained(((windows[0][0] + self.ignite_delay, t_f),))
                if later is not None and (later.reaches_pad or (not p.reaches_pad and self._miss(later) <= self._miss(p) + 1.0)):
                    p = later
            return p

        return solve

    def _windows(self, veh, free):
        """Engine burn windows for a plan, from the free-thrust plan and the
        engine starts left (VehicleSpec.max_ignitions): the landing burn is
        the free plan's last burn and runs to the end; a burn before it (the
        boost-back) only with a start to spare for it.  A burning engine with
        no start left burns to the end."""
        t = np.arange(free.knots + 1) * free.dt
        on = free.sigma > self.burn_sigma_frac * self.minimum_throttle * veh.spec.thrust_sl / veh.mass
        runs = []
        k = 0
        while k < len(on):
            if on[k]:
                j = k
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                runs.append((k, j))
                k = j + 1
            else:
                k += 1
        # a short dip of the thrust is no end of a burn (the engine would be
        # out, and a relight takes a start): runs this close are one burn
        merged = []
        for a, b in runs:
            if merged and (a - merged[-1][1]) * free.dt <= self.burn_gap_merge:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        runs = merged
        t_f = float(free.t_f)
        left = veh.spec.max_ignitions - veh.ignitions
        lit = bool(veh.engine_lit)
        if lit and left <= 0:
            return ((0.0, t_f),)                       # the last start is burning: to the end
        if not runs:
            return ((0.0, t_f),) if lit else ((max(0.0, t_f - 2.0), t_f),)
        land = (float(t[runs[-1][0]]), t_f)
        # the boost-back is the free plan's first burn, the landing burn its
        # last; a burn in between has no start to light it (dropped -- the
        # constrained plan does without it)
        if lit:
            # burning now with a start left: this burn may end (boost-back) --
            # but not on the word of a plan that misses the gate (cut short,
            # it leaves the booster short of the pad with only the landing
            # burn to come): then it burns on
            if not free.reaches_pad:
                return ((0.0, t_f),)
            if runs[0][0] == 0 and len(runs) > 1:
                return ((0.0, float(t[runs[0][1]])), land)
            return ((0.0, t_f),) if runs[0][0] == 0 else (land,)
        if left >= 2 and len(runs) > 1:
            return ((float(t[runs[0][0]]), float(t[runs[0][1]])), land)
        if left >= 1:
            return (land,)
        return ()

    @staticmethod
    def _thrust_available(veh, h):
        s = veh.spec
        _, p, _, _ = atmosphere(max(0.0, h))
        return s.thrust_vac - (s.thrust_vac - s.thrust_sl) * min(1.0, p / 101_325.0)

    # ------------------------------------------------------- predictions
    @staticmethod
    def _profile(z, k):
        """Log-law shear profile of strength k relative to the 10 m wind (vehicle.Wind)."""
        f = np.clip(np.log(np.maximum(0.5, z) / 0.05) / math.log(10.0 / 0.05), 0.25, 2.6)
        return np.maximum(0.05, 1.0 + k * (f - 1.0))

    def _wind(self, z):
        """Wind assumed at height ``z`` (planner coordinates) for the
        predictions: the wind measured now, scaled with the estimated profile."""
        w, z_now = self._wind_ref
        k = self.wind_shear_est
        shape = float(self._profile(z + self.h_off, k) / self._profile(z_now, k))
        return w * min(6.0, shape)

    def _update_trim(self, veh, wind_now, dt):
        """Filtered wind trim (sideways thrust acceleration that holds the
        upright booster against the aerodynamic force on it)."""
        if self.aero is None:
            return
        rho, _, _, a_s = atmosphere(max(0.0, float(veh.pos[2])))
        f = self.aero.quick_force(veh.vel - wind_now, np.array([0.0, 0.0, 1.0]), rho, a_s, veh.legs)
        target = -f[:2] / veh.mass
        cap = math.tan(math.radians(self.trim_max_deg)) * G0
        n = float(np.linalg.norm(target))
        if n > cap:
            target *= cap / n
        self._trim_xy += (target - self._trim_xy) * min(1.0, dt / self.trim_filter)

    def _update_hold_bias(self, veh, dt):
        """Filtered lateral thrust acceleration that cancels what the booster
        actually feels besides its commanded thrust (accelerometer: sensed
        specific force minus the command).  Unlike the modelled trim it
        includes the thrust lost to the gimbal holding the attitude against
        the wind's moment -- in a 15 m/s wind enough to push a fully trimmed,
        hovering booster off the catch axis at ~0.4 m/s^2.  Used over the gate
        (catch hold), where the commanded and the felt force are near steady."""
        sensed = (np.asarray(veh.last["thrust"], float) + np.asarray(veh.last["aero"], float))[:2] / veh.mass
        target = np.asarray(self.cmd_accel[:2], float) - sensed
        cap = math.tan(math.radians(self.trim_max_deg)) * G0
        n = float(np.linalg.norm(target))
        if n > cap:
            target *= cap / n
        self._hold_bias += (target - self._hold_bias) * min(1.0, dt / self.hold_bias_filter)

    def _trim_fade(self):
        """Heights (above the end of the burn) over which the wind trim fades
        to upright.  A tower catch does not end in a touchdown: the booster
        holds at the gate, leaning into the wind, and only comes upright once
        the arms have closed -- fading the trim over the last metres above the
        gate would let the wind push it off just before the hold."""
        return (-2.0, -1.0) if self.catch_link is not None else TRIM_FADE

    def _gate_trim(self, veh) -> np.ndarray:
        """Wind trim (sideways thrust acceleration, m/s^2) near the end of the
        burn in the predicted wind: the planner's tilt cone is centred on it."""
        if self.aero is None:
            return np.zeros(2)
        z = self.gate_height + 2.0
        rho, _, _, a_s = atmosphere(max(0.0, z + self.h_off))
        vr = np.array([0.0, 0.0, -self.gate_speed]) - self._wind(z)
        t = -self.aero.quick_force(vr, np.array([0.0, 0.0, 1.0]), rho, a_s, 1.0)[:2] / veh.mass
        cap = math.tan(math.radians(self.trim_max_deg)) * G0
        n = float(np.linalg.norm(t))
        return t * (cap / n) if n > cap else t

    def _update_wind_profile(self, veh, wind_now):
        """Record the measured wind and refit the shear strength (every 1 s)."""
        if self.elapsed >= self._next_wind_sample:
            self._next_wind_sample = self.elapsed + self.wind_sample_period
            self._wind_samples.append((self.elapsed, max(0.5, float(veh.altitude)), float(wind_now[0]), float(wind_now[1])))
            cut = self.elapsed - 3.0 * self.wind_memory
            while self._wind_samples and self._wind_samples[0][0] < cut:
                self._wind_samples.pop(0)
        if self.elapsed < self._next_shear_fit or len(self._wind_samples) < 8:
            return
        self._next_shear_fit = self.elapsed + 1.0
        a = np.asarray(self._wind_samples)
        wt = np.exp(-(self.elapsed - a[:, 0]) / self.wind_memory)
        w = a[:, 2:4]
        scale = float(np.sum(wt * np.sum(w * w, axis=1)) / np.sum(wt)) + 1e-6
        best = None
        for k in np.linspace(0.0, 3.0, 31):
            sh = self._profile(a[:, 1], k)
            c = (wt * sh) @ w / max(1e-9, float(np.sum(wt * sh * sh)))       # 10 m wind (2-vector)
            res = w - np.outer(sh, c)
            cost = float(np.sum(wt * np.sum(res * res, axis=1)) / np.sum(wt)) / scale
            cost += self.wind_shear_prior_weight * (k - 1.0) ** 2
            if best is None or cost < best[0]:
                best = (cost, float(k))
        self.wind_shear_est = best[1]

    # ------------------------------------------------------------- replan
    def _replan(self, veh, wind_now):
        """Re-plan from the current state at the time to go; every
        search_backoff a shorter time is tried (minimum time), and when the
        plan does not reach the gate longer times are searched until one does."""
        t_go = max(0.7, self.touchdown_time - self.elapsed)
        plan = self._solve_now(veh, wind_now, t_go)
        if self.elapsed >= self._next_search:
            if plan is None or not plan.reaches_pad:
                best = self._search_time(veh, wind_now, t_go)
                self._search_fails = 0 if best is not None and best.reaches_pad else min(self._search_fails + 1, 2)
                if best is not None:
                    plan = best
            else:
                d = max(0.3, self.retime_step * t_go)
                for t in (t_go - d, t_go + d):
                    p = self._solve_now(veh, wind_now, max(0.7, t))
                    if p is not None and p.reaches_pad and p.objective < plan.objective:
                        plan = p
                self._search_fails = 0
            self._next_search = self.elapsed + self.search_backoff * 2 ** self._search_fails
        self.last_status = self.planner.last_status
        if plan is not None and not plan.reaches_pad and self.plan is not None and self.elapsed - self.plan.created_at < 1.0 and (
                self.plan.reaches_pad or self._miss(plan) >= self._miss(self.plan)):
            # a re-plan that misses where the plan in hand still gets there
            # (or misses less) is not taken: it would also become the
            # reference the next plans are linearised about
            plan = None
            if self.touchdown_time - self.elapsed > 0.6:
                return
        if plan is not None:
            self._commit(plan)
        elif self.planner.last_status not in ("Solved", "AlmostSolved"):
            # no solution at all (numerical): fly on along the last plan
            self.solver_fail = (self.elapsed, self.planner.last_status)

    @staticmethod
    def _miss(p):
        return p.terminal_position_error + p.terminal_velocity_error + p.keep_out_violation + p.ground_violation

    def _solve_now(self, veh, wind_now, t_f, path=None):
        """One plan from the current state for final time t_f (references --
        drag, heights, thrust directions, obstacle faces -- from the plan in
        hand, or ``path`` round the obstacles)."""
        keep_out = self._keep_out(veh, path)
        return self._solver(veh, wind_now, self.burn_fraction, keep_out)(t_f)

    def solver_alert(self):
        """For the display: None, or (kind, status) -- 'fail' when the last
        re-plans found no solution (shown for 2 s), 'miss' when the plan in
        hand does not reach the gate."""
        if not self.burn_locked or self._holding:
            # holding at the gate (or after the burn) nothing is planned:
            # the plan in hand is stale and says nothing
            return None
        if self.solver_fail is not None and self.elapsed - self.solver_fail[0] < 2.0:
            return ("fail", self.solver_fail[1])
        p = self.plan
        if p is not None and self.elapsed - p.created_at < 1.5 and (
                p.terminal_position_error >= 1.0 or p.terminal_velocity_error >= 0.8
                or p.keep_out_violation >= 0.5 or p.ground_violation >= 0.5):
            # really short of the gate, or only through an obstacle / the
            # ground.  A plan that gets there leaning past the soft angle-of-
            # attack limit is re-searched (reaches_pad) but is not "cannot
            # reach": in a strong wind aloft that happens for a few seconds
            # after ignition while the linearisation settles.
            return ("miss", self.last_status)
        return None

    def _commit(self, plan):
        plan.created_at = self.elapsed
        self.plan = plan
        self.touchdown_time = self.elapsed + plan.t_f
        self.last_status = plan.status
        if plan.reaches_pad:
            self._route = None          # from now on along this plan

    def _search_time(self, veh, wind_now, t_guess):
        """Final time minimising J = propellant + time_weight t_f among plans
        that reach the gate (golden section); if none in the bracket reaches,
        longer times (x time_grow) until one does -- the search never settles
        on a plan that cannot get there.  At full thrust if the nominal budget
        reaches nowhere.  Returns the best reaching plan, or the nearest miss."""
        best = None
        nearest = None

        def value(p):
            nonlocal best, nearest
            if p is None:
                return math.inf
            if p.reaches_pad:
                if best is None or p.objective < best.objective:
                    best = p
                return p.objective
            miss = self._miss(p)
            if nearest is None or miss < nearest[0]:
                nearest = (miss, p)
            return p.objective + 1e4 + 100.0 * miss

        r, v, _ = self._state(veh)
        t_far = self._far_time(r, v)
        ceiling = 3.0 * t_far + 30.0
        for frac in dict.fromkeys((self.burn_fraction, 0.97)):
            self.burn_fraction = frac
            path = self._routes(veh, wind_now, frac, max(t_guess, 5.0)) if self.obstacles else None
            lo = max(1.0, 0.35 * min(t_guess, t_far))
            hi = 1.5 * max(t_guess, t_far) + 2.0
            golden_section_search(lambda t, path=path: _Scored(self._solve_now(veh, wind_now, t, path), value), lo, hi,
                                  self.time_search_evals)
            t = hi
            while best is None and t < ceiling:
                t = min(ceiling, t * self.time_grow + 2.0)
                value(self._solve_now(veh, wind_now, t, path))
            if best is not None:
                break
        if best is None:
            self.burn_fraction = self.burn_fraction_nominal
        return best if best is not None else (nearest[1] if nearest is not None else None)

    def _routes(self, veh, wind_now, frac, t_f):
        """Which way round the obstacles (over, round either side): one convex
        problem per candidate reference, the cheapest wins.  Returns the
        chosen reference path (None: no obstacles)."""
        if not self.obstacles:
            return None
        paths = detour_paths(self.obstacles, self._state(veh)[0], np.zeros(3), z_min=self._z_ground + 1.0)
        best = None
        for path in paths:
            p = self._solver(veh, wind_now, frac, self._keep_out(veh, path))(t_f)
            if p is not None and (best is None or p.objective < best[0]):
                best = (p.objective, path)
        self._route = best[1] if best is not None else paths[0]
        return self._route

    def _far_time(self, r, v):
        """A generous time to the gate from here: a vertical fall plus a
        sideways transfer at a modest acceleration."""
        h, vz = max(0.0, float(r[2]) - self.gate_height), float(v[2])
        t_fall = (vz + math.sqrt(vz * vz + 2.0 * G0 * h)) / G0
        d = float(np.linalg.norm(r[:2] - self._gate_xy))
        return t_fall + 2.0 * math.sqrt(d / 2.0) + float(np.linalg.norm(v[:2])) / 3.0

    def _publish(self):
        p = self.plan
        if p is None:
            return
        start = max(0.0, self.elapsed - p.created_at)
        n = max(2, min(200, int(max(0.0, p.t_f - start) / 0.1) + 2))
        pts = []
        for t in np.linspace(start, p.t_f, n):
            pos, _, _ = p.sample(float(t))
            pts.append((float(pos[0]), float(pos[1]), float(max(0.0, pos[2])) + self.h_off))
        self.predicted_points = pts

    # ------------------------------------------------------------ command
    def command(self, veh, wind_now, dt):
        c = veh.controls
        if veh.state in ("CRASHED", "LANDED", "CAUGHT"):
            c.throttle = 0.0
            c.rcs = (0.0, 0.0, 0.0)
            c.gimbal = (0.0, 0.0)
            c.fins = (0.0, 0.0, 0.0)
            self.phase = "LANDED" if veh.state in ("LANDED", "CAUGHT") else "STANDBY"
            self._mark()
            return
        if veh.state == "ON PAD":
            c.throttle = 0.0
            self.phase = "STANDBY"
            return
        self.elapsed += dt
        self._dt = dt
        self._wind_ref = (np.asarray(wind_now, float).copy(), max(0.5, float(veh.altitude)))
        self._update_wind_profile(veh, wind_now)
        s = veh.spec
        r, v, m = self._state(veh)
        g = G0
        h = r[2]

        # Touchdown: cut the engine as soon as a foot is on the ground (or the
        # catch collar is on the arms).
        if veh.feet_contact > 0 or veh.catch_contact > 0 or self._touched:
            self._touched = True
            c.throttle = 0.0
            self.phase = "LANDING BURN"
            self._attitude(veh, np.array([0.0, 0.0, 1.0]), use_rcs=True)
            self._mark()
            return

        if h < self.legs_deploy_height and self.use_legs:
            c.legs_down = True
        c.fins_deploy = True
        self._update_trim(veh, wind_now, dt)
        self._update_hold_bias(veh, dt)
        # dynamic pressure for the angle-of-attack limit, smoothed: the
        # booster's own sway and the gusts swing it by +-50 % in a strong
        # cross wind, and the limit (steep in q there) would flick on and off
        self._q_filt += (float(veh.last.get("q", 0.0)) - self._q_filt) * min(1.0, dt / self.q_filter)

        if not self.burn_locked:
            # autopilot just switched on: plan the whole recovery from here
            self._engage(veh, wind_now)

        # Engine performance from the measured thrust (accelerometer): a
        # weaker-than-modelled engine shrinks the planner's thrust limits, so
        # the plan does not count on braking it cannot deliver.
        t_model = veh.throttle * self._thrust_available(veh, h)
        if veh.throttle > 0.3 and t_model > 0.0:
            ratio = float(np.clip(np.linalg.norm(veh.last["thrust"]) / t_model, 0.6, 1.1))
            self.thrust_scale += (ratio - self.thrust_scale) * min(1.0, dt / 0.5)

        t_go = self.touchdown_time - self.elapsed
        # One burn all the way to leg contact: re-plan until the last ~0.6 s,
        # then fly the final plan.  Past its end the plan's reference is
        # "on the pad, sinking at touchdown speed", so a late vehicle keeps
        # descending instead of hovering.
        # Past the plan's end but still high or off the pad (the plan could
        # not be tracked): re-plan from here instead of trusting the old end.
        if self.catch_link is not None and (self._holding or (
                (t_go <= 0.6 or h <= self.gate_height + 0.2) and h < self.gate_height + 5.0
                and float(np.linalg.norm(r[:2])) < 6.0)):
            self._catch_hold(veh, r, v, dt)
            return
        stranded = t_go <= 0.6 and (h > 20.0 or float(np.linalg.norm(r[:2])) > 15.0)
        if self.planner.available and self.elapsed >= self.next_replan and (t_go > 0.6 or stranded):
            self._replan(veh, wind_now)
            self._publish()
            t_go = self.touchdown_time - self.elapsed
            # Long burns re-plan less often early on (each solve of a long,
            # 70-knot horizon costs ~0.1 s); 5 Hz over the last ~8 s.
            self.next_replan = self.elapsed + float(np.interp(t_go, [8.0, 30.0], [self.burn_replan_period, 0.6]))
        self.time_to_go = max(0.0, t_go)
        if t_go < 8.0 and self.use_legs:
            c.legs_down = True

        if self.plan is not None and self._coasting(veh, v, wind_now):
            return
        floor = None
        if self.plan is not None:
            tau = self.elapsed - self.plan.created_at
            p_ref, v_ref, u_ff = self.plan.sample(tau)
            w = self._window_at(self.plan, tau)
            boost = w is not None and w >= 0 and w < len(self.plan.windows) - 1
            self.phase = "BOOSTBACK" if boost else "LANDING BURN"
            c.fins_deploy = not boost               # grid fins stowed for the boost-back
            if w is not None and w >= 0:
                floor = self.minimum_throttle       # inside a burn: never let the engine go out
            fb = 0.5 * (p_ref - r) + 1.6 * (v_ref - v)
            nfb = float(np.linalg.norm(fb))
            if nfb > 5.0:
                fb *= 5.0 / nfb
            accel = u_ff + fb
            if h < self.sink_track_height:
                # Final approach envelope (same as the planner's): if sinking
                # faster than 1.2 + 0.5 h m/s, brake the excess.  Near the pad
                # the sink rate settles at touchdown speed with thrust ~ weight,
                # so cutting the engine at contact cannot bounce the vehicle.
                v_lim = -(self.gate_speed + self.sink_gain * max(0.0, h))
                if v[2] < v_lim:
                    accel[2] += 2.0 * (v_lim - v[2])
            # Tilt budget tightens with height (upright at contact) and with
            # dynamic pressure: there the body lift of an engines-first
            # booster opposes its tilt and outweighs the thrust's sideways
            # part, so more tilt would push the wrong way (and feed back).
            max_tilt = math.radians(float(np.interp(h, *TILT_BUDGET)))
            center = self._trim_xy * float(np.interp(h, self._trim_fade(), (0.0, 1.0)))
            trim_tilt = math.atan2(float(np.linalg.norm(center)), G0)
            vr = v - wind_now
            sp = float(np.linalg.norm(vr))
            an = float(np.linalg.norm(accel))
            if boost and sp > self.nose_first_speed and an > 1e-6 and float(np.dot(accel, vr)) > 0.0:
                # nose-first boost-back (fins stowed): thrust within
                # nose_first_aoa_deg of the motion, as the plan has it
                u_air = vr / sp
                d = accel / an
                cos_f = float(np.dot(d, u_air))
                cap = math.radians(self.nose_first_aoa_deg)
                if cos_f < math.cos(cap):
                    side = d - u_air * cos_f
                    d = u_air * math.cos(cap) + side / max(1e-9, float(np.linalg.norm(side))) * math.sin(cap)
                accel = d * an
                max_tilt = math.radians(89.0)
                center = None
            else:
                accel = self._aoa_limit(veh, accel, wind_now, max_tilt + trim_tilt, self._aoa_table(veh), self._q_filt)
        else:
            self.phase = "FALLBACK"
            a_max_now = s.thrust_sl / m
            vz_ref = -math.sqrt(2 * 0.5 * (a_max_now - g) * max(h, 0.0)) - self.touchdown_speed
            accel = np.array([-0.08 * r[0] - 0.9 * v[0], -0.08 * r[1] - 0.9 * v[1], g + 1.5 * (vz_ref - v[2])])
            max_tilt = math.radians(35.0)
            center = None
        self._mark()
        self._fly_accel(veh, accel, max_tilt, floor=floor, center=center)

    @staticmethod
    def _window_at(p, tau):
        """Index of the plan's burn window at tau (past the end: the landing
        burn); None when coasting; -1 for a plan without windows."""
        if p.windows is None:
            return -1
        if p.windows and tau >= p.windows[-1][0] - 1e-6:
            return len(p.windows) - 1
        for i, (a, b) in enumerate(p.windows):
            if a - 1e-6 <= tau <= b + 1e-6:
                return i
        return None

    def _coasting(self, veh, v, wind_now) -> bool:
        """Where the plan coasts (thrust below coast_accel), fly it with the
        engine off: engines-first into the flow (what the plan's aerodynamics
        assume), turning to the next burn's thrust direction up to
        burn_lookahead before it.  Returns False when the plan is thrusting."""
        p = self.plan
        tau = self.elapsed - p.created_at
        w = self._window_at(p, tau)
        if w is None:
            pass                                    # between burns
        elif w >= 0 or tau >= p.t_f or p.sigma_at(tau) >= self.coast_accel:
            return False
        c = veh.controls
        c.throttle = 0.0
        c.fins_deploy = True
        self.cmd_throttle = 0.0
        self.cmd_accel = np.zeros(3)
        self.phase = "COAST"
        self._mark()
        vr = np.asarray(v, float) - np.asarray(wind_now, float)
        sp = float(np.linalg.norm(vr))
        # next burn soon: turn to its thrust direction
        nxt = None
        if p.windows:
            starts = [a for a, _ in p.windows if a > tau]
            if starts and starts[0] - tau <= self.burn_lookahead:
                nxt = min(p.t_f, starts[0] + 0.5 * p.dt)
        if nxt is not None:
            u = np.asarray(p.sample(nxt)[2], float)
            n = float(np.linalg.norm(u))
            target = None if n < 1e-6 else self._aoa_limit(veh, u / n * G0, wind_now, math.radians(self.divert_tilt_deg),
                                                            self._aoa_table(veh), self._q_filt)
            if target is not None:
                target = target / max(1e-9, float(np.linalg.norm(target)))
        elif sp > 20.0:
            # the glide: the planned lift, by a small angle of attack
            k = min(int(max(tau, 0.0) / p.dt), p.knots - 1)
            a_des = np.zeros(3) if p.lift is None else np.asarray(p.lift[k], float)
            target = self._glide_attitude(veh, vr, a_des)
        else:
            target = None
        if target is None:
            target = -vr / sp if sp > 20.0 else np.array([0.0, 0.0, 1.0])
        if target[2] < 0.0 and sp <= 20.0:
            target = np.array([0.0, 0.0, 1.0])
        self.cmd_axis = target
        self._dir_prev = target.copy()
        self._attitude(veh, target, use_rcs=True)
        return True

    def _glide_attitude(self, veh, vr, a_des):
        """Attitude that gives the lift acceleration ``a_des`` (across the
        flow): engines-first, tilted by the angle of attack (secant on the
        aerodynamic model's lift curve, up to glide_aoa_deg)."""
        sp = float(np.linalg.norm(vr))
        u = vr / sp
        base = -u
        a_des = a_des - u * float(np.dot(a_des, u))
        an = float(np.linalg.norm(a_des))
        if an < 0.02 or self.aero is None:
            return base
        e_a = a_des / an
        rho, _, _, a_snd = atmosphere(max(0.0, float(veh.pos[2])))
        l0 = float(np.dot(self.aero.quick_force(vr, base, rho, a_snd, veh.legs), e_a))
        a_max = math.radians(self.glide_aoa_deg)
        m = veh.mass

        def lift(alpha, e):
            axis = math.cos(alpha) * base + math.sin(alpha) * e
            return float(np.dot(self.aero.quick_force(vr, axis, rho, a_snd, veh.legs), e_a)) - l0

        # Engines-first, the lift points against the tilt: find the side that
        # pushes along e_a, then the angle.
        alpha = math.radians(5.0)
        e, l_a = max(((e, lift(alpha, e)) for e in (-e_a, e_a)), key=lambda x: x[1])
        if l_a <= 0.0:
            return base
        for _ in range(2):
            alpha = float(np.clip(alpha * an * m / l_a, 0.0, a_max))
            if alpha >= a_max:
                break
            l_a = lift(alpha, e)
            if l_a <= 0.0:
                break
        return math.cos(alpha) * base + math.sin(alpha) * e

    def _catch_hold(self, veh, r, v, dt):
        """Tower catch: hold the collar just above the arm tops, centred on the
        catch axis, until the tower reports the arms closed round the barrel;
        then settle onto them (the engine is cut at contact).  The arms close
        only on a booster that is in their capture window and slow, so a
        booster that arrives a little off is caught after it has centred,
        instead of drifting past while they close."""
        if not self._holding:
            self._holding = True
            self._hold_t = 0.0
            self._hold_i = np.zeros(2)
            self._hold_h = self.gate_height
            self._secured = False
        self._hold_t += dt
        # latched: the arms touching the barrel can let go for a moment as
        # they push it to the centre; flipping between holding (leaning into
        # the wind) and settling (upright) on that rocks the booster
        self._secured = self._secured or bool(self.catch_link())
        settle = self._secured or self._hold_t > self.catch_hold_max or veh.propellant < 80.0
        if settle:
            vz_ref = -self.catch_settle_speed
            self._hold_h -= self.catch_settle_speed * dt
        else:
            vz_ref = 0.0
        e = -np.asarray(r[:2], float)
        if float(np.linalg.norm(e)) < self.catch_integral_radius:
            # integrate only near the axis: while it is still coming in from
            # metres off (arrival overshoot) the sum would wind up and carry
            # it past the axis, out of the capture window on the other side
            self._hold_i = np.clip(self._hold_i + e * dt, -4.0, 4.0)
        # settling: the closed arms hold it sideways -- gentle and upright
        k_xy = 0.5 if settle else 1.0
        tilt = math.radians(1.0 if settle else self.catch_hold_tilt_deg)
        # Towards the axis no faster than it can stop with half the tilt
        # budget: a saturated linear PD brakes too late and swings metres
        # past the axis in a strong wind.  Near the axis this is the plain PD
        # (0.6 e - 1.6 v).
        en = float(np.linalg.norm(e))
        v_ref = np.zeros(2)
        if en > 1e-6:
            v_ref = e / en * min(0.375 * en, math.sqrt(math.tan(tilt) * G0 * en))
        a_xy = k_xy * (1.6 * (v_ref - v[:2]) + 0.1 * self._hold_i)
        a_z = G0 + 1.2 * (self._hold_h - r[2]) + 2.0 * (vz_ref - v[2])
        self.phase = "CATCH HOLD"
        self.time_to_go = max(0.0, float(r[2]) / self.catch_settle_speed) if settle else math.nan
        self._mark()
        # Holding: lean into the wind (else it drifts out of the capture
        # window and the arms never close), centred on the measured bias.
        # Settling on closed arms: upright, they hold it sideways.
        secured = self._secured
        self._fly_accel(veh, np.array([a_xy[0], a_xy[1], a_z]), tilt, floor=0.9 * self.minimum_throttle,
                        center=None if secured else self._hold_bias)

    def _aoa_table(self, veh):
        """Angle-of-attack limit vs dynamic pressure, from physics: the
        largest angle between the thrust (body axis) and the oncoming flow,
        counted from engines-first, at which the aerodynamic moment stays
        within ``aoa_moment_margin`` of what the controls can hold: the
        gimbal (thrust ~ weight, at the current mass and centre of mass) and
        the four grid fins at full deflection (each fin in its own local
        flow, so their moment grows with the dynamic pressure like the
        aerodynamic one).  Both moments are taken from the models (per Pa at
        each angle, refreshed every ``aoa_refresh``).  A hovering booster in a cross wind leans into
        it at ~90-110 deg from the flow: what limits it there is the dynamic
        pressure, hardly the lean (leaning 20 deg adds ~20 % moment), and
        near the ground, where the wind is weaker, it can lean much more."""
        t_last, table = self._aoa_cache
        if self.aero is None or self.elapsed - t_last < self.aoa_refresh:
            return table
        s = veh.spec
        m, zc, _, _ = veh.mass_properties()
        m_gim = m * G0 * math.sin(s.gimbal_limit) * max(0.5, zc - s.gimbal_z)
        rho, _, _, a_s = atmosphere(max(0.0, float(veh.pos[2])))
        q_ref = 1000.0
        sp = math.sqrt(2.0 * q_ref / rho)
        last = getattr(self.aero, "last", None)
        angles = np.arange(0.0, 181.0, 10.0)
        k, k_fin = [], []
        w = np.array([1.0, 0.0, 0.0])                     # where the air goes, relative to the booster
        patterns = [np.array(p, float) * s.fin_max_defl for p in ((1, 0, -1, 0), (0, 1, 0, -1), (1, 1, -1, -1), (1, -1, -1, 1))]
        for ang in angles:
            a = math.radians(float(ang))
            axis = np.array([math.cos(a), 0.0, math.sin(a)])
            # rotation taking the body z axis to ``axis`` (about body y)
            R = np.array([[math.sin(a), 0.0, math.cos(a)], [0.0, 1.0, 0.0], [-math.cos(a), 0.0, math.sin(a)]])
            assert np.allclose(R[:, 2], axis)
            _, mom, _ = self.aero(_AttitudeProbe(veh, R), -w * sp, rho, a_s, veh.com_world())
            k.append(float(np.linalg.norm(mom)) / q_ref)
            # what the four fins can add (pitch/yaw, best deflection pattern)
            _, m0, _ = grid_fin_forces(s, R, zc, np.zeros(3), -w * sp, rho, a_s, np.zeros(4), 1.0)
            best = 0.0
            for d in patterns:
                _, m1, _ = grid_fin_forces(s, R, zc, np.zeros(3), -w * sp, rho, a_s, d, 1.0)
                dm = m1 - m0
                best = max(best, float(np.linalg.norm(dm - axis * float(np.dot(dm, axis)))))
            k_fin.append(best / q_ref)
        if last is not None:
            self.aero.last = last
        # highest dynamic pressure each angle can be held at:
        #   K q <= margin (M_gimbal + K_fin q)
        mg = self.aoa_moment_margin
        excess = np.asarray(k) - mg * np.asarray(k_fin)
        q_max = np.where(excess > 0.0, mg * m_gim / np.maximum(excess, 1e-12), np.inf)
        q_max = np.minimum.accumulate(q_max)                # the cone from engines-first must hold throughout
        out = []
        for q in AOA_Q_GRID:
            ok = q_max >= q
            if ok.all():
                out.append(180.0)
            elif not ok[0]:
                out.append(0.0)
            else:
                j = int(np.argmin(ok))                      # first angle that cannot be held
                # interpolate in 1/q_max between the last good and first bad angle
                a0, a1 = angles[j - 1], angles[j]
                r0, r1 = 1.0 / q_max[j - 1], 1.0 / q_max[j]
                out.append(float(a0 + (a1 - a0) * (1.0 / q - r0) / max(r1 - r0, 1e-12)))
        table = (AOA_Q_GRID, tuple(out))
        self._aoa_cache = (self.elapsed, table)
        return table

    @staticmethod
    def _aoa_limit(veh, accel, wind_now, max_tilt, table=Q_AOA_BUDGET, q_dyn=None):
        """Keep the commanded thrust within the angle-of-attack limit of the
        oncoming flow (``table``: dynamic pressure -> deg, normally
        ``_aoa_table``), but never beyond the tilt cone."""
        vr = veh.vel - wind_now
        sp = float(np.linalg.norm(vr))
        q_dyn = float(veh.last.get("q", 0.0)) if q_dyn is None else float(q_dyn)
        an = float(np.linalg.norm(accel))
        if sp < 3.0 or an < 1e-6 or q_dyn <= table[0][0]:
            return accel
        w = -vr / sp
        aoa = math.radians(float(np.interp(q_dyn, *table)))
        if aoa >= math.radians(179.0):
            return accel
        aoa = max(aoa, math.acos(float(np.clip(w[2], -1.0, 1.0))) - max_tilt + math.radians(3.0))
        d = accel / an
        ang = math.acos(float(np.clip(np.dot(d, w), -1.0, 1.0)))
        if ang <= aoa:
            return accel
        if w[2] < 0.0 and accel[2] > 0.0:
            # Rising through the air (a climb, or a crosswind stronger than
            # the sink): the flow comes from above, so no upward thrust is
            # "engines-first", and turning towards the flow would only add a
            # push downwind -- in a strong crosswind the booster then drifts
            # off with the wind.  Give up horizontal thrust only: the largest
            # share of the commanded horizontal part within the budget.
            lo, hi = 0.0, 1.0
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                t = np.array([accel[0] * mid, accel[1] * mid, accel[2]])
                if math.acos(float(np.clip(np.dot(t / np.linalg.norm(t), w), -1.0, 1.0))) <= aoa:
                    lo = mid
                else:
                    hi = mid
            # ... and what it gives up does not go into a climb: rising
            # already, at most the hover vertical acceleration (a climb in a
            # strong wind only meets a stronger one)
            az = min(float(accel[2]), G0) if float(veh.vel[2]) >= 0.0 else float(accel[2])
            return np.array([accel[0] * lo, accel[1] * lo, az])
        side = d - w * float(np.dot(d, w))
        sn = float(np.linalg.norm(side))
        if sn < 1e-9:
            return accel
        d_new = w * math.cos(aoa) + side / sn * math.sin(aoa)
        if d_new[2] > 0.2 and accel[2] > 0.0:
            # Keep the commanded vertical acceleration: the limit only gives
            # up horizontal thrust (with a sideways flow, turning the thrust
            # towards it would otherwise turn lost push into lift -- a climb).
            return d_new * float(accel[2]) / float(d_new[2])
        # the part of the command along the allowed direction
        return d_new * an * max(0.0, math.cos(ang - aoa))

    def _fly_accel(self, veh, accel, max_tilt, floor=None, align_gate=0.35, center=None):
        """Fly an acceleration command.  The horizontal part is kept within
        ``max_tilt`` of ``center`` (the wind trim, or upright)."""
        c = veh.controls
        m = veh.mass
        horiz = accel[:2].copy()
        az = max(float(accel[2]), 1e-3)
        c0 = np.zeros(2) if center is None else np.asarray(center, float) * (az / G0)
        off = horiz - c0
        on = float(np.linalg.norm(off))
        lim = math.tan(max_tilt) * az
        if on > lim:
            horiz = c0 + off * (lim / on)
        accel = np.array([horiz[0], horiz[1], az])
        self.cmd_accel = accel
        mag = float(np.linalg.norm(accel))
        d = accel / mag
        # Slew-rate limit on the commanded thrust direction: a vehicle this
        # size cannot follow direction jumps between replans, and chasing
        # them is what makes it rock.  Slower near the ground.
        if self._dir_prev is not None:
            slew = math.radians(float(np.interp(self._height(veh), *SLEW_DEG))) * self._dt
            ang = math.acos(max(-1.0, min(1.0, float(np.dot(self._dir_prev, d)))))
            if ang > slew:
                axis_r = np.cross(self._dir_prev, d)
                an = float(np.linalg.norm(axis_r))
                if an > 1e-9:
                    axis_r /= an
                    d = self._dir_prev * math.cos(slew) + np.cross(axis_r, self._dir_prev) * math.sin(slew)
                    d /= np.linalg.norm(d)
        # Angular-rate feed-forward of the (slew-limited) commanded direction.
        self._dir_rate = np.zeros(3)
        if self._dir_prev is not None and self._dt > 0:
            wv = np.cross(self._dir_prev, d) / self._dt
            self._dir_rate = veh.R.T @ wv
            self._dir_rate[2] = 0.0
        self._dir_prev = d.copy()
        err = math.acos(max(-1.0, min(1.0, float(np.dot(veh.axis, d)))))
        t_av = self._thrust_available(veh, veh.pos[2]) * self.thrust_scale
        if err < math.radians(25.0):
            # Small attitude lag: deliver the commanded component along d.
            thr = float(np.clip(m * mag / (t_av * math.cos(err)), 0.0, 1.0))
        else:
            thr = float(np.clip(m * mag / t_av, 0.0, 1.0)) * float(np.clip(math.cos(err), align_gate, 1.0))
        # ... but while the attitude still lags a sideways command, its excess
        # must not go into lift: the vertical thrust stays at most the
        # commanded vertical acceleration (no climbing on a divert).
        if veh.axis[2] > 0.2:
            thr = min(thr, m * (az + 0.5) / (t_av * float(veh.axis[2])))
        if floor is not None:
            thr = max(thr, floor)
        self.cmd_throttle = thr
        self.cmd_axis = d
        c.throttle = thr
        attitude_control(veh, d, self.k_angle, self.k_rate, self.max_rate,
                         veh.throttle < 0.25 or self._height(veh) < self.low_rcs_height, hold_roll=True)

    # ------------------------------------------------------------ engage
    def _engage(self, veh, wind_now):
        """Autopilot on: the shortest-time plan of the whole recovery from
        the current state, then a few re-solves about it (the first search
        is linearised about a guessed path)."""
        self._dir_prev = veh.axis.copy()
        self._gate_xy = np.zeros(2)
        self.burn_locked = True
        self._route = None
        self.plan = None
        r, v, _ = self._state(veh)
        t_est = self._far_time(r, v)
        self.touchdown_time = self.elapsed + t_est
        plan = self._search_time(veh, wind_now, t_est) if self.planner.available else None
        if plan is not None:
            self._commit(plan)
            for _ in range(self.engage_refine):
                p = self._solve_now(veh, wind_now, self.plan.t_f)
                if p is None or not p.reaches_pad:
                    break
                self._commit(p)
        self._next_search = self.elapsed + self.search_backoff
        self._publish()
        self.next_replan = self.elapsed + self.burn_replan_period

    def _mark(self):
        if self.phase in ("COAST", "BOOSTBACK", "LANDING BURN", "LANDED") and self.phase not in self.visited:
            self.visited.append(self.phase)

    # ------------------------------------------------------------ attitude
    def _attitude(self, veh, desired_axis, use_rcs):
        """SO(3) attitude control: TVC for pitch/yaw, RCS for roll (+ low thrust)."""
        attitude_control(veh, desired_axis, self.k_angle, self.k_rate, self.max_rate, use_rcs, hold_roll=True)
