"""Closed-loop landing guidance for the 2D rocket.

Falcon-9-style sequence, planned by the G-FOLD planner in ``planner.py``:

* **Coast.**  Engine off, engines-first.  The planner searches the final time
  that minimises fuel; the optimum is "coast, then one hard burn", so the
  engine lights exactly when that burn has to begin.  Planning uses 80 % of
  the maximum thrust so the closed loop keeps a control margin.
* **Boost-back.**  If the optimal plan burns, coasts, and burns again, the
  first burn is a boost-back that moves the landing point onto the pad.
* **Landing burn.**  Once the plan is one continuous burn the touchdown time
  is locked and the plan is re-solved at 5 Hz with a smooth objective and a
  minimum throttle, at 85 % thrust.  If the locked time becomes unreachable it
  moves to the *earliest* reachable time, never simply later (that would
  hover above the pad).  There is no separate terminal phase: the burn runs to
  contact, re-planning stops in the last ~1.2 s, and the tilt limit tightens
  with height so the vehicle is upright at touchdown.
* **Tracking at 120 Hz.**  Planned thrust (feed-forward) plus PD feedback on
  the planned position and velocity, and a cascaded attitude loop (angle ->
  capped body rate -> angular acceleration -> gimbal, with turn-rate
  feed-forward).  Cold-gas RCS holds attitude while the engine is off.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np

from .mathutil import move_angle, move_towards, wrap_angle
from .planner import ConvexLandingPlanner, LandingPlan, PlanConfig


class AutonomousGuidance:
    """Coast -> (boost-back) -> single landing burn -> touchdown."""

    # Re-planning cadence.
    coast_replan_period = 0.5
    burn_replan_period = 0.2
    # Thrust budget: plan the ignition with margin, fly the burn with more.
    ignition_thrust_fraction = 0.80
    burn_thrust_fraction = 0.85
    minimum_throttle_fraction = 0.30
    # Largest thrust tilt from vertical the planner may use for a divert.
    divert_tilt_deg = 70.0
    touchdown_speed = 1.0
    # No separate terminal phase: the landing burn is flown to contact.
    # Below ``late_height`` a short time-to-go is normal (the plan is about to
    # end on the pad); above it, it means the vehicle is running late.
    late_height = 20.0

    def __init__(self) -> None:
        self.planner = ConvexLandingPlanner()
        self.reset()

    def reset(self) -> None:
        self.elapsed = 0.0
        self.plan: LandingPlan | None = None
        self.next_replan = 0.0
        self.burn_locked = False
        self.touchdown_time: float | None = None
        self._failed_solves = 0
        self._previous_desired_angle: float | None = None
        self._desired_rate = 0.0
        self.burn_fraction = self.burn_thrust_fraction
        self._last_search = -math.inf
        self.predicted_points: list[tuple[float, float]] = []
        self.phase = "STANDBY"
        self.last_status = "idle"
        self.time_to_go = float("nan")
        self.last_command = (0.0, 0.0)
        self.touchdown_velocity = (0.0, 0.0)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _drag_accel(rocket, h: float, vx: float, vz: float, body=None) -> np.ndarray:
        """Drag acceleration in the local frame, same model as ``Rocket.step``.

        ``body`` is the unit vector of the vehicle axis (engine -> nose).  If
        omitted the vehicle is assumed to fly engines-first (axial flow).
        """
        speed = math.hypot(vx, vz)
        if speed < 1e-6:
            return np.zeros(2)
        altitude = max(0.0, h + rocket.pad_center_height)
        if altitude >= rocket.ATMOSPHERE_TOP:
            return np.zeros(2)
        density = rocket.SEA_LEVEL_DENSITY * math.exp(-altitude / rocket.ATMOSPHERE_SCALE_HEIGHT)
        area = rocket.FRONTAL_AREA
        if body is not None:
            ux, uz = vx / speed, vz / speed
            axial = abs(ux * body[0] + uz * body[1])
            broadside = abs(ux * body[1] - uz * body[0])
            area = rocket.FRONTAL_AREA * axial + rocket.SIDE_AREA * broadside
        k = 0.5 * density * rocket.DRAG_COEFFICIENT * area / rocket.mass
        return -k * speed * np.array([vx, vz])

    def _drag_model(self, rocket, state):
        """Drag along the previous plan (a successive-convexification step).

        The vehicle axis is taken from the previous plan's thrust direction
        (the TVC keeps the body aligned with it), so a tilted divert burn is
        charged the much larger broadside drag it really produces.
        """
        previous = self.plan
        now = self.elapsed
        axis = rocket.body_up()
        body_now = (float(axis.dot(rocket.local_right())), float(axis.dot(rocket.local_up())))
        current = self._drag_accel(rocket, state[1], state[2], state[3], body_now)

        def model(times: np.ndarray) -> np.ndarray:
            out = np.empty((len(times), 2))
            for i, t in enumerate(times):
                if t < 0.6:
                    # The next fraction of a second is flown in the present attitude.
                    out[i] = self._drag_accel(rocket, state[1], state[2], state[3], body_now)
                elif previous is not None:
                    pos, vel, accel, _ = previous.sample(now + t - previous.created_at)
                    norm = float(np.linalg.norm(accel))
                    body = accel / norm if norm > 2.0 else None
                    out[i] = self._drag_accel(rocket, pos[1], vel[0], vel[1], body)
                else:
                    out[i] = current
            return out

        return model

    def _initial_time_bracket(self, state, a_max: float) -> tuple[float, float]:
        x, h, vx, vz = state
        g = self.planner.gravity
        a_net = max(1.0, self.ignition_thrust_fraction * a_max - g)
        v_ign = math.sqrt(max(0.0, (vz * vz + 2.0 * g * max(h, 0.0)) / (1.0 + g / a_net)))
        t_vertical = max(0.0, (vz + v_ign) / g) + v_ign / a_net
        lateral = 0.5 * a_max
        t_lateral = abs(vx) / lateral + 2.0 * math.sqrt(abs(x) / lateral)
        t0 = max(2.0, t_vertical, t_lateral)
        return max(1.0, 0.35 * t0), 2.5 * t0 + 6.0

    def _publish_prediction(self, rocket) -> None:
        plan = self.plan
        if plan is None:
            self.predicted_points = []
            return
        pad_y = rocket.world_y_for_altitude(rocket.launch_x, 0.0)
        start = max(0.0, self.elapsed - plan.created_at)
        remaining = max(0.0, plan.t_f - start)
        samples = max(2, min(240, int(remaining / 0.1) + 2))
        points = []
        for t in np.linspace(start, plan.t_f, samples):
            pos, _, _, _ = plan.sample(float(t))
            points.append((rocket.launch_x + float(pos[0]), pad_y + max(0.0, float(pos[1]))))
        self.predicted_points = points

    # -- planning ----------------------------------------------------------------
    def _replan(self, rocket, state, a_max: float) -> None:
        throttle_accel = rocket.throttle * a_max
        up = rocket.local_up()
        right = rocket.local_right()
        thrust = rocket.last_thrust / rocket.mass
        u_now = (float(thrust.dot(right)), float(thrust.dot(up)))
        engine_on = rocket.throttle > 0.02
        drag_model = self._drag_model(rocket, state)
        plan: LandingPlan | None = None

        if not self.burn_locked:
            cfg = PlanConfig(
                mode="fuel",
                rho_min=0.0,
                rho_max=self.ignition_thrust_fraction * a_max,
                gamma_now=throttle_accel,
                u_now=u_now,
                engine_on=engine_on,
                max_tilt_deg=self.divert_tilt_deg,
                touchdown_speed=self.touchdown_speed,
                drag_model=drag_model,
            )
            if self.plan is None or self.touchdown_time is None:
                lo, hi = self._initial_time_bracket(state, a_max)
                plan = self.planner.search_final_time(state, lo, hi, cfg, evaluations=16)
            else:
                guess = max(1.0, self.touchdown_time - self.elapsed)
                kept = self.planner.solve(state, guess, cfg)
                kept_ok = kept is not None and kept.reaches_pad
                recently_searched = self.elapsed - self._last_search < self.coast_replan_period - 1e-6
                if kept_ok and (engine_on or recently_searched):
                    # A burn is in progress (boost-back), or the time was just
                    # optimised: keep the touchdown time fixed so consecutive
                    # plans agree (and the fast pre-ignition cadence costs one
                    # solve instead of a whole search).
                    plan = kept
                else:
                    self._last_search = self.elapsed
                    lo, hi = 0.85 * guess - 0.5, 1.15 * guess + 0.5
                    plan = self.planner.search_final_time(state, lo, hi, cfg, evaluations=8)
                    if plan is not None and min(plan.t_f - lo, hi - plan.t_f) < 0.03 * (hi - lo):
                        lo2, hi2 = self._initial_time_bracket(state, a_max)
                        wider = self.planner.search_final_time(state, lo2, hi2, cfg, evaluations=16)
                        if wider is not None and wider.objective < plan.objective:
                            plan = wider
                    # Hysteresis: only move the touchdown time for a clear gain.
                    if kept_ok and (plan is None or plan.objective > kept.objective - max(0.5, 0.005 * abs(kept.objective))):
                        plan = kept
            if plan is not None:
                reference = self.ignition_thrust_fraction * a_max
                later = plan.gammas[1:]
                if later.min() >= 0.25 * reference and plan.gammas[1] >= 0.5 * reference:
                    # One continuous burn from now until touchdown: this is
                    # the landing-burn ignition point.  Lock the touchdown time.
                    self.burn_locked = True
        else:
            t_go = self.touchdown_time - self.elapsed
            cfg = PlanConfig(
                mode="smooth",
                rho_min=self.minimum_throttle_fraction * a_max,
                rho_max=self.burn_fraction * a_max,
                gamma_now=throttle_accel,
                u_now=u_now,
                engine_on=True,
                max_tilt_deg=self.divert_tilt_deg,
                touchdown_speed=self.touchdown_speed,
                drag_model=drag_model,
            )
            plan = self.planner.solve(state, t_go, cfg)
            if plan is None:
                # A failed solve is not evidence that the target moved: keep
                # flying the previous plan and try again next cycle.
                self._failed_solves += 1
                if self._failed_solves < 3:
                    self.last_status = self.planner.last_status
                    return
            else:
                self._failed_solves = 0
            if plan is None or not plan.reaches_pad:
                # The locked touchdown time is no longer reachable.  Move it
                # to the *earliest* time that is reachable (never simply
                # "later": a smooth plan with spare time would descend lazily
                # and hover above the pad), raising the thrust budget if needed.
                found = self._earliest_reachable_time(state, cfg, t_go, a_max)
                if found is not None:
                    t_new, fraction = found
                    self.burn_fraction = max(self.burn_fraction, min(0.97, fraction + 0.05))
                    cfg.rho_max = self.burn_fraction * a_max
                    smooth = self.planner.solve(state, t_new, cfg)
                    if smooth is not None:
                        plan = smooth

        self.last_status = self.planner.last_status
        if plan is not None:
            plan.created_at = self.elapsed
            self.plan = plan
            self.touchdown_time = self.elapsed + plan.t_f
            self.last_status = plan.status

    def _earliest_reachable_time(self, state, cfg: PlanConfig, t_go: float, a_max: float) -> tuple[float, float] | None:
        """Shortest time of flight that reaches the pad, and the thrust used.

        The thrust budget is escalated (80 % -> 90 % -> 97 %) before the time
        of flight is stretched: stretching time on a falling vehicle only
        helps by flying low and sideways, which is exactly what must not
        happen near the ground.
        """
        for fraction in (self.ignition_thrust_fraction, 0.90, 0.97):
            check = PlanConfig(**{**cfg.__dict__, "mode": "fuel", "rho_max": fraction * a_max})
            reachable = partial(self._reaches_pad, state, check)
            lo = max(0.8, 0.6 * t_go)
            hi = max(t_go + 1.0, 1.25 * t_go)
            if not reachable(hi):
                continue
            if reachable(lo):
                return lo, fraction
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                if reachable(mid):
                    hi = mid
                else:
                    lo = mid
            return hi, fraction
        return None

    def _reaches_pad(self, state, cfg: PlanConfig, t_f: float) -> bool:
        plan = self.planner.solve(state, t_f, cfg)
        return plan is not None and plan.reaches_pad

    # -- main entry point ----------------------------------------------------
    def command(self, rocket, dt: float) -> None:
        """Run one 120 Hz guidance/control step on the nonlinear rocket."""
        if rocket.state in {"CRASHED", "LANDED"}:
            rocket.throttle = 0.0
            rocket.gimbal = move_angle(rocket.gimbal, 0.0, rocket.GIMBAL_RATE * dt)
            rocket.rcs_command = 0.0
            if rocket.state == "LANDED":
                self.phase = "LANDED"
            self.predicted_points = []
            return
        if rocket.state == "ON PAD":
            rocket.throttle = 0.0
            rocket.gimbal = move_angle(rocket.gimbal, 0.0, rocket.GIMBAL_RATE * dt)
            rocket.rcs_command = 0.0
            self.phase = "STANDBY"
            return

        dt = float(dt)
        self.elapsed += dt
        up = rocket.local_up()
        right = rocket.local_right()
        x = float(rocket.position.x - rocket.launch_x)
        h = float(rocket.altitude)
        vx = float(rocket.velocity.dot(right))
        vz = float(rocket.vertical_speed)
        state = np.array([x, h, vx, vz])
        self.touchdown_velocity = (vx, vz)
        a_max = rocket.MAX_THRUST / rocket.mass
        g = self.planner.gravity
        drag_now = np.array([rocket.last_drag.dot(right), rocket.last_drag.dot(up)]) / rocket.mass

        t_go = (self.touchdown_time - self.elapsed) if self.touchdown_time is not None else math.inf
        if self.burn_locked and t_go < 1.5 and h > self.late_height:
            # Running late (e.g. heavy disturbance): re-open the touchdown time.
            self.touchdown_time = self.elapsed + 3.0
            self.next_replan = self.elapsed
            t_go = 3.0

        if self.planner.available and self.elapsed >= self.next_replan:
            if not self.burn_locked or t_go > 1.2:
                self._replan(rocket, state, a_max)
                self._publish_prediction(rocket)
            period = self.burn_replan_period if self.burn_locked else self.coast_replan_period
            if not self.burn_locked and self.plan is not None:
                # Tighten the cadence as the planned ignition approaches.
                hot = np.nonzero(self.plan.gammas > 0.25 * self.ignition_thrust_fraction * a_max)[0]
                if hot.size:
                    until_ignition = hot[0] * self.plan.dt
                    period = float(np.clip(until_ignition - 0.05, 0.1, period))
            self.next_replan = self.elapsed + period
            t_go = (self.touchdown_time - self.elapsed) if self.touchdown_time is not None else math.inf
        self.time_to_go = t_go

        coasting = False
        if self.plan is not None:
            tau = self.elapsed - self.plan.created_at
            pos_ref, vel_ref, u_ff, _ = self.plan.sample(tau)
            gamma_ff = self.plan.gamma_at(tau)
            coasting = (not self.burn_locked) and gamma_ff < 0.06 * a_max
            fb = 0.45 * (pos_ref - state[:2]) + 1.5 * (vel_ref - state[2:])
            norm = float(np.linalg.norm(fb))
            if norm > 4.0:
                fb *= 4.0 / norm
            # Drag is already in the plan (re-predicted every re-plan); the PD
            # term absorbs the residual, so no separate drag cancellation.
            accel = u_ff + fb
            if self.burn_locked:
                self.phase = "LANDING BURN"
            else:
                if coasting:
                    self.phase = "COAST"
                else:
                    # A burn counts as boost-back only if the plan coasts
                    # again afterwards; otherwise it is the landing-burn
                    # ignition ramp just before the burn is locked.
                    k0 = min(int(max(tau, 0.0) / self.plan.dt) + 1, self.plan.knots)
                    later = self.plan.gammas[k0:]
                    gap = later.size > 0 and later.min() < 0.06 * a_max
                    self.phase = "BOOSTBACK" if gap else "LANDING BURN"
            max_tilt = math.radians(75.0)
            if self.burn_locked:
                # Past the plan's end the reference is "on the pad, sinking at
                # touchdown speed", so a late vehicle keeps descending.  Never
                # stall or climb just above the pad.
                if h < 10.0 and vz > -0.6 * self.touchdown_speed:
                    accel[1] = min(accel[1], 0.9 * g)
                # Tilt budget tightens with height: upright at contact.
                max_tilt = math.radians(float(np.interp(h, [0.3, 1.5, 3.0, 12.0, 40.0], [0.6, 2.0, 4.0, 6.0, 75.0])))
        else:
            # No solver available: bounded suicide-burn feedback law.
            self.phase = "FALLBACK"
            v_ref = -math.sqrt(2.0 * 0.5 * (a_max - g) * max(h, 0.0)) - self.touchdown_speed
            az = g - drag_now[1] + 1.5 * (v_ref - vz)
            ax = -drag_now[0] + 0.08 * (-x) - 0.9 * vx
            accel = np.array([ax, max(0.0, az)])
            max_tilt = math.radians(35.0)

        # -- actuators -----------------------------------------------------
        if coasting:
            # Engine off.  Like Falcon 9, coast engines-first (lowest drag,
            # and exactly what the planner's drag model assumes); only in
            # the last few seconds before ignition does RCS turn the vehicle
            # to the direction of the upcoming burn.
            rocket.throttle = move_towards(rocket.throttle, 0.0, rocket.THROTTLE_RATE * dt)
            burn_dir = self._upcoming_burn_direction(a_max, within=4.0)
            if burn_dir is None:
                speed = math.hypot(vx, vz)
                burn_dir = np.array([-vx, -vz]) / speed if speed > 5.0 else np.array([0.0, 1.0])
                if burn_dir[1] < 0.2:
                    burn_dir = np.array([np.sign(burn_dir[0]) * 0.98, 0.2])
            desired_world = right * float(burn_dir[0]) + up * float(burn_dir[1])
            self._attitude(rocket, math.atan2(desired_world.x, desired_world.y), dt, use_rcs=True)
            self.last_command = (0.0, 0.0)
            return

        # Limit the thrust tilt from the local vertical.
        ax, az = float(accel[0]), float(accel[1])
        az = max(az, 0.0)
        limit = math.tan(max_tilt) * az if max_tilt < math.radians(89.0) else math.inf
        if az < 1e-6:
            az = 1e-6
            limit = math.tan(max_tilt) * max(abs(ax), 1.0)
        ax = float(np.clip(ax, -limit, limit))
        self.last_command = (ax, az)
        thrust_world = right * ax + up * az
        magnitude = thrust_world.length()
        desired_angle = math.atan2(thrust_world.x, thrust_world.y)
        error = wrap_angle(rocket.angle - desired_angle)
        # Do not push hard in a wrong direction while the body is still turning.
        alignment = float(np.clip(math.cos(error), 0.35, 1.0))
        throttle = float(np.clip(magnitude / a_max, 0.0, 1.0)) * alignment
        if self.burn_locked:
            throttle = max(throttle, 0.9 * self.minimum_throttle_fraction)
        rocket.throttle = move_towards(rocket.throttle, throttle, rocket.THROTTLE_RATE * dt)
        self._attitude(rocket, desired_angle, dt, use_rcs=rocket.throttle < 0.25)

    def _upcoming_burn_direction(self, a_max: float, within: float = math.inf) -> np.ndarray | None:
        plan = self.plan
        if plan is None:
            return None
        tau = self.elapsed - plan.created_at
        start = min(int(max(tau, 0.0) / plan.dt), plan.knots)
        for k in range(start, plan.knots + 1):
            if k * plan.dt - tau > within:
                return None
            if plan.gammas[k] > 0.25 * self.ignition_thrust_fraction * a_max:
                u = plan.controls[k]
                norm = float(np.linalg.norm(u))
                if norm > 1e-6:
                    return u / norm
        return None

    # Attitude loop: angle error -> rate command (capped) -> angular accel.
    attitude_angle_gain = 1.0      # 1/s
    attitude_rate_gain = 2.0       # 1/s   (-> wn 1.4 rad/s, zeta 0.7)
    attitude_max_rate = math.radians(20.0)
    # The gimbal slews at only 42 deg/s; commanding small deflections keeps
    # its reversal time short, which is what prevents attitude overshoot.
    attitude_max_gimbal = math.radians(12.0)

    def _attitude(self, rocket, desired_angle: float, dt: float, *, use_rcs: bool) -> None:
        """Cascaded attitude loop driving TVC (and RCS at low thrust).

        The outer loop turns the pointing error into a body-rate command capped
        at 30 deg/s; the inner loop converts the rate error into an angular
        acceleration and, through the thrust-dependent lever arm, into a
        gimbal angle.  The rate cap is what removes the whip-like overshoot of
        a plain PD law on large re-pointing manoeuvres.
        """
        error = wrap_angle(rocket.angle - desired_angle)
        omega = rocket.angular_velocity
        inertia = rocket.moment_of_inertia
        # Feed-forward of the commanded turn rate (filtered derivative of the
        # desired angle) removes the steady lag while following a turning
        # thrust vector, e.g. the final "S" correction before touchdown.
        if self._previous_desired_angle is not None:
            raw_rate = wrap_angle(desired_angle - self._previous_desired_angle) / max(dt, 1e-6)
            raw_rate = float(np.clip(raw_rate, -self.attitude_max_rate, self.attitude_max_rate))
            blend = dt / (0.25 + dt)
            self._desired_rate += blend * (raw_rate - self._desired_rate)
        self._previous_desired_angle = desired_angle
        rate_cmd = self._desired_rate - float(np.clip(self.attitude_angle_gain * error, -self.attitude_max_rate, self.attitude_max_rate))
        rate_cmd = float(np.clip(rate_cmd, -self.attitude_max_rate, self.attitude_max_rate))
        alpha = self.attitude_rate_gain * (rate_cmd - omega)
        thrust = rocket.throttle * rocket.MAX_THRUST
        if thrust > 0.05 * rocket.MAX_THRUST:
            # alpha = -L * T * sin(gimbal) / I
            ratio = float(np.clip(-alpha * inertia / (rocket.THRUST_LEVER_ARM * thrust), -1.0, 1.0))
            limit = min(rocket.GIMBAL_LIMIT, self.attitude_max_gimbal)
            target = float(np.clip(math.asin(ratio), -limit, limit))
        else:
            target = 0.0
        rocket.gimbal = move_angle(rocket.gimbal, target, rocket.GIMBAL_RATE * dt)
        if use_rcs:
            authority = rocket.RCS_MAX_TORQUE / inertia
            rocket.rcs_command = float(np.clip(alpha / authority, -1.0, 1.0))
        else:
            rocket.rcs_command = 0.0

