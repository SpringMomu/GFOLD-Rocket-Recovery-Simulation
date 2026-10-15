"""Falcon-9-style recovery autopilot for the 3D booster.

Coast -> (boost-back) -> one landing burn straight to leg contact.  The burn is
planned by the G-FOLD planner (``planner.py``); attitude control and actuator
allocation are in ``attitude.py``.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np

from .attitude import attitude_control
from .planner import Plan3D, PlanConfig3D, Planner3D
from .vehicle import G0, atmosphere

# Landing-burn tilt budget vs height (m -> deg): upright at contact.
TILT_BUDGET = ([0.3, 1.5, 3.0, 12.0, 40.0], [0.6, 2.0, 5.0, 10.0, 75.0])


class Autopilot3D:
    """Falcon-9 style recovery: coast -> (boost-back) -> one landing burn to touchdown.

    * COAST: engine off, engines-first and aligned with the relative wind (the
      grid fins make that attitude aerodynamically stable).  A drag-aware
      ballistic prediction gives the impact point, and an ignition predictor
      decides the latest safe moment to light the engine.
    * BOOSTBACK: only when the ballistic impact point is far from the pad; the
      burn is steered so that the predicted impact moves onto the pad.
    * LANDING BURN: at ignition the touchdown time is locked and the 3-D
      G-FOLD SOCP (Planner3D) is re-solved every 0.2 s with a smooth
      objective; if the pad becomes unreachable the time moves to the
      earliest reachable one (never simply later).  The plan ends on the pad
      itself (base height 0, vertical, 1.2 m/s sink); the burn continues
      until a foot touches and the engine is cut -- no separate terminal
      phase.  The last ~0.6 s are flown on the final plan with a tilt limit
      that tightens with height, so the vehicle is upright at contact.
    * Grid fins: stowed for the boost-back, deployed for the coast/entry
      and the landing burn; they steer the attitude by their real
      aerodynamic forces (see attitude_control).
    """

    burn_replan_period = 0.2
    coast_predict_period = 0.25
    ignition_fraction = 0.62
    burn_fraction_nominal = 0.85
    minimum_throttle = 0.20
    divert_tilt_deg = 65.0
    touchdown_speed = 1.2
    gate_height = 0.0             # G-FOLD plan ends on the pad (base height 0) ...
    gate_speed = 1.2              # ... vertical, at the touchdown sink rate
    low_rcs_height = 15.0         # RCS assists TVC close to the ground
    sink_gain = 0.5               # final approach: sink <= 1.2 + 0.5 h (m/s): 6.2 at 10 m, 2.2 at 2 m
    sink_track_height = 15.0      # the tracker enforces the envelope below this height
    legs_deploy_height = 450.0
    boostback_miss = 350.0
    aero_steer_max_deg = 15.0
    lift_gradient_cap = 3.0       # body lift per radian of tilt, in units of thrust accel
    boostback_done_miss = 25.0
    # attitude loop
    k_angle = 2.0
    k_rate = 4.0
    max_rate = math.radians(30.0)

    def __init__(self, aero_model=None) -> None:
        self.planner = Planner3D()
        self.aero = aero_model
        self.reset()

    def reset(self) -> None:
        self.elapsed = 0.0
        self.plan: Plan3D | None = None
        self.next_replan = 0.0
        self.next_predict = 0.0
        self.burn_locked = False
        self.touchdown_time: float | None = None
        self.burn_fraction = self.burn_fraction_nominal
        self._failed = 0
        self._boostback = False
        self._boostback_done = False
        self._touched = False
        self._dir_prev = None
        self._gate_xy = np.zeros(2)
        self._dir_rate = np.zeros(3)
        self._dt = 1.0 / 240.0
        self._best_miss = math.inf
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
        r = np.array([veh.pos[0], veh.pos[1], veh.altitude])
        return r, veh.vel.copy(), veh.mass

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

        def model(times):
            n = len(times)
            out = np.zeros((n, 3))
            G = np.zeros((n, 3, 3))
            if self.aero is None:
                return out, G
            for i, t in enumerate(times):
                u_ref = None
                if prev is None or t < 0.3:
                    f = max(0.0, 1.0 - t / max(t_guess, 0.5))
                    h, v, m, sig = r0[2] + v0[2] * t * 0.5, v0 * f, m0, max(sig_now, 0.6 * s.thrust_sl / m0)
                    if t < 0.3 and veh.throttle > 0.05:
                        u_ref = veh.last["thrust"] / m0
                else:
                    tau = now + t - prev.created_at
                    pos, vel, acc = prev.sample(tau)
                    h, v = pos[2], vel
                    k = min(int(max(tau, 0.0) / prev.dt), prev.knots)
                    m = float(math.exp(prev.z[k]))
                    sig = max(3.0, prev.sigma_at(tau))
                    u_ref = acc
                rho, _, _, a = atmosphere(max(0.0, h))
                vr = v - wind_now
                sp = float(np.linalg.norm(vr))
                if sp < 3.0:
                    continue
                u = vr / sp
                base = -u
                # Linearise around the reference attitude (successive
                # linearisation: each replan uses the previous plan's tilt).
                tilt = np.zeros(3)
                if u_ref is not None and float(np.linalg.norm(u_ref)) > 2.0:
                    d_ref = u_ref / np.linalg.norm(u_ref)
                    tilt = d_ref - base * float(np.dot(d_ref, base))
                    if float(np.dot(d_ref, base)) < 0.2:
                        tilt = np.zeros(3)
                axis0 = base + tilt
                axis0 /= np.linalg.norm(axis0)
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
                G[i] = Gi
            return out, G

        return model

    def _config(self, veh, frac):
        s = veh.spec
        return PlanConfig3D(
            mode="smooth", thrust_min=self.minimum_throttle * s.thrust_sl, thrust_max=frac * s.thrust_sl, isp=s.isp_sl,
            dry_mass=s.dry_mass + 150.0, sigma_now=veh.throttle * s.thrust_sl / veh.mass,
            u_now=tuple(veh.last["thrust"] / veh.mass), engine_on=True,
            u_cmd=tuple(self.plan.sample(self.elapsed - self.plan.created_at)[2]) if self.plan is not None else None,
            # Low, short burns get a small tilt budget: a big lateral swing
            # close to the ground is exactly the "rocking" to avoid.
            # ... unless there is real sideways speed to kill (divert).
            max_tilt_deg=float(max(np.interp(veh.altitude, [20.0, 120.0, 500.0], [6.0, 14.0, self.divert_tilt_deg]),
                                   min(self.divert_tilt_deg, 2.0 * float(np.linalg.norm(veh.vel[:2]))))),
            touchdown_speed=self.gate_speed, target_alt=self.gate_height, target_xy=tuple(self._gate_xy),
            sink_gain=self.sink_gain,
        )

    @staticmethod
    def _thrust_available(veh, h):
        s = veh.spec
        _, p, _, _ = atmosphere(max(0.0, h))
        return s.thrust_vac - (s.thrust_vac - s.thrust_sl) * min(1.0, p / 101_325.0)

    # ------------------------------------------------------- predictions
    def _burn_need(self, veh, r, v, m, drag_up):
        """(thrust fraction needed for a landing burn started now, burn time, thrust direction)."""
        g = G0
        vz = float(v[2])
        h_eff = r[2] - self.gate_height - 0.45 * max(0.0, -vz)   # engine spool-up margin
        if vz > -(self.touchdown_speed + 2.0) and h_eff > 5.0:
            return 0.0, math.inf, np.array([0.0, 0.0, 1.0])
        if h_eff <= 0.5:
            return 9.0, 1.0, np.array([0.0, 0.0, 1.0])
        a_v = max(0.1, (vz * vz - self.gate_speed ** 2) / (2.0 * h_eff))
        t_b = max(0.5, (-vz - self.gate_speed) / a_v)
        a_lat = -(6.0 * r[:2] / t_b ** 2 + 4.0 * v[:2] / t_b) * 1.0
        need = np.array([a_lat[0], a_lat[1], a_v + g - drag_up / 3.0])
        frac = m * float(np.linalg.norm(need)) / self._thrust_available(veh, r[2])
        return frac, t_b, need / float(np.linalg.norm(need))

    def _ballistic(self, veh, r, v, m, wind_now, dt=0.25, t_max=240.0, stop_on_ignition=True):
        """Drag-aware coast prediction. Returns (points, impact_xy, t_impact, t_ignition, burn_time)."""
        pts = [r.copy()]
        p, u = r.copy(), v.copy()
        t = 0.0
        t_ign, t_burn = math.inf, math.inf
        while t < t_max:
            rho, _, _, a = atmosphere(max(0.0, p[2]))
            f = self.aero.quick_force(u - wind_now, None, rho, a, 0.0) if self.aero is not None else np.zeros(3)
            if stop_on_ignition and not math.isfinite(t_ign):
                frac, tb, _ = self._burn_need(veh, p, u, m, max(0.0, f[2] / m))
                if frac >= self.ignition_fraction:
                    t_ign, t_burn = t, tb
                    ign_p, ign_v = p.copy(), u.copy()
            acc = f / m + np.array([0.0, 0.0, -G0])
            u = u + acc * dt
            p = p + u * dt
            t += dt
            if len(pts) < 400:
                pts.append(p.copy())
            if p[2] <= 0.0:
                break
        impact = p[:2].copy()
        if math.isfinite(t_ign):
            # Visual prediction: coast until ignition, then a smooth arc to the pad.
            k = int(t_ign / dt)
            pts = pts[:k + 1]
            p0, v0 = ign_p, ign_v
            T = max(t_burn, 0.5)
            v_end = np.array([0.0, 0.0, -self.touchdown_speed])
            for s in np.linspace(0.0, 1.0, 30)[1:]:
                # Cubic Hermite from (p0, v0) to (pad, touchdown velocity); the pad is the origin.
                h00, h10, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, s ** 3 - s ** 2
                pts.append(h00 * p0 + h10 * T * v0 + h11 * T * v_end)
        return pts, impact, t, t_ign, t_burn

    # ------------------------------------------------------------- replan
    def _replan(self, veh, wind_now):
        r, v, m = self._state(veh)
        P = self.planner
        t_go = self.touchdown_time - self.elapsed
        cfg = self._config(veh, self.burn_fraction)
        cfg.drag_model = self._drag_model(veh, wind_now, t_go)
        plan = P.solve(r, v, m, t_go, cfg)
        if plan is None:
            self._failed += 1
            if self._failed < 3 and self.plan is not None:
                return
        else:
            self._failed = 0
        if (plan is None or not plan.reaches_pad) and (t_go > 5.0 or plan is None):
            found = self._earliest(r, v, m, cfg, t_go, veh)
            if found is not None:
                t_new, frac = found
                # Earliest reachable time at this thrust level plus a small
                # tracking margin (the smooth plan then is not saturated).
                t_new = t_new + min(1.0, 0.08 * t_new)
                self.burn_fraction = max(self.burn_fraction, frac)
                cfg.thrust_max = self.burn_fraction * veh.spec.thrust_sl
                cfg.drag_model = self._drag_model(veh, wind_now, t_new)
                smooth = P.solve(r, v, m, t_new, cfg)
                if smooth is not None:
                    plan = smooth
        self.last_status = P.last_status
        if plan is not None:
            plan.created_at = self.elapsed
            self.plan = plan
            self.touchdown_time = self.elapsed + plan.t_f
            self.last_status = plan.status

    def _earliest(self, r, v, m, cfg, t_go, veh):
        for frac in (self.burn_fraction, 0.92, 0.97):
            chk = PlanConfig3D(**{**cfg.__dict__, "mode": "fuel", "thrust_max": frac * veh.spec.thrust_sl})
            ok = partial(self._reaches_pad, r, v, m, chk)
            lo = max(0.8, 0.6 * t_go)
            hi = max(t_go + 1.5, 1.5 * t_go)
            if not ok(hi):
                continue
            if ok(lo):
                return lo, frac
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                if ok(mid):
                    hi = mid
                else:
                    lo = mid
            return hi, frac
        return None

    def _reaches_pad(self, r, v, m, cfg, t_f):
        plan = self.planner.solve(r, v, m, t_f, cfg)
        return plan is not None and plan.reaches_pad

    def _publish(self):
        p = self.plan
        if p is None:
            return
        start = max(0.0, self.elapsed - p.created_at)
        n = max(2, min(200, int(max(0.0, p.t_f - start) / 0.1) + 2))
        pts = []
        for t in np.linspace(start, p.t_f, n):
            pos, _, _ = p.sample(float(t))
            pts.append((float(pos[0]), float(pos[1]), float(max(0.0, pos[2]))))
        self.predicted_points = pts

    # ------------------------------------------------------------ command
    def command(self, veh, wind_now, dt):
        c = veh.controls
        if veh.state in ("CRASHED", "LANDED"):
            c.throttle = 0.0
            c.rcs = (0.0, 0.0, 0.0)
            c.gimbal = (0.0, 0.0)
            c.fins = (0.0, 0.0, 0.0)
            self.phase = "LANDED" if veh.state == "LANDED" else "STANDBY"
            self._mark()
            return
        if veh.state == "ON PAD":
            c.throttle = 0.0
            self.phase = "STANDBY"
            return
        self.elapsed += dt
        self._dt = dt
        s = veh.spec
        r, v, m = self._state(veh)
        g = G0
        h = r[2]

        # Touchdown: cut the engine as soon as a foot is on the ground.
        if veh.feet_contact > 0 or self._touched:
            self._touched = True
            c.throttle = 0.0
            self.phase = "LANDING BURN"
            self._attitude(veh, np.array([0.0, 0.0, 1.0]), use_rcs=True)
            self._mark()
            return

        if h < self.legs_deploy_height:
            c.legs_down = True
        if self.burn_locked:
            c.fins_deploy = True

        if not self.burn_locked:
            self._pre_ignition(veh, r, v, m, wind_now)
            return

        t_go = self.touchdown_time - self.elapsed
        # One burn all the way to leg contact: re-plan until the last ~0.6 s,
        # then fly the final plan.  Past its end the plan's reference is
        # "on the pad, sinking at touchdown speed", so a late vehicle keeps
        # descending instead of hovering.
        if self.planner.available and self.elapsed >= self.next_replan and t_go > 0.6:
            self._replan(veh, wind_now)
            self._publish()
            self.next_replan = self.elapsed + self.burn_replan_period
            t_go = self.touchdown_time - self.elapsed
        self.time_to_go = max(0.0, t_go)
        if t_go < 8.0:
            c.legs_down = True

        if self.plan is not None:
            self.phase = "LANDING BURN"
            tau = self.elapsed - self.plan.created_at
            p_ref, v_ref, u_ff = self.plan.sample(tau)
            fb = 0.5 * (p_ref - r) + 1.6 * (v_ref - v)
            nfb = float(np.linalg.norm(fb))
            if nfb > 5.0:
                fb *= 5.0 / nfb
            accel = u_ff + fb
            if h < 10.0 and v[2] > -0.6 * self.touchdown_speed:
                accel[2] = min(accel[2], 0.9 * g)     # never stall or climb above the pad
            if h < self.sink_track_height:
                # Final approach envelope (same as the planner's): if sinking
                # faster than 1.2 + 0.5 h m/s, brake the excess.  Near the pad
                # the sink rate settles at touchdown speed with thrust ~ weight,
                # so cutting the engine at contact cannot bounce the vehicle.
                v_lim = -(self.gate_speed + self.sink_gain * max(0.0, h))
                if v[2] < v_lim:
                    accel[2] += 2.0 * (v_lim - v[2])
            # Tilt budget tightens with height: upright at contact.
            max_tilt = math.radians(float(np.interp(h, *TILT_BUDGET)))
        else:
            self.phase = "FALLBACK"
            a_max_now = s.thrust_sl / m
            vz_ref = -math.sqrt(2 * 0.5 * (a_max_now - g) * max(h, 0.0)) - self.touchdown_speed
            accel = np.array([-0.08 * r[0] - 0.9 * v[0], -0.08 * r[1] - 0.9 * v[1], g + 1.5 * (vz_ref - v[2])])
            max_tilt = math.radians(35.0)
        self._mark()
        self._fly_accel(veh, accel, max_tilt, floor=0.9 * self.minimum_throttle if h > 1.0 else None)

    def _fly_accel(self, veh, accel, max_tilt, floor=None, align_gate=0.35):
        c = veh.controls
        m = veh.mass
        horiz = accel[:2].copy()
        az = max(float(accel[2]), 1e-3)
        hn = float(np.linalg.norm(horiz))
        lim = math.tan(max_tilt) * az
        if hn > lim:
            horiz *= lim / hn
        accel = np.array([horiz[0], horiz[1], az])
        self.cmd_accel = accel
        mag = float(np.linalg.norm(accel))
        d = accel / mag
        # Slew-rate limit on the commanded thrust direction: a vehicle this
        # size cannot follow direction jumps between replans, and chasing
        # them is what makes it rock.  Slower near the ground.
        if self._dir_prev is not None:
            slew = math.radians(float(np.interp(veh.altitude, [2.0, 30.0, 300.0], [8.0, 15.0, 25.0]))) * self._dt
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
        t_av = self._thrust_available(veh, veh.pos[2])
        if err < math.radians(25.0):
            # Small attitude lag: deliver the commanded component along d.
            thr = float(np.clip(m * mag / (t_av * math.cos(err)), 0.0, 1.0))
        else:
            thr = float(np.clip(m * mag / t_av, 0.0, 1.0)) * float(np.clip(math.cos(err), align_gate, 1.0))
        if floor is not None:
            thr = max(thr, floor)
        self.cmd_throttle = thr
        self.cmd_axis = d
        c.throttle = thr
        attitude_control(veh, d, self.k_angle, self.k_rate, self.max_rate,
                         veh.throttle < 0.25 or veh.altitude < self.low_rcs_height, hold_roll=True)

    # ----------------------------------------------------- before ignition
    def _pre_ignition(self, veh, r, v, m, wind_now):
        c = veh.controls
        s = veh.spec
        if self.elapsed >= self.next_predict or self.impact is None:
            pts, impact, t_imp, t_ign, t_burn = self._ballistic(veh, r, v, m, wind_now)
            self.impact = impact
            self.ignition_in = t_ign
            self.predicted_points = [(float(p[0]), float(p[1]), float(max(0.0, p[2]))) for p in pts[:: max(1, len(pts) // 120)]]
            self.time_to_go = t_ign + t_burn if math.isfinite(t_ign) else t_imp
            self.next_predict = self.elapsed + self.coast_predict_period
        miss = float(np.linalg.norm(self.impact))

        # ---- boost-back: move the ballistic impact point onto the pad.
        airspeed = float(np.linalg.norm(v - wind_now))
        if not self._boostback_done and (self._boostback or (miss > self.boostback_miss and r[2] > 1200.0 and airspeed < 180.0)):
            self._boostback = True
            self.phase = "BOOSTBACK"
            self._mark()
            t_fall = max(5.0, self.time_to_go)
            # Aim slightly short of the pad; the landing burn removes the rest.
            dv_h = -self.impact / t_fall
            dvn = float(np.linalg.norm(dv_h))
            self._best_miss = min(self._best_miss, miss)
            if miss < self.boostback_done_miss or dvn < 0.4 or r[2] < 700.0 or miss > self._best_miss + 30.0:
                self._boostback = False
                self._boostback_done = True
            else:
                d = np.array([dv_h[0], dv_h[1], 0.3 * dvn]) / dvn
                d /= np.linalg.norm(d)
                c.fins_deploy = False           # grid fins stay stowed for the boost-back
                err = math.acos(max(-1.0, min(1.0, float(np.dot(veh.axis, d)))))
                thr = float(np.clip(dvn / 8.0, 0.4, 0.9)) if err < math.radians(25.0) else 0.0
                c.throttle = thr
                self.cmd_throttle = thr
                self.cmd_axis = d
                self.cmd_accel = d * thr * s.thrust_sl / m
                self.next_predict = min(self.next_predict, self.elapsed + 0.1)
                self._attitude(veh, d, use_rcs=True)
                return

        # ---- coast, and the ignition decision.
        self.phase = "COAST"
        c.fins_deploy = True                    # grid fins out after the boost-back
        self._mark()
        drag_up = max(0.0, float(veh.last["aero"][2]) / m)
        frac, t_burn, burn_dir = self._burn_need(veh, r, v, m, drag_up)
        # Short, low burns (hover-slam) light a little earlier so the burn
        # lasts long enough to trim drift without a late sideways swing.
        ign = self.ignition_fraction if t_burn > 5.0 else float(np.interp(t_burn, [2.0, 5.0], [0.48, self.ignition_fraction]))
        if frac >= ign:
            self._ignite(veh, r, v, m, wind_now, t_burn)
            return
        c.throttle = 0.0
        self.cmd_throttle = 0.0
        self.cmd_accel = np.zeros(3)
        vr = v - wind_now
        sp = float(np.linalg.norm(vr))
        if self.ignition_in < 3.0:
            # Line up for the landing burn, but only pre-tilt as much as the
            # altitude warrants (a low, short burn should start nearly upright).
            cap = math.radians(float(np.clip(r[2] / 40.0, 1.0, 20.0)))
            if float(np.linalg.norm(v[:2])) > 8.0:
                cap = math.pi            # real divert: line up fully
            tl = math.acos(max(-1.0, min(1.0, float(burn_dir[2]))))
            target = burn_dir
            if tl > cap:
                hz = np.array([burn_dir[0], burn_dir[1], 0.0])
                hz /= max(1e-9, float(np.linalg.norm(hz)))
                target = hz * math.sin(cap) + np.array([0.0, 0.0, math.cos(cap)])
        elif sp > 25.0:
            target = self._aero_steer(veh, r, v, m, wind_now)   # engines-first, steered by body lift
        else:
            target = np.array([0.0, 0.0, 1.0])
        self.cmd_axis = target
        self._attitude(veh, target, use_rcs=True)

    def _aero_steer(self, veh, r, v, m, wind_now):
        """Engines-first attitude with a small angle of attack whose body/grid-fin
        lift moves the ballistic impact point toward the pad (Falcon-9 style)."""
        vr = v - wind_now
        sp = float(np.linalg.norm(vr))
        u = vr / sp
        base = -u
        if self.impact is None or self.aero is None:
            return base
        t_imp = max(3.0, self.time_to_go)
        t_ign = min(max(1.5, self.ignition_in - 3.0), t_imp)   # finish before lining up for the burn
        lever = t_ign * max(1.0, t_imp - 0.5 * t_ign)
        a_des = np.array([-self.impact[0], -self.impact[1], 0.0]) / lever
        a_des -= u * float(np.dot(a_des, u))
        an = float(np.linalg.norm(a_des))
        if an < 0.02:
            return base
        e_a = a_des / an
        rho, _, _, a_snd = atmosphere(max(0.0, r[2]))
        probe = math.radians(5.0)
        best = None
        for e in (-e_a, e_a):
            axis = math.cos(probe) * base + math.sin(probe) * e
            lift = float(np.dot(self.aero.quick_force(vr, axis, rho, a_snd, veh.legs), e_a))
            if lift > 0 and (best is None or lift > best[1]):
                best = (e, lift)
        if best is None:
            return base
        e, lift = best
        alpha = float(np.clip(an * m / (lift / probe), 0.0, math.radians(self.aero_steer_max_deg)))
        return math.cos(alpha) * base + math.sin(alpha) * e

    def _ignite(self, veh, r, v, m, wind_now, t_burn):
        self._dir_prev = veh.axis.copy()
        self._gate_xy = np.zeros(2)          # the burn itself lands on the pad centre
        self.burn_locked = True
        self._boostback = False
        self.touchdown_time = self.elapsed + max(1.5, 1.05 * t_burn) + 2.0   # + gentle final approach
        self.next_replan = self.elapsed
        self.plan = None
        self._replan(veh, wind_now)
        self._publish()
        self.next_replan = self.elapsed + self.burn_replan_period
        self.phase = "LANDING BURN"
        self._mark()
        veh.controls.throttle = self.minimum_throttle

    def _mark(self):
        if self.phase in ("COAST", "BOOSTBACK", "LANDING BURN", "LANDED") and self.phase not in self.visited:
            self.visited.append(self.phase)

    # ------------------------------------------------------------ attitude
    def _attitude(self, veh, desired_axis, use_rcs):
        """SO(3) attitude control: TVC for pitch/yaw, RCS for roll (+ low thrust)."""
        attitude_control(veh, desired_axis, self.k_angle, self.k_rate, self.max_rate, use_rcs, hold_roll=True)
