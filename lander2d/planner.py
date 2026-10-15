"""Convex planner for the 2D powered descent (lossless-convexification SOCP).

Second-order-cone program over N knots with a first-order-hold thrust
profile.  States ``(x, h, vx, vz)``, controls ``u = (ax, az)`` (thrust
acceleration) and ``Gamma >= ||u||`` (lossless relaxation of the thrust
magnitude bounds):

    r' = v,  v' = u + g + d(t)                 point mass, predicted drag d
    ||u_k|| <= Gamma_k,  rho_min <= Gamma_k <= rho_max
    u_z,k >= cos(theta(t_go)) Gamma_k          tilt limit, tighter near touchdown
    |x_k| <= tan(gamma) (h_k + slack_k)        glide-slope cone (soft)
    |Gamma_{k+1} - Gamma_k|, |u_x,k+1 - u_x,k| <= rate * dt
    r_N = 0, v_N = (0, -v_td)                  exact L1 penalty, always feasible

``mode="fuel"`` minimises ``sum Gamma dt`` and is used to search the final
time before ignition; ``mode="smooth"`` penalises acceleration and jerk and
is used for the locked landing burn.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from common.conic import ConicSolver, ConstraintRows, UpperQuadratic, golden_section_search, objective_value


@dataclass
class PlanConfig:
    """Per-solve settings for :class:`ConvexLandingPlanner`."""

    mode: str  # "fuel" (free-final-time search) or "smooth" (locked burn)
    rho_min: float
    rho_max: float
    gamma_now: float
    u_now: tuple[float, float]
    engine_on: bool
    max_tilt_deg: float = 60.0
    touchdown_speed: float = 1.0
    drag_model: Callable[[np.ndarray], np.ndarray] | None = None


@dataclass
class LandingPlan:
    """A solved trajectory.  Arrays use the local pad frame.

    ``states`` rows are ``(x, h, vx, vz)`` at the N+1 knots, ``controls`` are
    thrust accelerations ``(ax, az)`` at the knots (linearly interpolated
    between them) and ``gammas`` the thrust-magnitude slack of the lossless
    relaxation.
    """

    t_f: float
    dt: float
    states: np.ndarray
    controls: np.ndarray
    gammas: np.ndarray
    drag: np.ndarray
    status: str
    objective: float
    terminal_position_error: float
    terminal_velocity_error: float
    glide_slope_violation: float = 0.0
    created_at: float = 0.0
    config: PlanConfig | None = field(default=None, repr=False)

    @property
    def knots(self) -> int:
        return len(self.gammas) - 1

    def sample(self, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return (position, velocity, thrust-accel, drag) at ``tau`` s after creation."""
        n = self.knots
        g = np.array([0.0, -ConvexLandingPlanner.gravity])
        if tau >= self.t_f:
            last = self.states[-1]
            return last[:2].copy(), last[2:].copy(), self.controls[-1].copy(), self.drag[-1].copy()
        tau = max(0.0, tau)
        k = min(int(tau / self.dt), n - 1)
        t = tau - k * self.dt
        u0 = self.controls[k]
        u1 = self.controls[k + 1]
        du = (u1 - u0) / self.dt
        d = self.drag[k]
        p = self.states[k, :2]
        v = self.states[k, 2:]
        accel = u0 + du * t
        vel = v + u0 * t + 0.5 * du * t * t + (d + g) * t
        pos = p + v * t + 0.5 * u0 * t * t + du * t**3 / 6.0 + 0.5 * (d + g) * t * t
        return pos, vel, accel, d.copy()

    @property
    def reaches_pad(self) -> bool:
        """True when the plan lands on the pad without bending any soft limit."""
        return (
            self.terminal_position_error < 0.3
            and self.terminal_velocity_error < 0.2
            and self.glide_slope_violation < 1.0
        )

    def gamma_at(self, tau: float) -> float:
        if tau >= self.t_f:
            return float(self.gammas[-1])
        tau = max(0.0, tau)
        k = min(int(tau / self.dt), self.knots - 1)
        f = (tau - k * self.dt) / self.dt
        return float((1.0 - f) * self.gammas[k] + f * self.gammas[k + 1])


class ConvexLandingPlanner:
    """Lossless-convexification powered-descent SOCP (Clarabel)."""

    knots = 40
    gravity = 9.80665
    # Objective weights.
    terminal_position_weight = 400.0
    terminal_velocity_weight = 250.0
    glide_slope_weight = 30.0
    # Soft cone of half-angle 65 deg (from vertical) above the pad: keeps the
    # approach from skimming the ground without making the problem infeasible.
    glide_slope_tan = math.tan(math.radians(65.0))

    # Rate at which the planned lateral thrust may change (m/s^3).  It keeps
    # the requested thrust direction within what the TVC can rotate the body
    # through, so the plan is flyable rather than merely optimal.
    lateral_rate = 4.0

    def __init__(self, max_acceleration: float = 18.0, throttle_rate: float = 1.2) -> None:
        self.max_acceleration = float(max_acceleration)
        # Thrust-magnitude rate the plan may use (80 % of the engine limit).
        self.gamma_rate = 0.8 * throttle_rate * self.max_acceleration
        self.solve_count = 0
        self.last_status = "idle"
        self.solver = ConicSolver()

    @property
    def backend(self) -> str:
        return self.solver.backend

    @property
    def available(self) -> bool:
        return self.solver.available

    @staticmethod
    def _tilt_limit(time_to_go: np.ndarray, far_deg: float) -> np.ndarray:
        """Maximum thrust tilt (rad) versus time-to-go.

        Large diverts are allowed early; the last ~6 s are flown almost
        vertically (20 deg at 6 s, 3 deg at 2 s), like a real landing burn.
        """
        far = max(20.0, far_deg)
        deg = np.interp(time_to_go, [0.0, 2.0, 6.0, 15.0], [3.0, 3.0, 20.0, far])
        return np.radians(deg)

    def build(self, state, t_f: float, cfg: PlanConfig):
        """Assemble the Clarabel data ``(P, q, A, b, cone sizes, layout)``."""
        # More knots on long flights keeps each step near 1.2 s (drag and the
        # ignition time are resolved finely), capped for solve time.
        N = int(np.clip(round(t_f / 1.2), self.knots, 70))
        dt = t_f / N
        g = self.gravity
        nS = 4 * (N + 1)
        iu = nS
        ig = iu + 2 * (N + 1)
        iep = ig + (N + 1)
        iem = iep + 4
        isg = iem + 4
        n = isg + N

        def S(k, c):
            return 4 * k + c

        def U(k, c):
            return iu + 2 * k + c

        def G(k):
            return ig + k

        mid_times = (np.arange(N) + 0.5) * dt
        if cfg.drag_model is not None:
            drag = np.asarray(cfg.drag_model(mid_times), dtype=float).reshape(N, 2)
        else:
            drag = np.zeros((N, 2))

        row = ConstraintRows()

        # ---- zero cone: initial state, FOH dynamics, terminal target -----
        for c in range(4):
            row([(S(0, c), 1.0)], state[c])
        h2a = dt * dt / 3.0
        h2b = dt * dt / 6.0
        for k in range(N):
            dx, dz = drag[k]
            row([(S(k + 1, 0), 1.0), (S(k, 0), -1.0), (S(k, 2), -dt), (U(k, 0), -h2a), (U(k + 1, 0), -h2b)],
                0.5 * dt * dt * dx)
            row([(S(k + 1, 1), 1.0), (S(k, 1), -1.0), (S(k, 3), -dt), (U(k, 1), -h2a), (U(k + 1, 1), -h2b)],
                0.5 * dt * dt * (dz - g))
            row([(S(k + 1, 2), 1.0), (S(k, 2), -1.0), (U(k, 0), -0.5 * dt), (U(k + 1, 0), -0.5 * dt)],
                dt * dx)
            row([(S(k + 1, 3), 1.0), (S(k, 3), -1.0), (U(k, 1), -0.5 * dt), (U(k + 1, 1), -0.5 * dt)],
                dt * (dz - g))
        target = (0.0, 0.0, 0.0, -abs(cfg.touchdown_speed))
        for c in range(4):
            row([(S(N, c), 1.0), (iep + c, -1.0), (iem + c, 1.0)], target[c])
        n_zero = len(row)

        # ---- nonnegative cone (rows mean  A x <= b) ----------------------
        for c in range(4):
            row([(iep + c, -1.0)], 0.0)
            row([(iem + c, -1.0)], 0.0)
        for k in range(N):
            row([(isg + k, -1.0)], 0.0)
        rho_min = max(0.0, min(cfg.rho_min, cfg.rho_max))
        for k in range(N + 1):
            row([(G(k), 1.0)], cfg.rho_max)
            # Knot 0 is "now" and is pinned to the running engine instead.
            if rho_min > 0.0 and k > 0:
                row([(G(k), -1.0)], -rho_min)
        time_to_go = t_f - np.arange(N + 1) * dt
        cos_tilt = np.cos(self._tilt_limit(time_to_go, cfg.max_tilt_deg))
        # Knot 0 is the thrust being produced right now; it only has to point
        # upward-ish (the engine cannot pull), later knots follow the schedule.
        cos_tilt[0] = min(cos_tilt[0], math.cos(math.radians(80.0)))
        for k in range(N + 1):
            # Thrust tilt:  ||u|| cos(theta) <= u_z   (with Gamma = ||u||).
            row([(G(k), float(cos_tilt[k])), (U(k, 1), -1.0)], 0.0)
        tg = self.glide_slope_tan
        for k in range(1, N + 1):
            row([(S(k, 0), 1.0), (S(k, 1), -tg), (isg + k - 1, -1.0)], 0.0)
            row([(S(k, 0), -1.0), (S(k, 1), -tg), (isg + k - 1, -1.0)], 0.0)
        rate = self.gamma_rate * dt
        lateral = self.lateral_rate * dt
        for k in range(N):
            row([(G(k + 1), 1.0), (G(k), -1.0)], rate)
            row([(G(k), 1.0), (G(k + 1), -1.0)], rate)
            row([(U(k + 1, 0), 1.0), (U(k, 0), -1.0)], lateral)
            row([(U(k, 0), 1.0), (U(k + 1, 0), -1.0)], lateral)
        # The first knot is "now": it must agree with the running engine.
        spool = self.gamma_rate * 0.15
        row([(G(0), 1.0)], cfg.gamma_now + spool)
        row([(G(0), -1.0)], -(cfg.gamma_now - spool))
        if cfg.engine_on:
            slew = max(1.0, 0.3 * self.lateral_rate)
            for c in range(2):
                row([(U(0, c), 1.0)], cfg.u_now[c] + slew + (spool if c else 0.0))
                row([(U(0, c), -1.0)], -(cfg.u_now[c] - slew - (spool if c else 0.0)))
        n_nonneg = len(row) - n_zero

        # ---- second-order cones: ||(ux, uz)|| <= Gamma ------------------
        for k in range(N + 1):
            row([(G(k), -1.0)], 0.0)
            row([(U(k, 0), -1.0)], 0.0)
            row([(U(k, 1), -1.0)], 0.0)
        soc_dims = [3] * (N + 1)

        A = row.matrix(n)
        b_vec = row.rhs

        # ---- objective ---------------------------------------------------
        w = np.full(N + 1, dt)
        w[0] *= 0.5
        w[-1] *= 0.5  # trapezoidal quadrature weights
        hessian = UpperQuadratic(n)
        add_p = hessian.add
        q = np.zeros(n)
        constant = 0.0

        if cfg.mode == "fuel":
            fuel_w, accel_w, jerk_w = 1.0, 0.0, 0.004
        else:
            # Smooth landing burn: even thrust (acceleration energy) and low
            # jerk dominate; a small fuel term would push braking late.
            fuel_w, accel_w, jerk_w = 0.02, 0.05, 0.25
        for k in range(N + 1):
            q[G(k)] += fuel_w * w[k]
            if accel_w > 0.0:
                # accel_w * w_k * ||u_k - (0, g)||^2
                add_p(U(k, 0), U(k, 0), 2.0 * accel_w * w[k])
                add_p(U(k, 1), U(k, 1), 2.0 * accel_w * w[k])
                q[U(k, 1)] += -2.0 * accel_w * w[k] * g
                constant += accel_w * w[k] * g * g
        jw = jerk_w / dt
        for k in range(N):
            for c in range(2):
                add_p(U(k, c), U(k, c), 2.0 * jw)
                add_p(U(k + 1, c), U(k + 1, c), 2.0 * jw)
                add_p(U(k, c), U(k + 1, c), -2.0 * jw)
        if cfg.engine_on:
            # Continuity with the thrust vector the engine is producing now.
            w0 = 0.5
            for c in range(2):
                add_p(U(0, c), U(0, c), 2.0 * w0)
                q[U(0, c)] += -2.0 * w0 * cfg.u_now[c]
                constant += w0 * cfg.u_now[c] ** 2
        for c in range(4):
            weight = self.terminal_position_weight if c < 2 else self.terminal_velocity_weight
            q[iep + c] = weight
            q[iem + c] = weight
        q[isg : isg + N] = self.glide_slope_weight
        # Tiny regularisation keeps the KKT system well conditioned.
        for i in range(n):
            add_p(i, i, 1e-7)
        P = hessian.matrix()
        layout = dict(N=N, dt=dt, nS=nS, iu=iu, ig=ig, iep=iep, iem=iem, isg=isg, n=n)
        return P, q, A, b_vec, n_zero, n_nonneg, soc_dims, layout, drag, constant

    def solve(self, state, t_f: float, cfg: PlanConfig) -> LandingPlan | None:
        if not self.available or not (t_f > 0.2) or not np.all(np.isfinite(state)):
            return None
        P, q, A, b, n_zero, n_nonneg, soc_dims, L, drag, constant = self.build(state, t_f, cfg)
        x, status = self.solver.solve(P, q, A, b, n_zero, n_nonneg, soc_dims)
        self.last_status = status
        if x is None:
            return None
        self.solve_count += 1
        N = L["N"]
        states = x[: L["nS"]].reshape(N + 1, 4)
        controls = x[L["iu"] : L["ig"]].reshape(N + 1, 2)
        gammas = x[L["ig"] : L["iep"]]
        e = x[L["iep"] : L["iep"] + 4] + x[L["iem"] : L["iem"] + 4]
        objective = objective_value(P, q, x, constant)
        return LandingPlan(
            t_f=float(t_f),
            dt=L["dt"],
            states=states.copy(),
            controls=controls.copy(),
            gammas=gammas.copy(),
            drag=drag,
            status=status,
            objective=objective,
            terminal_position_error=float(math.hypot(e[0], e[1])),
            terminal_velocity_error=float(math.hypot(e[2], e[3])),
            glide_slope_violation=float(np.max(x[L["isg"] : L["isg"] + N], initial=0.0)),
            config=cfg,
        )

    def search_final_time(self, state, lo: float, hi: float, cfg: PlanConfig, evaluations: int = 10) -> LandingPlan | None:
        """Plan whose final time (searched in [lo, hi]) minimises the objective."""
        lo = max(0.5, float(lo))
        hi = max(lo + 0.5, float(hi))
        return golden_section_search(lambda t: self.solve(state, t, cfg), lo, hi, evaluations)
