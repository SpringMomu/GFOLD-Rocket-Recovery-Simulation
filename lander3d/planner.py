"""3D powered-descent planner: G-FOLD second-order-cone program with propellant.

Second-order-cone program over N+1 knots with first-order-hold thrust:

    states   r_k (3), v_k (3), z_k = ln m_k
    controls u_k = T_k / m_k (3), sigma_k >= ||u_k||   (lossless relaxation)

    r' = v,  v' = u + g + d(t),  z' = -alpha * sigma,  alpha = 1/(Isp g0)

    ||u_k|| <= sigma_k                                    (SOC, dim 4)
    sigma_k <= T_max e^{-z0_k} (1 - (z_k - z0_k))         (thrust upper bound,
    sigma_k >= T_min e^{-z0_k} (1 - (z_k - z0_k))          first-order in z around
                                                           the max-burn mass z0_k)
    z_N >= ln(m_dry) - s_fuel                             (propellant limit)
    ||(x_k, y_k)|| <= tan(gamma) (h_k + s_k)              (glide-slope cone, soft)
    n_j . r_k >= b_j - s_j                                (keep-out half-spaces, soft:
                                                           obstacles, see separating_planes)
    u_z,k >= cos(theta(t_go)) sigma_k                     (pointing: vertical at the end)
    |sigma_{k+1} - sigma_k|, |u_{x/y,k+1} - u_{x/y,k}| <= rate * dt
    r_N = 0, v_N = (0, 0, -v_td)                           (exact L1 penalty)

Objective: fuel (sum sigma dt ~ -z_N) while searching the ignition time; a
smooth thrust profile (acceleration, jerk, lateral offset) for the locked
landing burn.  Drag d(t) is predicted along the previous plan, together with
the lift gradient ``G`` of tilting the vehicle against the flow, which enters
the dynamics linearly in ``u``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from common.conic import ConicSolver, ConstraintRows, UpperQuadratic, golden_section_search, objective_value

from .vehicle import G0


@dataclass
class PlanConfig3D:
    mode: str
    thrust_min: float          # N
    thrust_max: float          # N
    isp: float                 # s
    dry_mass: float            # kg
    sigma_now: float           # current thrust acceleration magnitude
    u_now: tuple               # current thrust acceleration vector
    engine_on: bool
    max_tilt_deg: float = 60.0
    touchdown_speed: float = 1.0
    drag_model: Callable | None = None
    target_alt: float = 0.0          # plan end height above the pad (0 = touchdown)
    u_cmd: tuple | None = None       # previous plan's command now (continuity anchor)
    target_xy: tuple = (0.0, 0.0)    # gate position (normally the pad centre)
    # Final-approach sink envelope (soft): sink <= touchdown_speed + sink_gain * h.
    # Arrive slowly with thrust ~ weight instead of braking hard into contact.
    sink_gain: float = 0.0
    # Angle-of-attack cap vs dynamic pressure ((Pa...), (deg...)): the angle
    # between the thrust and the oncoming flow (engines-first attitude).  The
    # drag model then also returns the predicted dynamic pressure and flow
    # direction at each interval.
    q_aoa: tuple | None = None
    # Obstacles (soft): callable(knot times) -> [(k, n, b), ...], each asking
    # n . r_k >= b; see separating_planes.
    keep_out: Callable | None = None


@dataclass
class KeepOutBox:
    """Obstacle for the planned base point: an oriented box, already grown
    by the vehicle's radius, length and a margin (planner coordinates)."""

    center: np.ndarray
    axes: np.ndarray          # 3x3, columns = unit axes
    half: np.ndarray          # half extents along the axes


def _inside(box, p) -> bool:
    return bool(np.all(np.abs(box.axes.T @ (np.asarray(p, float) - box.center)) < box.half))


def _segment_hit(boxes, a, b, step=1.0):
    """First box the segment a -> b passes through: (box, last point before
    it, first point after it), or None."""
    n = max(2, int(np.linalg.norm(b - a) / step) + 2)
    pts = a + np.outer(np.linspace(0.0, 1.0, n), b - a)
    first = None
    for box in boxes:
        inside = np.all(np.abs((pts - box.center) @ box.axes) < box.half, axis=1)
        if inside.any():
            i = int(np.argmax(inside))
            if first is None or i < first[0]:
                first = (i, box, inside)
    if first is None:
        return None
    i, box, inside = first
    out = np.nonzero(~inside[i:])[0]
    return box, (pts[i - 1] if i > 0 else a), (pts[i + int(out[0])] if len(out) else b)


def detour_path(boxes, start, end, z_min=-math.inf, climb_cost=5.0, rounds=4):
    """Reference path from ``start`` to ``end`` around the keep-out boxes.

    Where the straight line crosses a box, its last point before and first
    point after the box are both moved just outside one side (over the top,
    round a side, or underneath; whichever is shortest, climbing counting
    ``climb_cost`` times).  The path only serves to pick the face each knot
    must stay outside of -- the trajectory itself is optimised by the SOCP.
    """
    pts = [np.asarray(start, float), np.asarray(end, float)]
    for _ in range(rounds):
        for j in range(len(pts) - 1):
            a, b = pts[j], pts[j + 1]
            hit = _segment_hit(boxes, a, b)
            if hit is not None:
                break
        else:
            return pts
        box, p_in, p_out = hit
        l_in = box.axes.T @ (p_in - box.center)
        l_out = box.axes.T @ (p_out - box.center)
        best = None
        for i in range(3):
            for sgn in (1.0, -1.0):
                w1, w2 = l_in.copy(), l_out.copy()
                w1[i] = w2[i] = sgn * (box.half[i] + 1.0)
                w1, w2 = box.center + box.axes @ w1, box.center + box.axes @ w2
                if min(w1[2], w2[2]) < z_min or any(_inside(o, w) for o in boxes for w in (w1, w2)):
                    continue
                legs = ((a, w1), (w1, w2), (w2, b))
                if any(_segment_hit([box], p, q) is not None for p, q in legs):
                    continue
                cost = sum(float(np.linalg.norm(q - p)) + climb_cost * max(0.0, float(q[2] - p[2])) for p, q in legs)
                if best is None or cost < best[0]:
                    best = (cost, w1, w2)
        if best is None:
            return pts
        pts[j + 1:j + 1] = [best[1], best[2]]
    return pts


def along_path(pts, fractions):
    """Points at the given fractions (0..1) of the length of a polyline."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = max(cum[-1], 1e-9)
    out = []
    for f in fractions:
        s = float(np.clip(f, 0.0, 1.0)) * total
        j = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
        u = (s - cum[j]) / max(seg[j], 1e-9)
        out.append(pts[j] + (pts[j + 1] - pts[j]) * u)
    return out


def separating_planes(boxes, points, target=None, reach=60.0):
    """Linearised obstacle avoidance (successive convexification).

    Keeping a point out of a convex box is not convex, but staying on the
    outer side of one of its faces is (a half-space).  For each reference
    point (the previous plan, or a detour, at the new knot times) and each
    box within ``reach``, pick a face the point is outside of and require
    that half-space: ``[(k, n, b)]`` with n . r_k >= b.  A face the
    ``target`` is outside of too is preferred: it never conflicts with
    arriving (e.g. "stay east of the tower" rather than "stay above it",
    which would hold the descent up); otherwise the face the point is most
    clearly outside of (or least inside).  The segment between two knots on
    the same side of a face stays out of the box too; only turning a corner
    between knots can clip it, which the margin covers.
    """
    out = []
    for k, p in enumerate(points):
        for box in boxes:
            rel = box.axes.T @ (np.asarray(p, float) - box.center)
            rel_t = None if target is None else box.axes.T @ (np.asarray(target, float) - box.center)
            best = None
            for i in range(3):
                for sgn in (1.0, -1.0):
                    d = sgn * rel[i] - box.half[i]
                    shared = rel_t is not None and d >= 0.0 and sgn * rel_t[i] - box.half[i] >= 0.0
                    key = (shared, d)
                    if best is None or key > best[0]:
                        best = (key, d, i, sgn)
            _, d, i, sgn = best
            if d > reach:
                continue
            n = sgn * box.axes[:, i]
            out.append((k, n, float(n @ box.center + box.half[i])))
    return out


@dataclass
class Plan3D:
    t_f: float
    dt: float
    r: np.ndarray
    v: np.ndarray
    z: np.ndarray
    u: np.ndarray
    sigma: np.ndarray
    drag: np.ndarray
    status: str
    objective: float
    terminal_position_error: float
    terminal_velocity_error: float
    glide_violation: float
    fuel_violation: float
    # largest slack of the lateral-speed / angle-of-attack / no-climb limits
    lateral_violation: float = 0.0
    aoa_violation: float = 0.0
    climb_violation: float = 0.0
    keep_out_violation: float = 0.0
    created_at: float = 0.0

    @property
    def knots(self) -> int:
        return len(self.sigma) - 1

    @property
    def reaches_pad(self) -> bool:
        # Reaching the pad by breaking the flyability limits (much faster
        # sideways than allowed, nose-first, climbing) does not count: the
        # tracker cannot follow such a plan, so the time must move instead.
        return self.reaches()

    def reaches(self, climb_ok: bool = False) -> bool:
        """``reaches_pad``; with ``climb_ok`` a climb is allowed (e.g. a hop
        over an obstacle that stands between a low booster and the gate)."""
        return (self.terminal_position_error < 1.0 and self.terminal_velocity_error < 0.8
                and self.glide_violation < 1.0 and self.fuel_violation < 1e-3
                and self.lateral_violation < 3.0 and (climb_ok or self.climb_violation < 2.0)
                and self.keep_out_violation < 0.5)

    @property
    def final_mass(self) -> float:
        return float(math.exp(self.z[-1]))

    def sample(self, tau: float):
        g = np.array([0.0, 0.0, -G0])
        if tau >= self.t_f:
            return self.r[-1].copy(), self.v[-1].copy(), self.u[-1].copy()
        tau = max(0.0, tau)
        k = min(int(tau / self.dt), self.knots - 1)
        t = tau - k * self.dt
        u0, u1 = self.u[k], self.u[k + 1]
        du = (u1 - u0) / self.dt
        d = self.drag[k]
        pos = self.r[k] + self.v[k] * t + 0.5 * u0 * t * t + du * t ** 3 / 6.0 + 0.5 * (d + g) * t * t
        vel = self.v[k] + u0 * t + 0.5 * du * t * t + (d + g) * t
        return pos, vel, u0 + du * t

    def sigma_at(self, tau: float) -> float:
        if tau >= self.t_f:
            return float(self.sigma[-1])
        tau = max(0.0, tau)
        k = min(int(tau / self.dt), self.knots - 1)
        f = (tau - k * self.dt) / self.dt
        return float((1 - f) * self.sigma[k] + f * self.sigma[k + 1])


class Planner3D:
    knots = 40
    terminal_position_weight = 400.0
    terminal_velocity_weight = 250.0
    glide_slope_weight = 30.0
    glide_slope_tan = math.tan(math.radians(80.0))
    fuel_slack_weight = 2.0e4
    sink_envelope_weight = 80.0
    # Final approach: lateral offset / speed weights (per s) vs time-to-go (s).
    approach_t = (0.0, 6.0, 10.0, 15.0)
    approach_pos_w = (0.3, 0.2, 0.02, 0.008)
    approach_vel_w = (0.4, 0.3, 0.03, 0.03)
    vertical_final_t = 10.0
    vertical_final_w = 0.1
    lateral_position_weight = 0.008
    lateral_velocity_weight = 0.03
    lateral_jerk_factor = 8.0
    lateral_rate = 4.0             # m/s^3, limit on the change of planned lateral thrust
    # Soft cap on the horizontal speed (a low divert): flying sideways faster
    # than this needs thrust along the motion against the broadside drag,
    # i.e. nose-first flight the booster cannot hold aerodynamically.
    lateral_speed_max = 30.0       # m/s (or the current speed, bled at lateral_speed_decay)
    lateral_speed_decay = 4.0      # m/s^2
    lateral_speed_weight = 200.0
    # ... and, like the sink envelope, the lower the slower: arrive over the
    # pad nearly vertical instead of braking a sideways speed at the bottom.
    lateral_speed_floor = 2.0      # m/s at the pad
    lateral_speed_gain = 0.15      # + m/s per m of height
    aoa_weight = 100.0
    no_climb_weight = 300.0
    min_sink = 1.0                 # m/s, the slowest planned descent
    climb_decay = 1.5              # m/s^2: a climb now must end at least this fast
    approach_cone_floor = 1.0      # m off the pad centre at the pad ...
    approach_cone_gain = 0.35      # ... + m per m of height
    approach_cone_weight = 60.0
    approach_cone_t = (6.0, 10.0)  # s to go: full weight -> none
    keep_out_weight = 2000.0       # per m inside an obstacle's half-space (per s)

    def __init__(self) -> None:
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
    def _tilt_limit(t_go: np.ndarray, far_deg: float) -> np.ndarray:
        far = max(8.0, far_deg)
        return np.radians(np.interp(t_go, [0.0, 1.0, 3.0, 8.0], [min(6.0, far), min(8.0, far), min(22.0, far), far]))

    def build(self, r0, v0, m0, t_f, cfg: PlanConfig3D):
        N = int(np.clip(round(t_f / 1.0), self.knots, 70))
        dt = t_f / N
        g = G0
        alpha = 1.0 / (cfg.isp * G0)
        NX = 7
        nS = NX * (N + 1)
        iu = nS
        nU = 4 * (N + 1)
        iep = iu + nU
        iem = iep + 6
        isg = iem + 6
        ifs = isg + N
        isk = ifs + 1                  # sink-envelope slacks, knots 1..N
        ivs = isk + N                  # lateral-speed slacks, knots 1..N
        iao = ivs + N                  # angle-of-attack slacks, knots 1..N
        icl = iao + N                  # no-climb slacks, knots 1..N
        iap = icl + N                  # final-approach-cone slacks, knots 1..N
        ikp = iap + N                  # keep-out slacks, one per half-space
        t_k = np.arange(N + 1) * dt
        keep = [kp for kp in cfg.keep_out(t_k) if kp[0] >= 1] if cfg.keep_out is not None else []
        n = ikp + len(keep)

        def S(k, c):
            return NX * k + c

        def U(k, c):
            return iu + 4 * k + c

        mid = (np.arange(N) + 0.5) * dt
        Gm = None
        q_mid = w_mid = None
        if cfg.drag_model is not None:
            res = cfg.drag_model(mid)
            if isinstance(res, tuple):
                drag, Gm = np.asarray(res[0], float).reshape(N, 3), np.asarray(res[1], float).reshape(N, 3, 3)
                if len(res) > 3:
                    q_mid, w_mid = np.asarray(res[2], float).reshape(N), np.asarray(res[3], float).reshape(N, 3)
            else:
                drag = np.asarray(res, float).reshape(N, 3)
        else:
            drag = np.zeros((N, 3))
        m_ref = np.maximum(cfg.dry_mass, m0 - alpha * cfg.thrust_max * t_k)
        z0 = np.log(m_ref)
        mu2 = cfg.thrust_max * np.exp(-z0)
        mu1 = cfg.thrust_min * np.exp(-z0)

        row = ConstraintRows()

        # ---- zero cone
        for c in range(3):
            row([(S(0, c), 1.0)], r0[c])
            row([(S(0, 3 + c), 1.0)], v0[c])
        row([(S(0, 6), 1.0)], math.log(m0))
        h2a, h2b = dt * dt / 3.0, dt * dt / 6.0
        for k in range(N):
            dk = drag[k]
            for c in range(3):
                gc = -g if c == 2 else 0.0
                lift_p, lift_v = [], []
                if Gm is not None:
                    # Body lift from tilting the thrust (and therefore the
                    # vehicle) against the flow: linear in u around the
                    # reference thrust magnitude.
                    for j in range(3):
                        if Gm[k, c, j] != 0.0:
                            lift_p.append((U(k, j), -0.5 * dt * dt * Gm[k, c, j]))
                            lift_v.append((U(k, j), -dt * Gm[k, c, j]))
                row([(S(k + 1, c), 1.0), (S(k, c), -1.0), (S(k, 3 + c), -dt), (U(k, c), -h2a), (U(k + 1, c), -h2b)] + lift_p,
                    0.5 * dt * dt * (dk[c] + gc))
                row([(S(k + 1, 3 + c), 1.0), (S(k, 3 + c), -1.0), (U(k, c), -0.5 * dt), (U(k + 1, c), -0.5 * dt)] + lift_v,
                    dt * (dk[c] + gc))
            row([(S(k + 1, 6), 1.0), (S(k, 6), -1.0), (U(k, 3), 0.5 * dt * alpha), (U(k + 1, 3), 0.5 * dt * alpha)], 0.0)
        target = (float(cfg.target_xy[0]), float(cfg.target_xy[1]), float(cfg.target_alt), 0.0, 0.0, -abs(cfg.touchdown_speed))
        for c in range(6):
            row([(S(N, c), 1.0), (iep + c, -1.0), (iem + c, 1.0)], target[c])
        n_zero = len(row)

        # ---- nonnegative cone (A x <= b)
        for c in range(6):
            row([(iep + c, -1.0)], 0.0)
            row([(iem + c, -1.0)], 0.0)
        for k in range(N):
            row([(isg + k, -1.0)], 0.0)
        row([(ifs, -1.0)], 0.0)
        for k in range(N):
            row([(isk + k, -1.0)], 0.0)
        for k in range(N):
            row([(ivs + k, -1.0)], 0.0)
        for k in range(N):
            row([(iao + k, -1.0)], 0.0)
        for k in range(N):
            row([(icl + k, -1.0)], 0.0)
        for k in range(N):
            row([(iap + k, -1.0)], 0.0)
        # Obstacles: n . r_k + s >= b  ->  -n . r_k - s <= -b
        for j, (k, nrm, b_) in enumerate(keep):
            row([(ikp + j, -1.0)], 0.0)
            row([(S(k, c), -float(nrm[c])) for c in range(3) if nrm[c] != 0.0] + [(ikp + j, -1.0)], -b_)
        # The landing burn only descends (soft): a divert is flown sinking,
        # never by climbing back up or hovering to wait.
        for k in range(1, N + 1):
            row([(S(k, 5), 1.0), (icl + k - 1, -1.0)], max(-self.min_sink, float(v0[2]) - self.climb_decay * t_k[k]))
        if cfg.sink_gain > 0.0:
            # vz_k >= -(v_td + c (h_k - h_target)) - s_k   (linear in the state)
            c_s = cfg.sink_gain
            v_td = abs(cfg.touchdown_speed)
            for k in range(1, N + 1):
                row([(S(k, 5), -1.0), (S(k, 2), -c_s), (isk + k - 1, -1.0)], v_td - c_s * cfg.target_alt)
        for k in range(N + 1):
            # sigma <= mu2 (1 - z + z0)  ->  sigma + mu2 z <= mu2 (1 + z0)
            row([(U(k, 3), 1.0), (S(k, 6), mu2[k])], mu2[k] * (1.0 + z0[k]))
            if cfg.thrust_min > 0.0 and t_k[k] >= 0.6:
                row([(U(k, 3), -1.0), (S(k, 6), -mu1[k])], -mu1[k] * (1.0 + z0[k]))
        # propellant: z_N >= ln(m_dry) - s_fuel
        row([(S(N, 6), -1.0), (ifs, -1.0)], -math.log(cfg.dry_mass))
        t_go = t_f - t_k
        tilt = self._tilt_limit(t_go, cfg.max_tilt_deg)
        if q_mid is not None and cfg.q_aoa is not None:
            # Same angle-of-attack cap as the tracker (knots 1..N, flow of the
            # interval before): the plan must not count on an attitude the
            # booster cannot hold against the aerodynamic moment -- e.g.
            # flying nose-first towards the pad in a fast low divert.  Kept
            # compatible with the tilt cone, so the problem stays feasible.
            for k in range(1, N + 1):
                w = w_mid[k - 1]
                if q_mid[k - 1] <= cfg.q_aoa[0][0] or not np.any(w):
                    continue
                aoa = math.radians(float(np.interp(q_mid[k - 1], *cfg.q_aoa)))
                aoa = max(aoa, math.acos(float(np.clip(w[2], -1.0, 1.0))) - tilt[k] + math.radians(3.0))
                if aoa < math.radians(170.0):
                    # soft (slack in m/s^2 of thrust off the allowed cone):
                    # thrust-rate limits may not allow an instant turn
                    row([(U(k, 3), math.cos(aoa)), (iao + k - 1, -1.0)] + [(U(k, c), -float(w[c])) for c in range(3)], 0.0)
        # The first knot may keep the current thrust direction (attitude lag),
        # but no more: a looser first knot is flown for the whole replan
        # period and the tracker clips it anyway.
        un = np.asarray(cfg.u_now, float)
        tilt_now = math.atan2(float(np.linalg.norm(un[:2])), max(1e-6, float(un[2]))) if cfg.engine_on else math.radians(75.0)
        tilt[0] = max(tilt[0], min(math.radians(80.0), tilt_now + math.radians(5.0)))
        cos_tilt = np.cos(tilt)
        for k in range(N + 1):
            row([(U(k, 3), float(cos_tilt[k])), (U(k, 2), -1.0)], 0.0)
        rate_s = 0.8 * 1.2 * cfg.thrust_max / m0 * dt
        lat = self.lateral_rate * dt
        for k in range(N):
            row([(U(k + 1, 3), 1.0), (U(k, 3), -1.0)], rate_s)
            row([(U(k, 3), 1.0), (U(k + 1, 3), -1.0)], rate_s)
            for c in (0, 1):
                row([(U(k + 1, c), 1.0), (U(k, c), -1.0)], lat)
                row([(U(k, c), 1.0), (U(k + 1, c), -1.0)], lat)
        spool = 0.8 * 1.2 * cfg.thrust_max / m0 * 0.15
        row([(U(0, 3), 1.0)], cfg.sigma_now + spool)
        row([(U(0, 3), -1.0)], -(cfg.sigma_now - spool))
        if cfg.engine_on:
            # Vertical thrust is pinned near the current value (engine spool);
            # the lateral components are only pulled toward it by the cost,
            # so a lagging attitude never makes the problem infeasible.
            row([(U(0, 2), 1.0)], cfg.u_now[2] + spool + 1.0)
            row([(U(0, 2), -1.0)], -(cfg.u_now[2] - spool - 1.0))
        n_nonneg = len(row) - n_zero

        # ---- second-order cones
        soc_dims = []
        for k in range(N + 1):
            row([(U(k, 3), -1.0)], 0.0)
            row([(U(k, 0), -1.0)], 0.0)
            row([(U(k, 1), -1.0)], 0.0)
            row([(U(k, 2), -1.0)], 0.0)
            soc_dims.append(4)
        tg = self.glide_slope_tan
        for k in range(1, N + 1):
            row([(S(k, 2), -tg), (isg + k - 1, -tg)], 0.0)
            row([(S(k, 0), -1.0)], 0.0)
            row([(S(k, 1), -1.0)], 0.0)
            soc_dims.append(3)

        v_cap = np.maximum(self.lateral_speed_max, math.hypot(v0[0], v0[1]) - self.lateral_speed_decay * t_k)
        for k in range(1, N + 1):
            row([(ivs + k - 1, -1.0)], float(v_cap[k]))
            row([(S(k, 3), -1.0)], 0.0)
            row([(S(k, 4), -1.0)], 0.0)
            soc_dims.append(3)
            row([(ivs + k - 1, -1.0), (S(k, 2), -self.lateral_speed_gain)],
                self.lateral_speed_floor - self.lateral_speed_gain * cfg.target_alt)
            row([(S(k, 3), -1.0)], 0.0)
            row([(S(k, 4), -1.0)], 0.0)
            soc_dims.append(3)
            # Final approach cone (soft), much steeper than the glide slope:
            # be over the pad before the last ~50 m, then come down upright.
            row([(iap + k - 1, -1.0), (S(k, 2), -self.approach_cone_gain)],
                self.approach_cone_floor - self.approach_cone_gain * cfg.target_alt)
            row([(S(k, 0), -1.0)], -float(cfg.target_xy[0]))
            row([(S(k, 1), -1.0)], -float(cfg.target_xy[1]))
            soc_dims.append(3)

        A = row.matrix(n)
        b_vec = row.rhs

        # ---- objective
        w = np.full(N + 1, dt)
        w[0] *= 0.5
        w[-1] *= 0.5
        hessian = UpperQuadratic(n)
        add_p = hessian.add
        q = np.zeros(n)
        const = 0.0

        if cfg.mode == "fuel":
            fuel_w, accel_w, jerk_w = 1.0, 0.0, 0.004
        else:
            fuel_w, accel_w, jerk_w = 0.02, 0.05, 0.25
        for k in range(N + 1):
            q[U(k, 3)] += fuel_w * w[k]
            if accel_w > 0:
                for c in range(3):
                    add_p(U(k, c), U(k, c), 2 * accel_w * w[k])
                q[U(k, 2)] += -2 * accel_w * w[k] * g
                const += accel_w * w[k] * g * g
        # Horizontal distance from the pad axis: pulls lateral corrections
        # early, while the vehicle is high and tilting is cheap.
        # Horizontal distance from the pad axis, and lateral speed: weighted
        # up as touchdown approaches, so the divert is done early (high) and
        # the last seconds are flown (almost) vertically -- with the burn
        # running all the way to contact there is no final phase to clean up
        # a late lateral error.  Quadratic costs, not constraints.
        if cfg.mode != "fuel":
            tgo_k = t_f - t_k
            wr = np.maximum(self.lateral_position_weight, np.interp(tgo_k, self.approach_t, self.approach_pos_w))
            wvv = np.maximum(self.lateral_velocity_weight, np.interp(tgo_k, self.approach_t, self.approach_vel_w))
            tx, ty = float(cfg.target_xy[0]), float(cfg.target_xy[1])
            for k in range(1, N + 1):
                for c_, t_ in ((0, tx), (1, ty)):
                    add_p(S(k, c_), S(k, c_), 2 * wr[k] * w[k])
                    q[S(k, c_)] += -2 * wr[k] * w[k] * t_
                    const += wr[k] * w[k] * t_ * t_
                add_p(S(k, 3), S(k, 3), 2 * wvv[k] * w[k])
                add_p(S(k, 4), S(k, 4), 2 * wvv[k] * w[k])
            # in the last seconds sideways thrust (= tilt) itself is costly
            for k in range(N + 1):
                if tgo_k[k] < self.vertical_final_t:
                    f = 1.0 - tgo_k[k] / self.vertical_final_t
                    for c_ in (0, 1):
                        add_p(U(k, c_), U(k, c_), 2 * self.vertical_final_w * f * w[k])
        jw = jerk_w / dt
        for k in range(N):
            for c in range(3):
                # Lateral jerk costs more: lateral thrust means tilting the
                # whole vehicle, and fast tilt reversals are what look like rocking.
                jc = jw * (self.lateral_jerk_factor if c < 2 and cfg.mode != "fuel" else 1.0)
                add_p(U(k, c), U(k, c), 2 * jc)
                add_p(U(k + 1, c), U(k + 1, c), 2 * jc)
                add_p(U(k, c), U(k + 1, c), -2 * jc)
        if cfg.engine_on:
            # Continuity with the command being flown (not with the lagging
            # measured thrust: anchoring to that makes every new plan postpone
            # the lateral correction a little more).
            # Blend: mostly what the engine is really doing now (attitude lag
            # is real), a little of the previous command (no jumps).
            anchor = cfg.u_now if cfg.u_cmd is None else tuple(0.7 * a + 0.3 * b for a, b in zip(cfg.u_now, cfg.u_cmd))
            for c in range(2):
                add_p(U(0, c), U(0, c), 1.5)
                q[U(0, c)] += -1.5 * anchor[c]
                const += 0.75 * anchor[c] ** 2
        for c in range(6):
            wgt = self.terminal_position_weight if c < 3 else self.terminal_velocity_weight
            q[iep + c] = wgt
            q[iem + c] = wgt
        q[isg:isg + N] = self.glide_slope_weight
        q[ifs] = self.fuel_slack_weight
        q[isk:isk + N] = self.sink_envelope_weight
        q[ivs:ivs + N] = self.lateral_speed_weight * dt
        q[iao:iao + N] = self.aoa_weight * dt
        q[icl:icl + N] = self.no_climb_weight * dt
        # only the last seconds: earlier a low, far booster must not be
        # pushed up to get inside it
        q[iap:iap + N] = self.approach_cone_weight * dt * np.interp(t_f - t_k[1:], self.approach_cone_t, [1.0, 0.0])
        q[ikp:n] = self.keep_out_weight * dt
        for i in range(n):
            add_p(i, i, 1e-7)
        P = hessian.matrix()
        layout = dict(N=N, dt=dt, nS=nS, iu=iu, iep=iep, iem=iem, isg=isg, ifs=ifs, ivs=ivs, iao=iao, icl=icl, ikp=ikp, n=n,
                      NX=NX, G=Gm)
        return P, q, A, b_vec, n_zero, n_nonneg, soc_dims, layout, drag, const

    def solve(self, r0, v0, m0, t_f, cfg: PlanConfig3D) -> Plan3D | None:
        if not self.available or not (t_f > 0.3):
            return None
        P, q, A, b, nz, nn, socs, L, drag, const = self.build(r0, v0, m0, t_f, cfg)
        x, status = self.solver.solve(P, q, A, b, nz, nn, socs)
        self.last_status = status
        if x is None:
            return None
        self.solve_count += 1
        N = L["N"]
        X = x[: L["nS"]].reshape(N + 1, L["NX"])
        Uc = x[L["iu"]: L["iep"]].reshape(N + 1, 4)
        if L["G"] is not None:
            drag = drag + np.einsum("kij,kj->ki", L["G"], Uc[:N, 0:3])
        e = x[L["iep"]: L["iep"] + 6] + x[L["iem"]: L["iem"] + 6]
        obj = objective_value(P, q, x, const)
        return Plan3D(
            t_f=float(t_f), dt=L["dt"], r=X[:, 0:3].copy(), v=X[:, 3:6].copy(), z=X[:, 6].copy(),
            u=Uc[:, 0:3].copy(), sigma=Uc[:, 3].copy(), drag=drag, status=status, objective=obj,
            terminal_position_error=float(np.linalg.norm(e[0:3])), terminal_velocity_error=float(np.linalg.norm(e[3:6])),
            glide_violation=float(np.max(x[L["isg"]: L["isg"] + N], initial=0.0)), fuel_violation=float(max(0.0, x[L["ifs"]])),
            lateral_violation=float(np.max(x[L["ivs"]: L["ivs"] + N], initial=0.0)),
            aoa_violation=float(np.max(x[L["iao"]: L["iao"] + N], initial=0.0)),
            climb_violation=float(np.max(x[L["icl"]: L["icl"] + N], initial=0.0)),
            keep_out_violation=float(np.max(x[L["ikp"]: L["n"]], initial=0.0)),
        )

    def search(self, r0, v0, m0, lo, hi, cfg, evaluations=10):
        """Plan whose final time (searched in [lo, hi]) minimises the objective."""
        lo = max(0.6, float(lo))
        hi = max(lo + 0.5, float(hi))
        return golden_section_search(lambda t: self.solve(r0, v0, m0, t, cfg), lo, hi, evaluations)
