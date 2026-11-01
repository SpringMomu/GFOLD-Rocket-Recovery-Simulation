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
    h_k >= h_ground - s_k                                 (the ground, soft)
    n_j . r_k >= b_j - s_j                                (keep-out half-spaces, soft:
                                                           obstacles, see separating_planes)
    w_k . u_k >= cos(alpha(q_k)) sigma_k - s_k            (angle of attack the booster can
                                                           hold at dynamic pressure q_k, soft)
    vz_k >= -(v_td + c h_k) - s_k                         (touchdown approach, soft)
    u . a_k >= cos(theta(t_go)) sigma_k                   (upright at touchdown)
    |sigma_{k+1} - sigma_k|, |u_{x/y,k+1} - u_{x/y,k}| <= rate * dt
    r_N = 0, v_N = (0, 0, -v_td)                           (exact L1 penalty)

Only physical limits (vehicle, actuators, aerodynamics, ground, obstacles)
and the touchdown condition are constraints: the route -- when to brake,
how fast and how high to fly, whether to go over an obstacle or round it --
is the optimiser's.  Objective: propellant (sum sigma dt) plus
``time_weight`` per second of flight (the final time is searched on this
total by the caller), with a small jerk term for the actuators.  Drag d(t)
is predicted along the previous plan, together with the lift gradient ``G``
of tilting the vehicle against the flow, which enters the dynamics linearly
in ``u``.
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
    # Touchdown approach (soft): sink <= touchdown_speed + sink_gain * h --
    # loose high up, it only keeps the last metres slow (no braking hard into
    # contact), like the tracker's envelope.
    sink_gain: float = 0.0
    # No hover near the pad (soft): sink at least min_sink between these
    # heights above the gate (at the reference heights, height_ref).
    min_sink: float = 0.0
    min_sink_band: tuple = (1.5, 10.0)
    # Angle-of-attack cap vs dynamic pressure ((Pa...), (deg...)): the angle
    # between the thrust and the oncoming flow (engines-first attitude).  The
    # drag model then also returns the predicted dynamic pressure and flow
    # direction at each interval.
    q_aoa: tuple | None = None
    # Obstacles (soft): callable(knot times) -> [(k, n, b), ...], each asking
    # n . r_k >= b; see separating_planes.
    keep_out: Callable | None = None
    # The ground (planner coordinates of the base point; soft, like the
    # obstacles).
    ground_alt: float = -math.inf
    # Length from the base to the top of the body: beside an obstacle the
    # tilted body's top must stay on the same side as the base (the boxes
    # are grown upwards by the length, which only covers an upright body).
    body_length: float = 0.0
    # Wind trim (m/s^2, x and y): the sideways thrust acceleration that holds
    # the upright booster against the wind near the end.  The tilt cone is
    # centred on the trim axis (lean into the wind) instead of the vertical,
    # like the tracker's tilt budget.
    tilt_trim: tuple = (0.0, 0.0)
    trim_fade: tuple = (0.5, 5.5)    # m above the gate: upright below, full trim above
    # The tracker's tilt budget by height ((m...), (deg...)): the plan must
    # not count on a late sideways swing the tracker will not fly near the
    # ground.  Applied at the reference heights ``height_ref(knot times)``
    # (the previous plan: successive linearisation), else at the height of
    # the touchdown approach t_go before the end.
    tilt_budget: tuple | None = None
    height_ref: Callable | None = None
    # The tracker's slew-rate limit on the thrust direction by height ((m...),
    # (deg/s...)): caps the change of the planned lateral thrust (at the
    # reference heights, with a margin for the attitude lag) -- a reversal
    # faster than the booster can turn is flown late and overshoots.
    slew_deg: tuple | None = None
    # Reference thrust direction (unit vectors) at given times, for the slew
    # limit: throttling up along a tilted direction is not turning (at
    # ignition the engine goes from nothing to full along a body already
    # aligned with a slanted flow).  None: vertical.
    dir_ref: Callable | None = None
    # Cost of a second of flight, in propellant units (m/s of thrust
    # acceleration integrated): the final time is chosen on propellant +
    # time_weight * t_f -- the cheapest route, and of nearly equal ones the
    # fastest.  (Gravity losses alone already make every extra second cost
    # ~g: hovering is never optimal.)
    time_weight: float = 0.0


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


def _detour_options(boxes, a, b, z_min):
    """Ways past the first box the segment a -> b runs into: for each face of
    that box, its last point before and first point after the box moved just
    outside that face (over the top, round a side, underneath).  Returns
    (index of the segment's hit, [(length, w1, w2), ...] shortest first), or
    None if the segment is clear."""
    hit = _segment_hit(boxes, a, b)
    if hit is None:
        return None
    box, p_in, p_out = hit
    l_in = box.axes.T @ (p_in - box.center)
    l_out = box.axes.T @ (p_out - box.center)
    opts = []
    for i in range(3):
        for sgn in (1.0, -1.0):
            # just outside this face -- or further out, past whatever else
            # stands there (round the tower's side means round the arms too)
            for extra in np.arange(1.0, 31.0, 1.0):
                w1, w2 = l_in.copy(), l_out.copy()
                w1[i] = w2[i] = sgn * (box.half[i] + extra)
                w1, w2 = box.center + box.axes @ w1, box.center + box.axes @ w2
                if not any(_inside(o, w) for o in boxes for w in (w1, w2)):
                    break
            else:
                continue
            if min(w1[2], w2[2]) < z_min:
                continue
            legs = ((a, w1), (w1, w2), (w2, b))
            if any(_segment_hit([box], p, q) is not None for p, q in legs):
                continue
            opts.append((sum(float(np.linalg.norm(q - p)) for p, q in legs), w1, w2))
    opts.sort(key=lambda o: o[0])
    return opts


def detour_path(boxes, start, end, z_min=-math.inf, rounds=4, first=0):
    """Reference path from ``start`` to ``end`` around the keep-out boxes.

    Where the straight line crosses a box it is bent just outside one of
    the box's faces; ``first`` picks which way past the first obstacle (0 =
    the shortest), later ones take the shortest.  The path only serves to
    pick the face each knot must stay outside of -- the trajectory itself,
    and which way it goes, is the optimiser's (see ``detour_paths``).
    """
    pts = [np.asarray(start, float), np.asarray(end, float)]
    for r in range(rounds):
        for j in range(len(pts) - 1):
            opts = _detour_options(boxes, pts[j], pts[j + 1], z_min)
            if opts is not None:
                break
        else:
            return pts
        if not opts:
            return pts
        _, w1, w2 = opts[min(first if r == 0 else 0, len(opts) - 1)]
        pts[j + 1:j + 1] = [w1, w2]
    return pts


def detour_paths(boxes, start, end, z_min=-math.inf, limit=3, near=15.0):
    """Distinct ways round the obstacles (over, round either side ...):
    candidate references, one convex problem each -- the cheapest plan picks
    the route.  With obstacles close round the end (the catch arms) one more:
    to a point straight above it, clear of them, then straight down -- the
    way in between two obstacles open at the top, which no single bend
    round one of them finds."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    opts = _detour_options(boxes, start, end, z_min)
    paths = ([detour_path(boxes, start, end, z_min, first=i) for i in range(min(limit, len(opts)))]
             if opts else [detour_path(boxes, start, end, z_min)])
    tops = [float(b.center[2] + np.sum(np.abs(b.axes[2]) * b.half)) for b in boxes
            if float(np.linalg.norm(_closest_xy(b, end) - end[:2])) < near]
    if tops:
        above = np.array([end[0], end[1], max(max(tops) + 2.0, end[2])])
        if not any(_inside(b, above) for b in boxes) and _segment_hit(boxes, above, end) is None:
            paths.append(detour_path(boxes, start, above, z_min)[:-1] + [above, end])
    return paths


def _closest_xy(box, p):
    """Horizontal point of the box nearest to ``p`` (ignoring height)."""
    rel = box.axes.T @ (np.asarray(p, float) - box.center)
    rel = np.clip(rel, -box.half, box.half)
    q = box.center + box.axes @ rel
    return q[:2]


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
    fuel_violation: float
    # largest slacks of the soft physical limits
    aoa_violation: float = 0.0        # m/s^2 of thrust outside the angle-of-attack cone
    keep_out_violation: float = 0.0   # m inside an obstacle
    ground_violation: float = 0.0     # m below the ground
    limit_violation: float = 0.0      # m/s^2 outside the tilt cone / slew rate / first-knot thrust (now)
    created_at: float = 0.0

    @property
    def knots(self) -> int:
        return len(self.sigma) - 1

    @property
    def reaches_pad(self) -> bool:
        # Reaching the pad by breaking a physical limit (through an obstacle
        # or the ground, nose-first beyond what the air allows) does not
        # count: the tracker cannot fly such a plan.
        return (self.terminal_position_error < 1.0 and self.terminal_velocity_error < 0.8
                and self.fuel_violation < 1e-3 and self.keep_out_violation < 0.5
                and self.ground_violation < 0.5 and self.aoa_violation < 2.0 and self.limit_violation < 1.0)

    @property
    def final_mass(self) -> float:
        return float(math.exp(self.z[-1]))

    def sample(self, tau: float):
        g = np.array([0.0, 0.0, -G0])
        if tau >= self.t_f:
            # Past the end: carry on at the final velocity (on the pad,
            # sinking at touchdown speed), with the thrust that just holds
            # it.  Holding the end point and the last knot's (braking)
            # thrust instead balances the feedback above the pad -- a hover.
            return self.r[-1] + self.v[-1] * (tau - self.t_f), self.v[-1].copy(), -(g + self.drag[-1])
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
    fuel_slack_weight = 2.0e4
    sink_envelope_weight = 80.0
    jerk_weight = 0.1
    # Effort of sideways manoeuvring (per (m/s^2)^2 s): the net sideways
    # acceleration, thrust plus the predicted aerodynamic force -- holding
    # still against a cross wind costs nothing here (that thrust is needed),
    # swinging about does.  In propellant a sideways acceleration a costs
    # only ~a^2 / 2g per second, so a purely propellant-optimal plan swings
    # hard and brakes late sideways -- with the attitude lag the tracker
    # then overshoots (round and round the catch point).  It says nothing
    # about the route.  (On the thrust alone it made fighting the wind
    # expensive: the plan let the wind carry the booster off and came back
    # low and late.)
    lateral_accel_weight = 1.0
    # Touchdown phase (not the route): over the last ~10 s be over the pad
    # and nearly still sideways -- the burn runs to contact, so a lateral
    # error left for the last metres can only be fixed by a swing the
    # tracker will not fly near the ground.  Weights (per s) vs time to go.
    approach_t = (0.0, 6.0, 10.0)
    approach_pos_w = (0.3, 0.2, 0.0)
    approach_vel_w = (0.4, 0.3, 0.0)
    lateral_jerk_factor = 8.0      # lateral thrust changes mean tilting the whole vehicle
    lateral_rate = 4.0             # m/s^3, limit on the change of planned lateral thrust
    slew_margin = 0.7              # share of the tracker's slew rate the plan may use
    aoa_weight = 100.0             # per m/s^2 of thrust outside the angle-of-attack cone (per s)
    keep_out_weight = 2000.0       # per m inside an obstacle's half-space (per s)
    ground_weight = 2000.0         # per m below the ground (per s)
    elastic_weight = 1.0e4         # per m/s^2 outside a state-tied limit (tilt cone, slew, first knot)

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
        ifs = iem + 6
        isk = ifs + 1                  # touchdown-approach slacks, knots 1..N
        iao = isk + N                  # angle-of-attack slacks, knots 1..N
        igr = iao + N                  # ground slacks, knots 1..N-1
        imn = igr + N                  # no-hover (minimum sink near the ground) slacks, knots 1..N
        ikp = imn + N                  # keep-out slacks, one per half-space
        t_k = np.arange(N + 1) * dt
        keep = [kp for kp in cfg.keep_out(t_k) if kp[0] >= 1] if cfg.keep_out is not None else []
        # Elastic slacks of the limits tied to the state the booster is in
        # now (tilt cone, slew rate, the first knot pinned to the current
        # thrust): the actual vehicle can be outside them -- leaning further
        # than the budget in a gust, at full throttle in the feedback margin
        # -- and then no plan satisfies them all.  Heavily penalised, so they
        # bind exactly whenever they can, but the problem is always feasible.
        itc = ikp + len(keep)          # tilt cone, knots 0..N
        isl = itc + N + 1              # lateral slew, intervals 0..N-1
        ipn = isl + N                  # first knot: thrust magnitude, vertical thrust
        icn = ipn + 2                  # first knot sideways: command continuity, lag behind the vehicle
        n = icn + 2

        def S(k, c):
            return NX * k + c

        def U(k, c):
            return iu + 4 * k + c

        mid = (np.arange(N) + 0.5) * dt
        Gm = None
        q_mid = w_mid = dq_mid = None
        if cfg.drag_model is not None:
            res = cfg.drag_model(mid)
            if isinstance(res, tuple):
                drag, Gm = np.asarray(res[0], float).reshape(N, 3), np.asarray(res[1], float).reshape(N, 3, 3)
                if len(res) > 3:
                    q_mid, w_mid = np.asarray(res[2], float).reshape(N), np.asarray(res[3], float).reshape(N, 3)
                if len(res) > 4:
                    dq_mid = np.asarray(res[4], float).reshape(N)
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
        row([(ifs, -1.0)], 0.0)
        for k in range(N):
            row([(isk + k, -1.0)], 0.0)
        for k in range(N):
            row([(iao + k, -1.0)], 0.0)
        for k in range(N):
            row([(igr + k, -1.0)], 0.0)
        for k in range(N):
            row([(imn + k, -1.0)], 0.0)
        for i in range(itc, n):
            row([(i, -1.0)], 0.0)
        if math.isfinite(cfg.ground_alt):
            # the ground: h_k + s_k >= h_ground  (the last knot is the gate)
            for k in range(1, N):
                row([(S(k, 2), -1.0), (igr + k - 1, -1.0)], -float(cfg.ground_alt))
        # Obstacles: n . r_k + s >= b  ->  -n . r_k - s <= -b; beside a box
        # (a side face) the top of the tilted body too: top = r + L u / sigma,
        # with sigma ~ g (braking harder only tilts less for the same u, so
        # this is on the safe side)
        lever = cfg.body_length / G0
        for j, (k, nrm, b_) in enumerate(keep):
            row([(ikp + j, -1.0)], 0.0)
            row([(S(k, c), -float(nrm[c])) for c in range(3) if nrm[c] != 0.0] + [(ikp + j, -1.0)], -b_)
            if lever > 0.0 and abs(float(nrm[2])) < 0.5:
                row([(S(k, c), -float(nrm[c])) for c in range(2) if nrm[c] != 0.0]
                    + [(U(k, c), -lever * float(nrm[c])) for c in range(2) if nrm[c] != 0.0] + [(ikp + j, -1.0)], -b_)
        if cfg.min_sink > 0.0:
            # No hover just above the pad (the acceptance rule): over the last
            # metres the booster keeps sinking, vz_k <= -v_min + s_k -- a
            # sideways error must be fixed higher up, not by stopping low
            # down where the tilt budget makes it slow.
            h_k = np.asarray(cfg.height_ref(t_k), float) if cfg.height_ref is not None else None
            if h_k is not None:
                for k in range(1, N):
                    hk = float(h_k[k]) - cfg.target_alt
                    if cfg.min_sink_band[0] < hk < cfg.min_sink_band[1]:
                        row([(S(k, 5), 1.0), (imn + k - 1, -1.0)], -cfg.min_sink)
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
        # Height the final approach (sink envelope) is at t_go before the end.
        c_s = cfg.sink_gain if cfg.sink_gain > 0.0 else 0.5
        h_go = abs(cfg.touchdown_speed) / c_s * np.expm1(c_s * np.minimum(t_go, 20.0))
        if cfg.tilt_budget is not None:
            h_ref = np.asarray(cfg.height_ref(t_k), float) if cfg.height_ref is not None else h_go
            tilt = np.minimum(tilt, np.radians(np.interp(h_ref - cfg.target_alt, *cfg.tilt_budget)))
        if q_mid is not None and cfg.q_aoa is not None:
            # Same angle-of-attack cap as the tracker (knots 1..N, flow of the
            # interval before): the plan must not count on an attitude the
            # booster cannot hold against the aerodynamic moment -- e.g.
            # flying nose-first towards the pad in a fast low divert.  Kept
            # compatible with the tilt cone, so the problem stays feasible.
            # The limit depends on the dynamic pressure, which depends on the
            # height (a cross wind is weaker lower down): linearised in the
            # knot's height about the reference, with the thrust at its
            # reference ~g in that term, so that flying lower to lean into
            # the wind is visible to the optimiser.
            h_mid = np.asarray(cfg.height_ref(mid), float) if cfg.height_ref is not None else None
            for k in range(1, N + 1):
                w = w_mid[k - 1]
                if not np.any(w):
                    continue
                floor = math.acos(float(np.clip(w[2], -1.0, 1.0))) - tilt[k] + math.radians(3.0)

                def cos_lim(qq, floor=floor):
                    return math.cos(max(math.radians(float(np.interp(qq, *cfg.q_aoa))), floor))

                qk = float(q_mid[k - 1])
                c0 = cos_lim(qk)
                dc = 0.0
                if dq_mid is not None and h_mid is not None:
                    dlt = max(20.0, 0.05 * qk)
                    dc = (cos_lim(qk + dlt) - cos_lim(max(0.0, qk - dlt))) / (qk + dlt - max(0.0, qk - dlt)) * float(dq_mid[k - 1])
                if c0 <= math.cos(math.radians(170.0)) and dc == 0.0:
                    continue
                # soft (slack in m/s^2 of thrust off the allowed cone):
                # thrust-rate limits may not allow an instant turn
                terms = [(U(k, 3), c0), (iao + k - 1, -1.0)] + [(U(k, c), -float(w[c])) for c in range(3)]
                rhs = 0.0
                if dc != 0.0:
                    terms.append((S(k, 2), G0 * dc))
                    rhs = G0 * dc * float(h_mid[k - 1])
                row(terms, rhs)
        # The first knot may keep the current thrust direction (attitude lag),
        # but no more: a looser first knot is flown for the whole replan
        # period and the tracker clips it anyway.
        un = np.asarray(cfg.u_now, float)
        tilt_now = math.atan2(float(np.linalg.norm(un[:2])), max(1e-6, float(un[2]))) if cfg.engine_on else math.radians(75.0)
        tilt[0] = max(tilt[0], min(math.radians(80.0), tilt_now + math.radians(5.0)))
        cos_tilt = np.cos(tilt)
        fade = np.interp(h_go, cfg.trim_fade, (0.0, 1.0))
        for k in range(N + 1):
            # u . axis >= cos(tilt) sigma  (cone about the wind-trim axis,
            # upright at the very end like the tracker's)
            axis = np.array([float(cfg.tilt_trim[0]) * fade[k], float(cfg.tilt_trim[1]) * fade[k], G0])
            axis /= np.linalg.norm(axis)
            row([(U(k, 3), float(cos_tilt[k])), (itc + k, -1.0)] + [(U(k, c), -float(axis[c])) for c in range(3) if axis[c] != 0.0], 0.0)
        rate_s = 0.8 * 1.2 * cfg.thrust_max / m0 * dt
        lat = np.full(N, self.lateral_rate)
        if cfg.slew_deg is not None:
            h_s = np.asarray(cfg.height_ref(mid), float) if cfg.height_ref is not None else np.interp(mid, t_k, h_go)
            lat = np.minimum(lat, self.slew_margin * G0 * np.radians(np.interp(h_s - cfg.target_alt, *cfg.slew_deg)))
        lat = lat * dt
        for k in range(N):
            row([(U(k + 1, 3), 1.0), (U(k, 3), -1.0)], rate_s)
            row([(U(k, 3), 1.0), (U(k + 1, 3), -1.0)], rate_s)
        spool = 0.8 * 1.2 * cfg.thrust_max / m0 * 0.15
        row([(U(0, 3), 1.0), (ipn, -1.0)], cfg.sigma_now + spool)
        row([(U(0, 3), -1.0), (ipn, -1.0)], -(cfg.sigma_now - spool))
        if cfg.engine_on:
            # Vertical thrust is pinned near the current value (engine spool);
            # the lateral components are only pulled toward it by the cost.
            row([(U(0, 2), 1.0), (ipn + 1, -1.0)], cfg.u_now[2] + spool + 1.0)
            row([(U(0, 2), -1.0), (ipn + 1, -1.0)], -(cfg.u_now[2] - spool - 1.0))
            # Sideways, the first knot continues the command being flown
            # (re-plans must not jump it: five re-plans a second, each free to
            # start elsewhere, swung the command far faster than the slew
            # limit inside a plan -- the body lagged a second behind and the
            # booster crossed the catch axis at 4 m/s between the arms), and
            # that command may run ahead of what the vehicle is really doing
            # by about one attitude lag of slew, no more.
            slew_now = float(lat[0]) / dt
            lag_band = 0.8 * slew_now / self.slew_margin + 0.5
            for c in (0, 1):
                if cfg.u_cmd is not None:
                    row([(U(0, c), 1.0), (icn, -1.0)], float(cfg.u_cmd[c]) + 0.1)
                    row([(U(0, c), -1.0), (icn, -1.0)], -(float(cfg.u_cmd[c]) - 0.1))
                row([(U(0, c), 1.0), (icn + 1, -1.0)], float(cfg.u_now[c]) + lag_band)
                row([(U(0, c), -1.0), (icn + 1, -1.0)], -(float(cfg.u_now[c]) - lag_band))
        n_nonneg = len(row) - n_zero

        # ---- second-order cones
        soc_dims = []
        for k in range(N + 1):
            row([(U(k, 3), -1.0)], 0.0)
            row([(U(k, 0), -1.0)], 0.0)
            row([(U(k, 1), -1.0)], 0.0)
            row([(U(k, 2), -1.0)], 0.0)
            soc_dims.append(4)
        d_ref = np.asarray(cfg.dir_ref(mid), float).reshape(N, 3) if cfg.dir_ref is not None else np.tile([0.0, 0.0, 1.0], (N, 1))
        for k in range(N):
            # sideways slew: || du_xy - d_xy dsigma || <= rate dt (+ slack) --
            # the tilt rate is the size of the change (not each axis's), and
            # a change of thrust along the reference direction is no turn
            dx, dy = float(d_ref[k, 0]), float(d_ref[k, 1])
            row([(isl + k, -1.0)], float(lat[k]))
            row([(U(k + 1, 0), -1.0), (U(k, 0), 1.0), (U(k + 1, 3), dx), (U(k, 3), -dx)], 0.0)
            row([(U(k + 1, 1), -1.0), (U(k, 1), 1.0), (U(k + 1, 3), dy), (U(k, 3), -dy)], 0.0)
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

        # propellant (sum of thrust acceleration x dt), sideways effort, time
        lam = self.lateral_accel_weight
        for k in range(N + 1):
            q[U(k, 3)] += w[k]
            d_k = drag[min(k, N - 1)]
            for c_ in (0, 1):
                # lam |u_c + d_c|^2: the net sideways acceleration
                add_p(U(k, c_), U(k, c_), 2 * lam * w[k])
                q[U(k, c_)] += 2 * lam * w[k] * float(d_k[c_])
                const += lam * w[k] * float(d_k[c_]) ** 2
        const += cfg.time_weight * t_f
        tgo_k = t_f - t_k
        wr = np.interp(tgo_k, self.approach_t, self.approach_pos_w)
        wvv = np.interp(tgo_k, self.approach_t, self.approach_vel_w)
        tx, ty = float(cfg.target_xy[0]), float(cfg.target_xy[1])
        for k in range(1, N + 1):
            if wr[k] <= 0.0 and wvv[k] <= 0.0:
                continue
            for c_, t_ in ((0, tx), (1, ty)):
                add_p(S(k, c_), S(k, c_), 2 * wr[k] * w[k])
                q[S(k, c_)] += -2 * wr[k] * w[k] * t_
                const += wr[k] * w[k] * t_ * t_
                add_p(S(k, 3 + c_), S(k, 3 + c_), 2 * wvv[k] * w[k])
        jw = self.jerk_weight / dt
        for k in range(N):
            for c in range(3):
                # Lateral jerk costs more: lateral thrust means tilting the
                # whole vehicle, and fast tilt reversals are what look like rocking.
                jc = jw * (self.lateral_jerk_factor if c < 2 else 1.0)
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
        q[ifs] = self.fuel_slack_weight
        q[isk:isk + N] = self.sink_envelope_weight
        q[imn:imn + N] = self.sink_envelope_weight
        q[iao:iao + N] = self.aoa_weight * dt
        q[igr:igr + N] = self.ground_weight * dt
        q[ikp:itc] = self.keep_out_weight * dt
        q[itc:n] = self.elastic_weight
        q[icn] = 0.1 * self.elastic_weight     # continuity gives way to the vehicle's own lag
        for i in range(n):
            add_p(i, i, 1e-7)
        P = hessian.matrix()
        layout = dict(N=N, dt=dt, nS=nS, iu=iu, iep=iep, iem=iem, ifs=ifs, iao=iao, igr=igr, ikp=ikp, itc=itc, n=n,
                      NX=NX, G=Gm)
        return P, q, A, b_vec, n_zero, n_nonneg, soc_dims, layout, drag, const

    def solve(self, r0, v0, m0, t_f, cfg: PlanConfig3D) -> Plan3D | None:
        if not self.available or not (t_f > 0.3):
            return None
        P, q, A, b, nz, nn, socs, L, drag, const = self.build(r0, v0, m0, t_f, cfg)
        x, status = self.solver.solve(P, q, A, b, nz, nn, socs)
        if x is None and status not in ("no-solver",):
            # The problem is always feasible (every state-tied limit is
            # elastic), so a failure is numerical -- the iteration cap or no
            # progress on a badly scaled step: once more with more room.
            x, status = self.solver.solve(P, q, A, b, nz, nn, socs, max_iter=400)
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
            fuel_violation=float(max(0.0, x[L["ifs"]])),
            aoa_violation=float(np.max(x[L["iao"]: L["iao"] + N], initial=0.0)),
            ground_violation=float(np.max(x[L["igr"]: L["igr"] + N], initial=0.0)),
            keep_out_violation=float(np.max(x[L["ikp"]: L["itc"]], initial=0.0)),
            limit_violation=float(np.max(x[L["itc"]: L["n"] - 2], initial=0.0)),
        )

    def search(self, r0, v0, m0, lo, hi, cfg, evaluations=10):
        """Plan whose final time (searched in [lo, hi]) minimises the objective."""
        lo = max(0.6, float(lo))
        hi = max(lo + 0.5, float(hi))
        return golden_section_search(lambda t: self.solve(r0, v0, m0, t, cfg), lo, hi, evaluations)
