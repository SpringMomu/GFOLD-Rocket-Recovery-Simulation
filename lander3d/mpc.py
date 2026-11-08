"""Model-predictive tracker of the G-FOLD plan (landing burn).

G-FOLD plans the whole burn (``Planner3D``: minutes of flight, ~40 nodes,
re-solved every 0.2-0.6 s); this tracker closes the loop round it at 10 Hz
over the next couple of seconds.  It replaces the PD correction
``kp (p_ref - r) + kd (v_ref - v)`` added to the plan's thrust.

Why not PD: the correction does not act on the booster directly -- it turns
the thrust, and the thrust turns with the attitude loop (gimbal, fins, the
slew-rate limit), about half a second behind the command.  A PD loop does not
know about that lag: it keeps pushing while the thrust is still swinging
round, overshoots, and the attitude follows the oscillation.  The tracker
here predicts it.

Model (per axis, independent; the error is relative to the plan)::

    e'  = ev                       position error  r - p_ref
    ev' = a                        velocity error  v - v_ref
    a'  = (du + w - a) / tau       thrust acceleration minus the plan's, lagging
                                   the command by tau (sideways: the attitude
                                   loop; vertical: the throttle)

``w = -tau * d u_ff/dt`` is the plan's own thrust change (the thrust lags the
feed-forward too, so the tracker leads it).  ``a`` starts at the correction
in force (Autopilot3D.mpc_a0_mode 0): no measurement of the thrust goes in --
the measured thrust or body axis carries the attitude loop's own sway, and a
tracker that answers it drives it (the rocking this replaces).  Exact
zero-order-hold discretisation with step ``dt``, horizon ``N`` steps.

Problem (a small second-order-cone program, same Clarabel back end as the
planner, ``common/conic.py``)::

    min  sum_k  q_pos |e_k|^2 + q_vel |ev_k|^2         (x terminal_weight at k = N)
              + r_u |du_k|^2 + r_rate |du_k - du_{k-1}|^2
    s.t. the dynamics above, the measured initial state,
         |du_k| <= limit                               (same cap as the PD correction)

The structure (and so the matrices P, A) depends only on the step, the
horizon, the lags and the weights; a solve only changes the vectors q, b.
The first move ``du_0`` is the correction for the next ``period``.
"""

from __future__ import annotations

import math

import numpy as np

from common.conic import ConicSolver, ConstraintRows, UpperQuadratic

NS = 9          # state per step: e (3), ev (3), a (3)


def _expm(M: np.ndarray) -> np.ndarray:
    """Matrix exponential of a small matrix (scaling and squaring + Taylor)."""
    norm = float(np.max(np.sum(np.abs(M), axis=1)))
    s = max(0, math.ceil(math.log2(norm)) + 1) if norm > 0.5 else 0
    A = M / (2.0 ** s)
    out = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, 18):
        term = term @ A / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def discretise(tau: float, dt: float):
    """Exact ZOH discretisation of the chain e' = ev, ev' = a, a' = (u - a)/tau."""
    tau = max(float(tau), 1e-3)
    M = np.zeros((4, 4))
    M[0, 1] = 1.0
    M[1, 2] = 1.0
    M[2, 2] = -1.0 / tau
    M[2, 3] = 1.0 / tau
    E = _expm(M * dt)
    return E[:3, :3], E[:3, 3]


class TrackingMPC:
    """Tracks the plan's position/velocity by a thrust-acceleration correction."""

    def __init__(self, solver: ConicSolver | None = None) -> None:
        self.solver = solver or ConicSolver()
        self._key = None
        self._P = None
        self._A = None
        self._b_template = None
        self.last_status = ""
        self.solves = 0
        self.failures = 0

    @property
    def available(self) -> bool:
        return self.solver.available

    # ----------------------------------------------------------- structure
    @staticmethod
    def _s(k: int, c: int) -> int:
        return NS * k + c

    def _u(self, N: int, k: int, i: int) -> int:
        return NS * (N + 1) + 3 * k + i

    def _build(self, N, dt, taus, q_pos, q_vel, terminal_weight, r_u, r_rate):
        n = NS * (N + 1) + 3 * N
        disc = [discretise(t, dt) for t in taus]
        rows = ConstraintRows()
        # initial state (9 rows; rhs filled in per solve)
        for c in range(NS):
            rows([(self._s(0, c), 1.0)], 0.0)
        # dynamics: s_{k+1} - Ad s_k - Bd du_k = Bd w_k
        for k in range(N):
            for i in range(3):
                Ad, Bd = disc[i]
                for j in range(3):
                    entries = [(self._s(k + 1, 3 * j + i), 1.0)]
                    for col in range(3):
                        if Ad[j, col] != 0.0:
                            entries.append((self._s(k, 3 * col + i), -float(Ad[j, col])))
                    entries.append((self._u(N, k, i), -float(Bd[j])))
                    rows(entries, 0.0)
        n_zero = len(rows)
        # |du_k| <= limit
        for k in range(N):
            rows([], 0.0)                        # limit (rhs per solve)
            for i in range(3):
                rows([(self._u(N, k, i), -1.0)], 0.0)
        self._A = rows.matrix(n)
        self._n_zero = n_zero
        self._soc = [4] * N
        self._disc = disc

        Q = UpperQuadratic(n)
        for k in range(1, N + 1):
            f = terminal_weight if k == N else 1.0
            for i in range(3):
                Q.add(self._s(k, i), self._s(k, i), 2.0 * q_pos * f)
                Q.add(self._s(k, 3 + i), self._s(k, 3 + i), 2.0 * q_vel * f)
        for k in range(N):
            for i in range(3):
                u = self._u(N, k, i)
                Q.add(u, u, 2.0 * r_u)
                Q.add(u, u, 2.0 * r_rate)                       # (du_k - du_{k-1})^2, k = 0 against the applied one
                if k + 1 < N:
                    un = self._u(N, k + 1, i)
                    Q.add(un, un, 2.0 * r_rate)
                    Q.add(u, un, -2.0 * r_rate)
        self._P = Q.matrix()
        self._n = n

    # --------------------------------------------------------------- solve
    def solve(self, e0, ev0, a0, u_dot, u_prev, limit, *, N, dt, taus, q_pos, q_vel,
              terminal_weight, r_u, r_rate, max_iter=60):
        """The correction sequence ``(N, 3)`` (m/s^2), or None if nothing solves.

        e0, ev0, a0 : initial position / velocity error (actual - plan) and
                      thrust-acceleration offset (actual - plan's)
        u_dot       : (N, 3) the plan's thrust-acceleration rate over the horizon
        u_prev      : the correction applied now (the rate cost's anchor)
        """
        key = (int(N), round(float(dt), 6), tuple(round(float(t), 6) for t in taus), float(q_pos), float(q_vel),
               float(terminal_weight), float(r_u), float(r_rate))
        if key != self._key:
            self._build(int(N), float(dt), taus, q_pos, q_vel, terminal_weight, r_u, r_rate)
            self._key = key
        N = int(N)
        b = np.zeros(self._A.shape[0])
        b[0:3] = e0
        b[3:6] = ev0
        b[6:9] = a0
        row = NS
        for k in range(N):
            for i in range(3):
                _, Bd = self._disc[i]
                w = -float(taus[i]) * float(u_dot[k, i])
                for j in range(3):
                    b[row] = Bd[j] * w
                    row += 1
        for k in range(N):
            b[self._n_zero + 4 * k] = float(limit)
        q = np.zeros(self._n)
        for i in range(3):
            q[self._u(N, 0, i)] = -2.0 * r_rate * float(u_prev[i])
        x, status = self.solver.solve(self._P, q, self._A, b, self._n_zero, 0, self._soc, max_iter)
        self.last_status = status
        self.solves += 1
        if x is None:
            self.failures += 1
            return None
        u = x[NS * (N + 1):].reshape(N, 3)
        return u
