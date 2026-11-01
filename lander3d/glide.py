"""Convex glide planner: the engine-off fall from now to the landing-burn
ignition, steered by aerodynamic lift so that the landing burn needs no
sideways flight.

Second-order-cone program over N intervals (trapezoidal integration):

    v_k+1 = v_k + dt (d_k + J_k ((v_k + v_k+1)/2 - v_ref,k) + g + a_k)
    r_k+1 = r_k + dt (v_k + v_k+1) / 2

    a_k   glide lift acceleration (the control): across the flow
          (w_k . a_k = 0) and at most what the body and grid fins really give
          at the largest glide angle of attack (||a_k|| <= L_k)
    d_k   drag along the reference (engines-first, aligned with the flow),
          J_k = -(D_k/|v|)(I + w w^T) its derivative in the velocity -- the
          plan may fly faster or slower than its reference and sees the drag
          change (successive convexification about the previous glide)

    miss  = c + A (x_N - x_ref,N): where the landing burn lit at the end of
          the glide would end, linearised about the reference ignition state

    min   w_miss ||miss||_1 + w_lift sum ||a_k||^2 dt + w_rate sum ||a_k+1 - a_k||^2 / dt

The miss is an exact L1 penalty, so the problem is always feasible; what is
left of it says the glide cannot do it alone (the boost-back must do more).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.conic import ConicSolver, ConstraintRows, UpperQuadratic

from .vehicle import G0


@dataclass
class GlidePlan:
    t_go: float                 # s to the landing-burn ignition
    dt: float
    r: np.ndarray               # (N+1, 3)
    v: np.ndarray               # (N+1, 3)
    lift: np.ndarray            # (N, 3) lift acceleration per interval
    lift_max: np.ndarray        # (N,)
    miss: np.ndarray            # (2,) predicted landing-burn miss (linearised)
    status: str
    created_at: float = 0.0

    @property
    def usage(self) -> float:
        """Largest share of the available lift the plan uses."""
        ok = self.lift_max > 1e-6
        if not np.any(ok):
            return 0.0
        return float(np.max(np.linalg.norm(self.lift[ok], axis=1) / self.lift_max[ok]))

    def lift_at(self, tau: float) -> np.ndarray:
        k = int(np.clip(tau / self.dt, 0, len(self.lift) - 1))
        return self.lift[k].copy()


class GlidePlanner:
    knots = 25
    miss_weight = 50.0          # per m of landing-burn miss (L1)
    lift_weight = 1.0           # per (m/s^2)^2 s of lift
    lift_rate_weight = 2.0      # per (m/s^2)^2 / s of its change

    def __init__(self) -> None:
        self.solver = ConicSolver()
        self.last_status = "idle"

    @property
    def available(self) -> bool:
        return self.solver.available

    def solve(self, r0, v0, t_go, ref) -> GlidePlan | None:
        """``ref``: the reference along the glide, per interval k (mid
        times): ``drag`` (N,3), ``jac`` (N,3,3), ``v_ref`` (N,3), ``flow``
        (N,3) unit flow direction (engines-first axis), ``lift_max`` (N,);
        and at the end ``x_end`` (6,) the reference ignition state, ``miss``
        (2,) its landing-burn miss and ``miss_jac`` (2,6) its derivative."""
        if not self.available or not t_go > 0.5:
            return None
        N = int(ref["drag"].shape[0])
        dt = t_go / N
        nX = 6 * (N + 1)
        ia = nX                      # lift, 3 per interval
        isl = ia + 3 * N             # L1 slacks of the miss (2)
        n = isl + 2

        def X(k, c):
            return 6 * k + c

        def A(k, c):
            return ia + 3 * k + c

        row = ConstraintRows()
        # ---- zero cone: start, dynamics, lift across the flow
        for c in range(3):
            row([(X(0, c), 1.0)], float(r0[c]))
            row([(X(0, 3 + c), 1.0)], float(v0[c]))
        g = np.array([0.0, 0.0, -G0])
        for k in range(N):
            J = ref["jac"][k]
            for c in range(3):
                # v_k+1 - v_k - dt a_k - dt/2 J (v_k + v_k+1) = dt (d + g - J v_ref)
                terms = [(X(k + 1, 3 + c), 1.0), (X(k, 3 + c), -1.0), (A(k, c), -dt)]
                for j in range(3):
                    if J[c, j] != 0.0:
                        terms += [(X(k, 3 + j), -0.5 * dt * J[c, j]), (X(k + 1, 3 + j), -0.5 * dt * J[c, j])]
                row(terms, dt * float(ref["drag"][k][c] + g[c] - J[c] @ ref["v_ref"][k]))
                row([(X(k + 1, c), 1.0), (X(k, c), -1.0), (X(k, 3 + c), -0.5 * dt), (X(k + 1, 3 + c), -0.5 * dt)], 0.0)
            w = ref["flow"][k]
            if ref["lift_max"][k] > 1e-6 and np.any(w):
                row([(A(k, c), float(w[c])) for c in range(3) if w[c] != 0.0], 0.0)
            else:
                for c in range(3):
                    row([(A(k, c), 1.0)], 0.0)
        n_zero = len(row)
        # ---- nonnegative: |miss_c| <= s_c, miss = c + M (x_N - x_end)
        M, c0, xe = ref["miss_jac"], ref["miss"], ref["x_end"]
        for i in range(2):
            base = float(c0[i] - M[i] @ xe)
            terms = [(X(N, j), float(M[i, j])) for j in range(6) if M[i, j] != 0.0]
            row(terms + [(isl + i, -1.0)], -base)
            row([(col, -val) for col, val in terms] + [(isl + i, -1.0)], base)
        n_nonneg = len(row) - n_zero
        # ---- second-order cones: ||a_k|| <= L_k
        socs = []
        for k in range(N):
            if ref["lift_max"][k] > 1e-6:
                row([], float(ref["lift_max"][k]))
                for c in range(3):
                    row([(A(k, c), -1.0)], 0.0)
                socs.append(4)
        Amat = row.matrix(n)
        b = row.rhs
        # ---- objective
        H = UpperQuadratic(n)
        q = np.zeros(n)
        q[isl:isl + 2] = self.miss_weight
        for k in range(N):
            for c in range(3):
                H.add(A(k, c), A(k, c), 2.0 * self.lift_weight * dt)
                if k + 1 < N:
                    jw = self.lift_rate_weight / dt
                    H.add(A(k, c), A(k, c), 2.0 * jw)
                    H.add(A(k + 1, c), A(k + 1, c), 2.0 * jw)
                    H.add(A(k, c), A(k + 1, c), -2.0 * jw)
        for i in range(n):
            H.add(i, i, 1e-7)
        P = H.matrix()
        x, status = self.solver.solve(P, q, Amat, b, n_zero, n_nonneg, socs)
        if x is None:
            x, status = self.solver.solve(P, q, Amat, b, n_zero, n_nonneg, socs, max_iter=400)
        self.last_status = status
        if x is None:
            return None
        Xs = x[:nX].reshape(N + 1, 6)
        lift = x[ia:ia + 3 * N].reshape(N, 3)
        miss = c0 + M @ (Xs[-1] - xe)
        return GlidePlan(t_go=float(t_go), dt=dt, r=Xs[:, 0:3].copy(), v=Xs[:, 3:6].copy(), lift=lift.copy(),
                         lift_max=np.asarray(ref["lift_max"], float).copy(), miss=np.asarray(miss, float), status=status)
