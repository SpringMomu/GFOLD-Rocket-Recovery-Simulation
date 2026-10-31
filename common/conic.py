"""Conic-program plumbing shared by the 2D and 3D guidance.

The planners assemble ``min 1/2 x'Px + q'x  s.t.  Ax + s = b`` with
``s`` in {0} x R+ x SOC(d1) x ... and hand it to :class:`ConicSolver`.  The
solver uses the Rust DLL (``native/``, ABI 2) when it loads and falls back to
the Python ``clarabel`` package, which gets exactly the same matrices.
"""

from __future__ import annotations

import ctypes
import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

try:
    import clarabel
    from scipy import sparse
except ImportError:  # minimal install: the guidance falls back to plain feedback
    clarabel = None
    sparse = None

NATIVE_ABI_VERSION = 2
NATIVE_BACKEND_PREFIX = "clarabel-rust-conic-"
NATIVE_DLL = Path(__file__).resolve().parents[1] / "clarabel_hover.dll"
MAX_ITERATIONS = 100


class _NativeDiagnostics(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("iterations", ctypes.c_uint32),
        ("objective", ctypes.c_double),
        ("solve_time_seconds", ctypes.c_double),
    ]


class NativeClarabel:
    """ctypes adapter for the Rust DLL (generic conic entry point, ABI 2)."""

    _double_p = ctypes.POINTER(ctypes.c_double)
    _size_p = ctypes.POINTER(ctypes.c_size_t)
    _status_names = {1: "Solved", 4: "AlmostSolved"}

    def __init__(self, path: Path) -> None:
        library = ctypes.CDLL(str(path))
        library.clarabel_hover_abi_version.argtypes = []
        library.clarabel_hover_abi_version.restype = ctypes.c_uint32
        version = library.clarabel_hover_abi_version()
        if version != NATIVE_ABI_VERSION:
            raise RuntimeError(
                f"native solver ABI {version} is outdated (need {NATIVE_ABI_VERSION}); "
                "run native\\build_solver.ps1"
            )
        library.clarabel_hover_backend_name.argtypes = []
        library.clarabel_hover_backend_name.restype = ctypes.c_char_p
        d, s = self._double_p, self._size_p
        library.clarabel_conic_solve.argtypes = [
            ctypes.c_size_t, ctypes.c_size_t,       # n, m
            s, s, d, ctypes.c_size_t,               # P colptr/rowval/nzval/nnz
            d,                                      # q
            s, s, d, ctypes.c_size_t,               # A colptr/rowval/nzval/nnz
            d,                                      # b
            ctypes.c_size_t, ctypes.c_size_t,       # zero rows, nonneg rows
            s, ctypes.c_size_t,                     # soc dims, soc count
            ctypes.c_uint32,                        # max iterations
            d, ctypes.c_size_t,                     # x_out, len
            ctypes.POINTER(_NativeDiagnostics),
        ]
        library.clarabel_conic_solve.restype = ctypes.c_int32
        name = library.clarabel_hover_backend_name()
        if not name:
            raise RuntimeError("native backend has no identity")
        self.library = library
        self.name = name.decode("ascii", errors="replace")

    def solve(self, P, q, A, b, n_zero, n_nonneg, soc_dims, max_iter):
        def csc_parts(M):
            M = M.tocsc()
            M.sort_indices()
            return (
                np.ascontiguousarray(M.indptr, dtype=np.uintp),
                np.ascontiguousarray(M.indices, dtype=np.uintp),
                np.ascontiguousarray(M.data, dtype=np.float64),
            )

        pc, pr, pv = csc_parts(P)
        ac, ar, av = csc_parts(A)
        q = np.ascontiguousarray(q, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        socs = np.ascontiguousarray(soc_dims, dtype=np.uintp)
        x = np.empty(q.size, dtype=np.float64)
        diagnostics = _NativeDiagnostics()

        def dp(a):
            return a.ctypes.data_as(self._double_p)

        def sp(a):
            return a.ctypes.data_as(self._size_p)

        code = self.library.clarabel_conic_solve(
            q.size, b.size,
            sp(pc), sp(pr), dp(pv), pv.size,
            dp(q),
            sp(ac), sp(ar), dp(av), av.size,
            dp(b),
            n_zero, n_nonneg,
            sp(socs), socs.size,
            max_iter,
            dp(x), x.size,
            ctypes.byref(diagnostics),
        )
        status = self._status_names.get(code)
        if status is None:
            return None, f"native:{code}"
        if not np.all(np.isfinite(x)):
            return None, "native:nonfinite"
        return x, status


class ConicSolver:
    """Picks the native DLL, the Python ``clarabel`` package, or neither."""

    def __init__(self, dll: Path = NATIVE_DLL) -> None:
        self.native_error = ""
        self._native: NativeClarabel | None = None
        enabled = os.environ.get("ROCKET_NATIVE_SOLVER", "1").lower() not in {"0", "false", "no"}
        if enabled and sparse is not None:
            try:
                self._native = NativeClarabel(dll)
            except (AttributeError, OSError, RuntimeError) as error:
                self.native_error = str(error)
        if self._native is not None:
            self.backend = self._native.name
        elif clarabel is not None and sparse is not None:
            self.backend = "clarabel-python"
        else:
            self.backend = "feedback-fallback"

    @property
    def available(self) -> bool:
        return self.backend != "feedback-fallback"

    def solve(self, P, q, A, b, n_zero, n_nonneg, soc_dims, max_iter=MAX_ITERATIONS):
        """Return ``(x, status)``; ``x`` is None unless the status is acceptable."""
        if self._native is not None:
            try:
                return self._native.solve(P, q, A, b, n_zero, n_nonneg, soc_dims, max_iter)
            except (OSError, RuntimeError, ValueError) as error:
                self.native_error = str(error)
                self._native = None
                self.backend = "clarabel-python" if clarabel is not None else "feedback-fallback"
        if clarabel is None:
            return None, "no-solver"
        cones = [clarabel.ZeroConeT(n_zero), clarabel.NonnegativeConeT(n_nonneg)]
        cones.extend(clarabel.SecondOrderConeT(d) for d in soc_dims)
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = max_iter
        try:
            result = clarabel.DefaultSolver(P, q, A, b, cones, settings).solve()
        except Exception as error:  # a solver failure must not stop the control loop
            return None, f"python:{type(error).__name__}"
        status = str(result.status)
        if status not in {"Solved", "AlmostSolved"} or result.x is None:
            return None, status
        x = np.asarray(result.x, dtype=float)
        if not np.all(np.isfinite(x)):
            return None, "nonfinite"
        return x, status


class ConstraintRows:
    """Collects ``A x (<=|=) b`` rows in the order they are added."""

    def __init__(self) -> None:
        self._rows: list[int] = []
        self._cols: list[int] = []
        self._vals: list[float] = []
        self._rhs: list[float] = []

    def __len__(self) -> int:
        return len(self._rhs)

    def __call__(self, entries, rhs: float) -> None:
        index = len(self._rhs)
        for col, val in entries:
            self._rows.append(index)
            self._cols.append(col)
            self._vals.append(val)
        self._rhs.append(float(rhs))

    @property
    def rhs(self) -> np.ndarray:
        return np.asarray(self._rhs, dtype=float)

    def matrix(self, n_cols: int):
        return sparse.csc_matrix((self._vals, (self._rows, self._cols)), shape=(len(self._rhs), n_cols))


class UpperQuadratic:
    """Upper triangle of the objective Hessian ``P`` (duplicates are summed)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._rows: list[int] = []
        self._cols: list[int] = []
        self._vals: list[float] = []

    def add(self, i: int, j: int, value: float) -> None:
        if i > j:
            i, j = j, i
        self._rows.append(i)
        self._cols.append(j)
        self._vals.append(value)

    def matrix(self):
        P = sparse.csc_matrix((self._vals, (self._rows, self._cols)), shape=(self.n, self.n))
        P.sum_duplicates()
        return P


def objective_value(P, q, x, constant: float) -> float:
    """Objective for the upper-triangular ``P`` representation."""
    full = P + P.T - sparse.diags(P.diagonal())
    return float(0.5 * x @ (full @ x) + q @ x + constant)


def golden_section_search(solve: Callable[[float], object | None], lo: float, hi: float, evaluations: int):
    """Plan with the lowest ``objective`` among the final times probed in [lo, hi]."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    best = None

    def value(t: float) -> float:
        nonlocal best
        plan = solve(t)
        if plan is None:
            return math.inf
        if best is None or plan.objective < best.objective:
            best = plan
        return plan.objective

    a, b = lo, hi
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    fc, fd = value(c), value(d)
    for _ in range(max(0, evaluations - 2)):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = value(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = value(d)
    return best
