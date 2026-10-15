"""The conic-program plumbing shared by both planners."""

import numpy as np
import pytest
from scipy import sparse

from common.conic import ConicSolver, ConstraintRows, UpperQuadratic, golden_section_search, objective_value


def test_constraint_rows_build_matrix_in_insertion_order():
    rows = ConstraintRows()
    rows([(0, 1.0), (2, -2.0)], 3.0)
    rows([(1, 4.0)], 5.0)
    assert len(rows) == 2
    assert rows.matrix(3).toarray().tolist() == [[1.0, 0.0, -2.0], [0.0, 4.0, 0.0]]
    assert rows.rhs.tolist() == [3.0, 5.0]


def test_upper_quadratic_keeps_upper_triangle_and_sums_duplicates():
    form = UpperQuadratic(2)
    form.add(1, 0, 1.0)
    form.add(0, 1, 2.0)
    form.add(1, 1, 3.0)
    assert form.matrix().toarray().tolist() == [[0.0, 3.0], [0.0, 3.0]]


def test_objective_value_uses_symmetric_hessian():
    form = UpperQuadratic(2)
    form.add(0, 0, 2.0)
    form.add(0, 1, 1.0)
    P = form.matrix()
    x = np.array([1.0, 2.0])
    q = np.array([1.0, 0.0])
    # 0.5 * x'[[2,1],[1,0]]x + q'x + 4 = 0.5 * (2 + 4) + 1 + 4
    assert objective_value(P, q, x, 4.0) == pytest.approx(8.0)


@pytest.fixture(scope="module")
def solver():
    solver = ConicSolver()
    if not solver.available:
        pytest.skip("no Clarabel back-end installed")
    return solver


def test_linear_program(solver):
    # min x  s.t.  x >= 1
    x, status = solver.solve(sparse.csc_matrix((1, 1)), np.array([1.0]), sparse.csc_matrix([[-1.0]]),
                             np.array([-1.0]), 0, 1, [])
    assert status in {"Solved", "AlmostSolved"}
    assert x[0] == pytest.approx(1.0, abs=1e-5)


def test_second_order_cone(solver):
    # min t  s.t.  ||(3, 4)|| <= t
    A = sparse.csc_matrix([[-1.0], [0.0], [0.0]])
    x, status = solver.solve(sparse.csc_matrix((1, 1)), np.array([1.0]), A, np.array([0.0, 3.0, 4.0]), 0, 0, [3])
    assert status in {"Solved", "AlmostSolved"}
    assert x[0] == pytest.approx(5.0, abs=1e-5)


def test_golden_section_finds_minimum_and_skips_infeasible():
    class Plan:
        def __init__(self, t):
            self.t, self.objective = t, (t - 7.0) ** 2

    best = golden_section_search(lambda t: Plan(t) if t > 2.0 else None, 1.0, 20.0, 14)
    assert best.t == pytest.approx(7.0, abs=0.1)
    assert golden_section_search(lambda t: None, 1.0, 2.0, 6) is None
