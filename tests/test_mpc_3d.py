"""Model-predictive tracker of the landing burn (lander3d/mpc.py)."""

import numpy as np
import pytest

from lander3d.mpc import TrackingMPC, discretise

KW = dict(N=12, dt=0.2, taus=(0.6, 0.6, 0.2), q_pos=1.0, q_vel=0.5, terminal_weight=4.0, r_u=1.0, r_rate=1.0)


def test_discretisation_matches_fine_integration():
    Ad, Bd = discretise(0.6, 0.2)
    x, h = np.array([1.0, -0.5, 0.3]), 1e-5
    y = x.copy()
    for _ in range(int(0.2 / h)):
        y = y + h * np.array([y[1], y[2], (2.0 - y[2]) / 0.6])
    assert np.allclose(Ad @ x + Bd * 2.0, y, atol=1e-4)


def _closed_loop(mpc, tau_true, kp=None, kd=None, t_end=15.0):
    """1-D error with a lagging thrust, 10 Hz updates, 240 Hz integration."""
    e, v, a, du, nxt, h = 10.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 240.0
    out = []
    for i in range(int(t_end / h)):
        if i * h >= nxt:
            if mpc is None:
                du = float(np.clip(-kp * e - kd * v, -5.0, 5.0))
            else:
                seq = mpc.solve(np.array([e, 0, 0.0]), np.array([v, 0, 0.0]), np.array([a, 0, 0.0]),
                                np.zeros((12, 3)), np.array([du, 0, 0.0]), 5.0, **KW)
                du = float(seq[0, 0])
            nxt = i * h + 0.1
        a += (du - a) / tau_true * h
        v += a * h
        e += v * h
        out.append(e)
    return np.array(out)


@pytest.mark.parametrize("tau_true", [0.4, 0.6, 1.0])
def test_lagging_thrust_does_not_overshoot(tau_true):
    mpc = TrackingMPC()
    if not mpc.available:
        pytest.skip("no conic solver")
    e = _closed_loop(mpc, tau_true)
    assert e.min() > -0.5                  # PD 0.5/0.1 swings to -15 m here, 0.5/0.5 to -6 m
    assert abs(e[-1]) < 0.3


def test_correction_respects_the_limit_and_the_lead():
    mpc = TrackingMPC()
    if not mpc.available:
        pytest.skip("no conic solver")
    seq = mpc.solve(np.array([200.0, -300.0, 50.0]), np.zeros(3), np.zeros(3), np.zeros((12, 3)), np.zeros(3), 5.0, **KW)
    assert np.all(np.linalg.norm(seq, axis=1) <= 5.0 + 1e-4)
    # on the plan, the plan's thrust turning: the tracker leads it (the thrust lags)
    u_dot = np.tile([1.0, 0.0, 0.0], (12, 1))
    seq = mpc.solve(np.zeros(3), np.zeros(3), np.zeros(3), u_dot, np.zeros(3), 5.0, **KW)
    assert seq[0, 0] > 0.2
