"""Fast planning paths: the vectorised code must agree with the references.

The 3D planner evaluates its aerodynamic model at every knot (and twice more
for the lift gradient).  Doing that per node in Python was most of the matrix
assembly time, so the knots are now evaluated in one batch:
``AeroModel.quick_force_many``, ``Plan3D.sample_many`` and the vectorised wind
profile.  These tests pin them to the scalar implementations that remain in
the code (``quick_force``, ``Plan3D.sample``, ``_drag_model_scalar``).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from lander3d.aero import AeroModel
from lander3d.planner import Plan3D
from lander3d.sim import Sim3D
from lander3d.vehicle import VehicleSpec


def _locked_sim(scenario: int, max_steps: int = 9000) -> Sim3D:
    """A simulation that has just lit its landing burn."""
    sim = Sim3D(cfd=False, scenario=scenario, wind_on=True)
    for _ in range(max_steps):
        sim.step(1.0 / 240.0)
        if sim.autopilot.burn_locked:
            break
    assert sim.autopilot.burn_locked, "the scenario never ignited"
    return sim


def test_quick_force_many_matches_quick_force():
    aero = AeroModel(VehicleSpec())
    rng = np.random.default_rng(1)
    n = 24
    vr = rng.normal(size=(n, 3)) * 60.0
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    rho = rng.uniform(0.02, 1.2, n)
    a_snd = np.full(n, 320.0)
    ref = np.array([aero.quick_force(vr[i], axis[i], rho[i], a_snd[i], 1.0) for i in range(n)])
    got = aero.quick_force_many(vr, axis, rho, a_snd, 1.0)
    assert np.allclose(got, ref, rtol=1e-11, atol=1e-9)


def test_sample_many_matches_sample():
    rng = np.random.default_rng(2)
    N = 12
    plan = Plan3D(t_f=6.0, dt=0.5, r=rng.normal(size=(N + 1, 3)), v=rng.normal(size=(N + 1, 3)),
                  z=np.log(5000.0) * np.ones(N + 1), u=rng.normal(size=(N + 1, 3)) * 0.5,
                  sigma=np.ones(N + 1), drag=rng.normal(size=(N + 1, 3)) * 0.1, status="Solved",
                  objective=1.0, terminal_position_error=0.0, terminal_velocity_error=0.0, fuel_violation=0.0)
    taus = np.array([-1.0, 0.0, 0.2, 0.6, 3.2, 5.9, 6.0, 7.5])
    pos, vel, acc = plan.sample_many(taus)
    for i, tau in enumerate(taus):
        p, v, a = plan.sample(float(tau))
        assert np.allclose(pos[i], p) and np.allclose(vel[i], v) and np.allclose(acc[i], a)


@pytest.mark.parametrize("scenario", [0, 3])
def test_drag_model_matches_scalar(scenario):
    sim = _locked_sim(scenario)
    ap = sim.autopilot
    veh = sim.vehicle
    wind = sim.wind.at(veh.pos, veh.time)
    t_f = max(2.0, (ap.touchdown_time or 0.0) - ap.elapsed)
    mid = (np.arange(40) + 0.5) * (t_f / 40)
    fast = ap._drag_model(veh, wind, t_f)(mid)
    ref = ap._drag_model_scalar(veh, wind, t_f)(mid)
    for a, b in zip(fast, ref):
        a, b = np.asarray(a, float), np.asarray(b, float)
        assert np.allclose(a, b, rtol=1e-10, atol=1e-10)


def test_async_planning_commits_a_plan():
    sim = Sim3D(cfd=False, scenario=0, wind_on=True, async_plan=True)
    ap = sim.autopilot
    try:
        assert ap.async_plan, "the planning thread did not start on this platform"
        last = time.perf_counter()
        end = last + 6.0
        while time.perf_counter() < end and ap.plan is None:
            now = time.perf_counter()
            sim.advance_realtime(now - last)
            last = time.perf_counter()
            time.sleep(0.002)
        assert ap.plan is not None, "no plan was committed from the planning thread"
        assert np.all(np.isfinite(ap.plan.r)) and ap.plan.sane
        assert abs(ap.touchdown_time - (ap.plan.created_at + ap.plan.t_f)) < 1e-6
    finally:
        ap.close_planning()


def test_planning_thread_never_blocks_the_step():
    """After ignition, the steps that carry the 5 Hz re-plans stay short: the
    solves run on the planning thread, not on the simulation thread."""
    sim = Sim3D(cfd=False, scenario=0, wind_on=True, async_plan=True)
    try:
        last = time.perf_counter()
        while not sim.autopilot.burn_locked and time.perf_counter() - last < 10.0:
            now = time.perf_counter()
            sim.advance_realtime(now - last)
            last = time.perf_counter()
            time.sleep(0.002)
        assert sim.autopilot.burn_locked
        worst = 0.0
        end = time.perf_counter() + 2.0
        while time.perf_counter() < end:
            now = time.perf_counter()
            sim.advance_realtime(now - last)
            last = time.perf_counter()
            worst = max(worst, last - now)
            time.sleep(0.002)
        assert worst < 0.08, f"a single step blocked for {worst * 1e3:.0f} ms"
        assert sim.autopilot.planner.solve_count > 0
    finally:
        sim.autopilot.close_planning()
