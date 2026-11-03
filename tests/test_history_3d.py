"""Flight records and trajectory attitude data use the simulated state."""

import numpy as np

from lander3d.server import snapshot
from lander3d.sim import PHYSICS_DT, Sim3D
from lander3d.vehicle import Vehicle


def test_landed_fin_servos_return_to_neutral_without_folding():
    v = Vehicle()
    v.state = "LANDED"
    v.fin_defl = np.array([0.3, -0.2, 0.1, -0.1])
    v.fins_deploy = 1.0
    v.step(0.1)
    assert np.max(np.abs(v.fin_defl)) < 0.3
    assert v.fins_deploy == 1.0
    v.step(1.0)
    assert np.allclose(v.fin_defl, 0.0)


def test_history_and_actual_attitude_are_recorded_and_reset():
    sim = Sim3D(cfd=False, scenario=0)
    for _ in range(150):
        sim.step(PHYSICS_DT)
    st = snapshot(sim)
    assert len(st["history"]) >= 2
    assert len(st["trail"]) == len(st["trail_axes"])
    assert np.isclose(np.linalg.norm(st["trail_axes"][0]), 1.0, atol=1e-3)
    assert st["history"][0][0] == 0.0
    sim.load_scenario(2)
    assert len(sim.history) == 1 and sim.history[0][0] == 0.0
    assert len(sim.trail) == len(sim.trail_axes) == 1


def test_published_solver_axes_stay_aligned_when_the_snapshot_is_decimated():
    class Plan:
        created_at = 0.0
        t_f = 30.0

        def sample_many(self, times):
            points = np.column_stack((times, np.zeros_like(times), times + 1.0))
            thrust = np.tile([3.0, 0.0, 4.0], (len(times), 1))
            return points, points, thrust

    sim = Sim3D(cfd=False, scenario=0)
    sim.autopilot.plan = Plan()
    sim.autopilot.burn_locked = True
    sim.autopilot._publish()
    state = snapshot(sim)
    assert len(state["plan"]) == len(state["plan_axes"]) == 80
    assert np.allclose(state["plan_axes"][0], [0.6, 0.0, 0.8])
