import numpy as np

from lander3d.server import snapshot
from lander3d.sim import Sim3D


def test_full_trajectory_survives_step_and_snapshot(monkeypatch):
    sim = Sim3D(cfd=False, wind_on=False)
    sim.autopilot_on = False
    sim.trail = [np.array([float(i * 5), 0.0, 100.0]) for i in range(1601)]
    sim.vehicle.pos = np.array([8005.0, 0.0, 100.0])
    monkeypatch.setattr(sim.vehicle, "step", lambda *args: None)

    sim.step(0.01)
    trail = snapshot(sim)["trail"]
    assert len(trail) == 1602
    assert trail[0] == [0.0, 0.0, 100.0]
    assert trail[-1] == [8005.0, 0.0, 100.0]
