"""Manual flight is fly-by-wire: the stick commands rates, released it holds the attitude."""

import math

import numpy as np
import pytest

from lander3d.sim import PHYSICS_DT, Sim3D


@pytest.mark.parametrize("scenario", [0, 3, 8])
def test_released_stick_holds_the_attitude(scenario):
    sim = Sim3D(cfd=False)
    sim.load_scenario(scenario, autopilot=False)
    sim.throttle_manual = 0.6
    t, worst = 0.0, 0.0
    while t < 16.0 and sim.vehicle.state == "FLYING":
        sim.handle_input({"w": 2.0 < t < 4.0, "d": 6.0 < t < 7.0, "q": 9.0 < t < 10.0}, [])
        sim.step(PHYSICS_DT)
        t += PHYSICS_DT
        if t > 12.0:
            worst = max(worst, math.degrees(float(np.linalg.norm(sim.vehicle.omega))))
    assert sim.vehicle.state == "FLYING"
    assert worst < 8.0          # no tumbling once the keys are released


def test_stick_pitches_the_booster():
    sim = Sim3D(cfd=False)
    sim.reset_to_pad()
    sim.throttle_manual = 1.0
    for _ in range(int(4.0 / PHYSICS_DT)):
        sim.step(PHYSICS_DT)
    tilt0 = sim.vehicle.tilt_deg()
    for _ in range(int(1.0 / PHYSICS_DT)):
        sim.handle_input({"w": True}, [])
        sim.step(PHYSICS_DT)
    sim.handle_input({}, [])
    for _ in range(int(3.0 / PHYSICS_DT)):
        sim.step(PHYSICS_DT)
    assert sim.vehicle.tilt_deg() > tilt0 + 5.0
    assert math.degrees(float(np.linalg.norm(sim.vehicle.omega[:2]))) < 2.0
