"""Unpowered guidance uses the available aerodynamic force without a fixed attitude cap."""

import math

import numpy as np

from lander3d.sim import Sim3D


def test_glide_can_command_beyond_the_old_limit_and_steers_towards_the_target():
    sim = Sim3D(cfd=False)
    ap, veh = sim.autopilot, sim.vehicle
    veh.reset(position=(0, 0, 1500), velocity=(0, 0, -100), on_pad=False)
    ap._ign_state = (15.0, None, None, None)
    ap._brake_time = 10.0
    ap._burn_miss_now = np.array([500.0, 0.0])
    axis = ap._aero_steer(veh, np.array([0.0, 0.0, 1500.0]), veh.vel, veh.mass, np.zeros(3))
    assert math.degrees(math.acos(axis[2])) > 15.0
    # Tail-first lift points opposite the tilt, towards negative x.
    assert axis[0] > 0.0
    ap._burn_miss_now[:] = 0.0
    assert np.allclose(ap._aero_steer(veh, veh.pos, veh.vel, veh.mass, np.zeros(3)), [0, 0, 1])
