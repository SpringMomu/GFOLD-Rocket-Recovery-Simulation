"""Closed-loop 3D landings (same limits as ``python main3d.py --test``)."""

import json

import numpy as np
import pytest

from lander3d.planner import KeepOutBox, Plan3D, along_path, detour_path, separating_planes
from lander3d.regression import run_scenario
from lander3d.server import finite, snapshot
from lander3d.sim import SCENARIOS, Sim3D
from lander3d.tower import ARM_CLOSED, FACE_CLEARANCE, CatchTower
from lander3d.vehicle import Vehicle


@pytest.mark.parametrize("index", [pytest.param(i, id=sc["name"], marks=[] if i == 0 else pytest.mark.slow)
                                   for i, sc in enumerate(SCENARIOS)])
def test_scenario_lands_or_is_caught(index):
    result = run_scenario(index)
    assert result["success"], result


def test_closed_arms_clear_a_centred_booster():
    tower = CatchTower()
    tower.angle = ARM_CLOSED
    veh = Vehicle()
    veh.reset(position=(0.0, 0.0, tower.catch_base_z), on_pad=False)
    assert all(abs(g - FACE_CLEARANCE) < 1e-9 for g in tower.face_gaps(veh))


def test_keep_out_goes_over_the_tower_and_keeps_east_of_it_at_the_end():
    boxes = [KeepOutBox(c - np.array([0.0, 0.0, CatchTower().catch_base_z]), ax, hf)
             for c, ax, hf in CatchTower.keep_out_boxes(Vehicle().spec)]
    start, gate = np.array([-150.0, 0.0, 85.0]), np.zeros(3)
    path = detour_path(boxes, start, gate)
    pts = along_path(path, np.linspace(0.0, 1.0, 400))
    inside = [p for p in pts for b in boxes if np.all(np.abs(b.axes.T @ (p - b.center)) < b.half)]
    assert not inside and max(p[2] for p in path[1:-1]) > 30.0    # over the top, not through
    # at the gate the tower plane is its east face, which the gate satisfies
    (_, n, b), = [kp for kp in separating_planes(boxes[:1], [gate], target=gate)]
    assert np.allclose(n, [1.0, 0.0, 0.0]) and n @ gate >= b


def test_relaxed_plans_are_judged_without_the_limits_they_relax():
    z = np.zeros(3)
    plan = Plan3D(t_f=10.0, dt=1.0, r=z, v=z, z=z, u=z, sigma=z, drag=z, status="Solved", objective=0.0,
                  terminal_position_error=0.2, terminal_velocity_error=0.1, glide_violation=0.0, fuel_violation=0.0,
                  lateral_violation=8.0, climb_violation=0.0)
    assert not plan.reaches_pad                      # far too fast sideways for the normal limits
    plan.relax = 1
    assert not plan.reaches_pad                      # level 1 only frees the vertical profile
    plan.relax = 2
    assert plan.reaches_pad
    plan.climb_violation = 3.0                       # ... but even relaxed, no climbing beyond 5 m/s
    assert not plan.reaches_pad


def test_snapshot_is_valid_json_for_the_browser():
    sim = Sim3D(cfd=False)
    sim.advance_realtime(0.1)
    # allow_nan=False is what the server uses: a NaN would break the front end.
    text = json.dumps(finite(snapshot(sim)), allow_nan=False)
    assert json.loads(text)["scenario"] == sim.scenario_name


def _first_phase(pos, vel):
    sim = Sim3D(cfd=False)
    sim.vehicle.reset(position=pos, velocity=vel, propellant=4000, on_pad=False)
    sim.vehicle.pos[2] += sim.vehicle.rest_height
    sim.autopilot.reset()
    sim.autopilot_on = True
    sim.step(1.0 / 240.0)
    return sim.autopilot.phase


def test_mid_range_miss_boosts_back_instead_of_a_low_sideways_burn():
    # ~240 m off from a slow 2 km fall: the glide cannot remove it and the
    # landing burn would brake to a hover at ~80 m to traverse sideways.
    assert _first_phase((150.0, 100.0, 2000.0), (0.0, 0.0, -30.0)) == "BOOSTBACK"
    # A small (wind) drift is left to the glide.
    assert _first_phase((0.0, 0.0, 2000.0), (0.0, 0.0, -30.0)) == "COAST"
