"""Closed-loop 3D landings (same limits as ``python main3d.py --test``)."""

import json

import numpy as np
import pytest

from lander3d.planner import KeepOutBox, along_path, detour_path, separating_planes
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


def test_snapshot_is_valid_json_for_the_browser():
    sim = Sim3D(cfd=False)
    sim.advance_realtime(0.1)
    # allow_nan=False is what the server uses: a NaN would break the front end.
    text = json.dumps(finite(snapshot(sim)), allow_nan=False)
    assert json.loads(text)["scenario"] == sim.scenario_name
