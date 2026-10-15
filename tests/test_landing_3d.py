"""Closed-loop 3D landings (same limits as ``python main3d.py --test``)."""

import json

import pytest

from lander3d.regression import run_scenario
from lander3d.server import finite, snapshot
from lander3d.sim import SCENARIOS, Sim3D


@pytest.mark.parametrize("index", [pytest.param(i, id=sc["name"], marks=[] if i == 0 else pytest.mark.slow)
                                   for i, sc in enumerate(SCENARIOS)])
def test_scenario_lands_on_the_pad(index):
    result = run_scenario(index)
    assert result["success"], result


def test_snapshot_is_valid_json_for_the_browser():
    sim = Sim3D(cfd=False)
    sim.advance_realtime(0.1)
    # allow_nan=False is what the server uses: a NaN would break the front end.
    text = json.dumps(finite(snapshot(sim)), allow_nan=False)
    assert json.loads(text)["scenario"] == sim.scenario_name
