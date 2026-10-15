"""Closed-loop 2D landings (same limits as ``python main.py --landing-test``)."""

import pytest

pytest.importorskip("pygame")

from lander2d.scenarios import PRESETS, run_landing_scenario


@pytest.mark.parametrize(
    "params",
    [pytest.param(p, id=name, marks=[] if i == 0 else pytest.mark.slow) for i, (name, p) in enumerate(PRESETS)],
)
def test_preset_lands_on_the_pad(params):
    result = run_landing_scenario(**params)
    assert result["success"], result
