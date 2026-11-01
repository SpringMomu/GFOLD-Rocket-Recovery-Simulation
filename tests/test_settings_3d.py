"""Guidance settings of the front end's solver panel (action "solvercfg:<json>")."""

import json
import math

import numpy as np

from lander3d import settings
from lander3d.autopilot import Autopilot3D
from lander3d.planner import Planner3D
from lander3d.server import finite, snapshot
from lander3d.sim import Sim3D

# option key -> (owner, attribute)
TARGETS = {
    "wind_plan_gain": ("ap", "wind_plan_gain"), "aero_pred": ("ap", "aero_pred"), "trim_gain": ("ap", "trim_gain"),
    "shear_fit": ("ap", "shear_fit"), "aoa_margin": ("ap", "aoa_moment_margin"), "time_weight": ("ap", "time_weight"),
    "divert_tilt_deg": ("ap", "divert_tilt_deg"), "burn_fraction": ("ap", "burn_fraction_nominal"),
    "time_search_evals": ("ap", "time_search_evals"), "retime_step": ("ap", "retime_step"),
    "lateral_aero_share": ("pl", "lateral_aero_share"), "knots": ("pl", "knots"),
    "lateral_accel_weight": ("pl", "lateral_accel_weight"), "jerk_weight": ("pl", "jerk_weight"),
    "lateral_rate": ("pl", "lateral_rate"), "max_iter": ("pl", "max_iter"),
}


def test_defaults_are_the_original_values():
    assert set(TARGETS) == set(settings.BY_KEY)
    for key, (owner, attr) in TARGETS.items():
        original = getattr(Autopilot3D if owner == "ap" else Planner3D, attr)
        assert float(original) == float(settings.BY_KEY[key].default), key


def test_apply_puts_the_values_on_the_guidance():
    sim = Sim3D(cfd=False)
    values = settings.defaults()
    values.update(wind_plan_gain=0.4, knots=24, shear_fit=0, burn_fraction=0.7, max_iter=60)
    settings.apply(sim.autopilot, values)
    assert sim.autopilot.wind_plan_gain == 0.4 and sim.autopilot.planner.knots == 24
    assert sim.autopilot.shear_fit is False and sim.autopilot.planner.max_iter == 60
    assert sim.autopilot.burn_fraction == 0.7          # not locked yet: used at once
    sim.autopilot.reset()
    assert sim.autopilot.burn_fraction == 0.7 and sim.autopilot.wind_shear_est == 1.0


def test_solvercfg_action_clamps_and_ignores_bad_input():
    sim = Sim3D(cfd=False)
    sim.apply_action('solvercfg:{"wind_plan_gain": 9, "knots": 3, "aero_pred": 1.4, "trim_gain": 0.5, "nope": 1}')
    c = sim.solver_cfg
    assert c["wind_plan_gain"] == 1.5 and c["knots"] == 20 and c["aero_pred"] == 1 and c["trim_gain"] == 0.5
    before = dict(c)
    for bad in ("nonsense", "null", "[1, 2]", '{"knots": "a"}', '{"knots": [1]}', '{"trim_gain": NaN}',
                '{"trim_gain": Infinity}', '{"shear_fit": null}', '{"max_iter": {}}'):
        sim.apply_action("solvercfg:" + bad)
        assert sim.solver_cfg == before, bad
    sim.apply_action("solverreset")
    assert sim.solver_cfg == settings.defaults() and sim.autopilot.wind_plan_gain == 1.0


def test_snapshot_reports_values_and_ranges():
    sim = Sim3D(cfd=False)
    sim.apply_action('solvercfg:{"lateral_rate": 2.5}')
    snap = json.loads(json.dumps(finite(snapshot(sim)), allow_nan=False))
    assert snap["solver_cfg"]["lateral_rate"] == 2.5
    assert {o[0] for o in snap["solver_opts"]} == set(settings.BY_KEY)
    for key, kind, lo, hi, step, default in snap["solver_opts"]:
        assert lo <= default <= hi and step > 0 and kind in ("float", "int", "bool"), key


def test_wind_options_change_what_the_planner_sees():
    sim = Sim3D(cfd=False)
    ap, veh = sim.autopilot, sim.vehicle
    ap._wind_ref = (np.array([10.0, 0.0, 0.0]), 10.0)
    base = ap._wind_plan(100.0).copy()
    assert np.array_equal(base, ap._wind(100.0))                       # default: untouched
    ap.wind_plan_gain = 0.5
    assert np.allclose(ap._wind_plan(100.0), 0.5 * base)
    ap.wind_plan_gain = 1.0
    full = ap._gate_trim(veh)
    assert np.linalg.norm(full) > 0.1
    ap.trim_gain = 0.5
    assert np.allclose(ap._gate_trim(veh), 0.5 * full)
    ap.trim_gain = 1.0
    # aerodynamic prediction: none -> zero force; drag only -> no lift gradient
    model = ap._drag_model(veh, np.array([10.0, 0.0, 0.0]), 20.0)
    t = np.linspace(0.5, 9.5, 10)
    _, _, q2, w2, _ = model(t)
    ap.aero_pred = 1
    drag1, G1, _, _, _ = ap._drag_model(veh, np.array([10.0, 0.0, 0.0]), 20.0)(t)
    ap.aero_pred = 0
    drag0, G0, q0, w0, _ = ap._drag_model(veh, np.array([10.0, 0.0, 0.0]), 20.0)(t)
    assert np.all(G1 == 0) and np.all(drag0 == 0) and np.all(G0 == 0)
    assert np.array_equal(q0, q2) and np.array_equal(w0, w2)          # the angle-of-attack cap still sees the flow
    assert math.isfinite(float(np.sum(drag1)))


def test_planner_options_change_the_problem():
    sim = Sim3D(cfd=False)
    pl = sim.autopilot.planner
    veh = sim.vehicle
    veh.reset(position=(0, 0, 400), velocity=(10, 0, -40), propellant=3000, on_pad=False)
    r, v, m = np.array([0.0, 0.0, 400.0]), np.array([10.0, 0.0, -40.0]), float(veh.mass)
    cfg = sim.autopilot._config(veh, 0.85)
    base = pl.solve(r, v, m, 14.0, cfg)
    assert base is not None
    pl.knots = 24
    coarse = pl.solve(r, v, m, 14.0, cfg)
    assert coarse is not None and coarse.r.shape[0] < base.r.shape[0]
