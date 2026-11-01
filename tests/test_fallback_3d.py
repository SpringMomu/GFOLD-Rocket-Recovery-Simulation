"""A plan that has run away is not flown or used as a reference; the plan-less fallback law brings the booster down."""

import numpy as np
import pytest

from lander3d.planner import Plan3D
from lander3d.sim import PHYSICS_DT, Sim3D


def _plan(**kw):
    z = np.zeros((4, 3))
    args = dict(t_f=10.0, dt=1.0, r=z, v=z, z=np.zeros(4), u=z, sigma=np.zeros(4), drag=z, status="Solved", objective=0.0,
                terminal_position_error=0.0, terminal_velocity_error=0.0, fuel_violation=0.0)
    args.update(kw)
    return Plan3D(**args)


def test_a_runaway_plan_is_not_sane():
    assert _plan().sane
    assert not _plan(terminal_velocity_error=18.0).sane          # trades the landing speed for distance
    assert not _plan(fuel_violation=0.8).sane                    # burns propellant that is not there
    assert not _plan(v=np.full((4, 3), 900.0)).sane              # speeds no booster has
    assert not _plan(v=np.full((4, 3), np.nan)).sane
    assert _plan(terminal_position_error=1500.0).sane            # short of the pad, but a landing


def test_a_runaway_plan_is_no_reference_for_the_next_one():
    ap = Sim3D(cfd=False).autopilot
    ap.burn_locked = True
    ap.plan = _plan()
    assert ap._ref_plan() is ap.plan
    ap.plan = _plan(terminal_velocity_error=50.0)
    assert ap._ref_plan() is None


def test_the_touchdown_time_search_stops_where_the_propellant_does():
    sim = Sim3D(cfd=False)
    veh, ap = sim.vehicle, sim.autopilot
    spec = veh.spec
    mdot_min = ap.minimum_throttle * spec.thrust_sl / (spec.isp_sl * 9.80665)
    t_cap = ap._burn_time_limit(veh, veh.mass)
    assert t_cap == pytest.approx(0.92 * (veh.mass - spec.dry_mass - 150.0) / mdot_min, rel=1e-3)
    assert ap._burn_time_limit(veh, spec.dry_mass + 150.0 + 0.5 * (veh.mass - spec.dry_mass - 150.0)) == pytest.approx(0.5 * t_cap, rel=1e-3)


def test_the_fallback_law_lands_without_a_plan():
    sim = Sim3D(cfd=False)
    sim.load_scenario(0)
    sim.autopilot.planner.solver.backend = "feedback-fallback"      # no solver: no plans at all
    while sim.time < 30.0 and sim.vehicle.state == "FLYING":
        sim.step(PHYSICS_DT)
    td = sim.vehicle.touchdown or {}
    assert sim.vehicle.state == "LANDED", (sim.vehicle.crash_reason, td)
    assert abs(td["vz"]) < 6.0 and td["vh"] < 2.0


@pytest.mark.slow
def test_a_boostback_in_a_15_m_s_wind_does_not_crash():
    """At 15 m/s (30 m/s aloft) the pad is out of reach from this boost-back; the booster must still come down softly."""
    sim = Sim3D(cfd=False)
    sim.load_scenario(3)
    sim.wind.configure(speed=(15.0, 15.0))
    while sim.time < 120.0 and sim.vehicle.state == "FLYING":
        sim.step(PHYSICS_DT)
    td = sim.vehicle.touchdown or {}
    assert sim.vehicle.state == "LANDED", (sim.vehicle.state, sim.vehicle.crash_reason, td)
    assert abs(td["vz"]) < 4.0 and td["vh"] < 5.0
