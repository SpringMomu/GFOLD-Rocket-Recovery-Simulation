"""The lift the guidance predicts is the lift of the booster as the attitude loop trims it."""

import math

import numpy as np

from lander3d.sim import PHYSICS_DT, Sim3D
from lander3d.vehicle import atmosphere


def _sim():
    sim = Sim3D(cfd=False)
    sim.vehicle.reset(position=(0.0, 0.0, 1000.0), velocity=(0.0, 0.0, -100.0), on_pad=False)
    return sim


def test_the_untrimmed_quick_force_is_unchanged_without_a_trim():
    sim = _sim()
    rho, _, _, a = atmosphere(800.0)
    v = np.array([-20.0, 5.0, -110.0])
    axis = np.array([0.3, 0.1, 0.95])
    axis /= np.linalg.norm(axis)
    f = sim.aero.quick_force(v, axis, rho, a, 0.5)
    assert np.allclose(f, sim.aero.quick_force(v, axis, rho, a, 0.5, trim=None))
    half = sim.aero.quick_force(v, axis, rho, a, 0.5, trim=lambda th, q: 0.5)
    base = sim.aero.quick_force(v, -v / np.linalg.norm(v), rho, a, 0.5)
    # only the lift (what the attitude differs by from the aligned booster) is scaled, the drag is not
    lift, lift_half = f - base, half - base
    assert np.linalg.norm(lift_half) < np.linalg.norm(lift)


def test_trimmed_lift_factor_is_sane_and_can_be_switched_off():
    sim = _sim()
    ap, veh = sim.autopilot, sim.vehicle
    kappa = ap._lift_trim(veh)
    assert kappa is not None
    # aligned with the flow there is no lift, so no ratio: the factor is its
    # limit there (a default of 1 bent the lift curve over the first 10 deg)
    assert abs(kappa(0.0, 5000.0) - kappa(math.radians(2.0), 5000.0)) < 0.02
    for deg in (5, 10, 15, 20, 30, 60, 90, 150):
        k = kappa(math.radians(deg), 5000.0)
        assert 0.0 <= k <= 1.5, (deg, k)
    assert min(kappa(math.radians(d), 5000.0) for d in (5, 10, 15, 20)) < 0.9     # the fins do take lift off
    ap.lift_trim = False
    assert ap._lift_trim(veh) is None


def test_the_trimmed_model_is_closer_to_the_flown_lift_than_zero_deflection():
    sim = Sim3D(cfd=False)
    sim.load_scenario(2)                                            # 2000 m, straight down, in a cross wind
    sim.wind.configure(from_deg=(180.0, 180.0), speed=(12.0, 12.0))
    ap, veh = sim.autopilot, sim.vehicle
    err0, err1 = [], []
    next_sample = 5.0
    while sim.time < 15.0:
        sim.step(PHYSICS_DT)
        if sim.time >= next_sample:
            next_sample += 2.0
            vr = veh.vel - sim.wind.at(veh.pos, veh.time)
            rho, _, _, a = atmosphere(veh.pos[2])
            actual = veh.last["aero"] / veh.mass
            f0 = sim.aero.quick_force(vr, veh.axis, rho, a, veh.legs, fins=veh.fins_deploy) / veh.mass
            f1 = sim.aero.quick_force(vr, veh.axis, rho, a, veh.legs, fins=veh.fins_deploy, trim=ap._lift_trim(veh)) / veh.mass
            err0.append(float(np.linalg.norm(actual - f0)))
            err1.append(float(np.linalg.norm(actual - f1)))
    assert len(err0) >= 4
    assert np.mean(err1) < 0.5 * np.mean(err0), (np.mean(err0), np.mean(err1))


def test_the_lift_gain_is_calibrated_in_flight_and_stays_bounded():
    sim = Sim3D(cfd=False)
    sim.load_scenario(5)                                            # 12 km entry: high dynamic pressure
    sim.wind.configure(from_deg=(90.0, 90.0), speed=(12.0, 12.0))
    ap = sim.autopilot
    while sim.time < 30.0:
        sim.step(PHYSICS_DT)
    assert 0.3 <= ap._lift_gain <= 1.5
    # the trimmed model, taken at the Mach number flown, is the lift flown:
    # the calibration hardly has to correct it (at a fixed 1000 Pa -- Mach
    # 0.27 at 12 km instead of ~0.85 -- it had to take off ~40 %)
    assert 0.8 < ap._lift_gain < 1.2
