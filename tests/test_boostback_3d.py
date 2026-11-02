"""Boost-back numerical shooting, physical propagation and cutoff accounting."""

import copy
import math

import numpy as np
import pytest

from common.conic import ConicSolver
from lander3d.boostback import Prediction, correction, predict, steering, turn_towards
from lander3d.sim import Sim3D


def test_rank_deficient_endpoint_sensitivity_has_a_bounded_conic_step():
    # One axis has no authority. This used to fail a determinant test; an
    # elastic terminal constraint should leave that miss and correct the other.
    solver = ConicSolver()
    if not solver.available:
        pytest.skip("Clarabel unavailable")
    prediction = Prediction(np.zeros(3), np.array([100.0, 30.0]), 0.0, 500.0, 0.0, 20.0, True)
    jac = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    step, status = correction(solver, np.zeros(3), prediction, jac, np.zeros(3), 4.0, 100.0)
    assert status in ("Solved", "AlmostSolved")
    assert np.linalg.norm(step) <= 4.00001
    assert np.linalg.norm(prediction.miss + jac @ step) < np.linalg.norm(prediction.miss)


def test_predictor_owns_its_vehicle_actuators_and_aerodynamic_state():
    sim = Sim3D(cfd=False, scenario=3, wind_on=False)
    veh, ap = sim.vehicle, sim.autopilot
    state = {k: x.copy() for k, x in vars(veh).items() if isinstance(x, np.ndarray)}
    controls, info = copy.copy(veh.controls), copy.deepcopy(sim.aero.last)
    mass, throttle, propellant = veh.mass, veh.throttle, veh.propellant
    r, v, m = ap._state(veh)
    result = predict(ap, veh, r, v, m, np.zeros(3), np.array([-180.0, -15.0, 60.0]))
    assert result.valid and result.duration > 1.0 and result.fuel > 0.0
    assert result.fuel + result.reserve < propellant
    assert veh.mass == mass and veh.throttle == throttle and veh.propellant == propellant
    assert vars(veh.controls) == vars(controls)
    assert sim.aero.last == info
    for k, value in state.items():
        assert np.array_equal(getattr(veh, k), value), k


def test_zero_impulse_still_propagates_engine_shutdown():
    sim = Sim3D(cfd=False, scenario=3, wind_on=False)
    veh, ap = sim.vehicle, sim.autopilot
    r, v, m = ap._state(veh)
    off = predict(ap, veh, r, v, m, np.zeros(3), np.zeros(3))
    veh.throttle = 1.0
    tail = predict(ap, veh, r, v, m, np.zeros(3), np.zeros(3))
    assert off.fuel == 0.0
    assert tail.fuel > 0.0 and tail.duration >= 1.0 / veh.spec.throttle_rate
    assert veh.throttle == 1.0


def test_velocity_to_go_does_not_count_gravity_as_delivered_impulse():
    sim = Sim3D(cfd=False, scenario=3, wind_on=False)
    ap, veh = sim.autopilot, sim.vehicle
    ap._boostback = True
    ap._bb_dv = np.array([-100.0, 0.0, 50.0])
    ap._bb_next = 100.0
    before = np.linalg.norm(ap._bb_dv)
    # The navigation velocity changed, but the engine has delivered no thrust.
    veh.vel[2] -= 10.0
    sim.step(1.0 / 240.0)
    assert np.linalg.norm(ap._bb_dv) == pytest.approx(before)


def test_nose_first_cone_is_relative_to_air_and_direction_slew_is_bounded():
    sim = Sim3D(cfd=False)
    ap = sim.autopilot
    wind = np.array([0.0, 40.0, 0.0])
    velocity = np.array([80.0, 40.0, 0.0])
    d = steering(ap, np.array([1.0, 1.0, 1.0]), velocity, wind)
    assert math.degrees(math.acos(d[0])) == pytest.approx(ap.nose_first_aoa_deg)
    opposite = turn_towards(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), 0.1)
    assert np.linalg.norm(opposite) == pytest.approx(1.0)
    assert math.acos(opposite[2]) == pytest.approx(0.1)


def test_fuel_starved_prediction_is_not_accepted():
    sim = Sim3D(cfd=False, scenario=3, wind_on=False)
    sim.vehicle.rp1 = sim.vehicle.lox = 100.0
    r, v, m = sim.autopilot._state(sim.vehicle)
    result = predict(sim.autopilot, sim.vehicle, r, v, m, np.zeros(3), np.array([-180.0, 0.0, 60.0]))
    assert not result.valid


def test_ignition_event_interpolation_handles_an_infinite_previous_burn_time(monkeypatch):
    # Before descent the burn time is infinite. Interpolating infinity into
    # the first finite event used to poison the fuel reserve with NaN/inf.
    sim = Sim3D(cfd=False, scenario=3, wind_on=False)
    ap, veh = sim.autopilot, sim.vehicle
    calls = iter([(0.0, math.inf, np.array([0.0, 0.0, 1.0])),
                  (1.0, 3.0, np.array([0.0, 0.0, 1.0]))])
    monkeypatch.setattr(ap, "_burn_need", lambda *args: next(calls))
    r, v, m = ap._state(veh)
    with np.errstate(invalid="raise"):
        event = ap._coast(veh, r, v, m, np.zeros(3), interpolate=True)
    assert event is not None and np.isfinite(event[3]) and event[3] == 3.0
