"""Boost-back numerical shooting, physical propagation and cutoff accounting."""

import copy
import math

import numpy as np
import pytest

from common.conic import ConicSolver
from lander3d.boostback import (
    Prediction,
    braking_handover,
    correction,
    predict,
    prediction_matches_state,
    steering,
    turn_towards,
)
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
    assert len(result.trajectory) > 3
    assert np.allclose(result.trajectory[0][1], r)
    assert any(t == pytest.approx(result.duration + result.coast_time) for t, _ in result.trajectory)
    assert np.allclose(result.trajectory[-1][1][:2], result.miss)
    assert all(a[0] < b[0] for a, b in zip(result.trajectory, result.trajectory[1:]))
    assert result.fuel + result.reserve < propellant
    assert veh.mass == mass and veh.throttle == throttle and veh.propellant == propellant
    assert vars(veh.controls) == vars(controls)
    assert sim.aero.last == info
    for k, value in state.items():
        assert np.array_equal(getattr(veh, k), value), k


def test_boostback_display_follows_live_state_and_ages_accepted_trajectory():
    sim = Sim3D(cfd=False, wind_on=False)
    ap = sim.autopilot
    ap._bb_prediction = Prediction(np.ones(3), np.zeros(2), 1.0, 1.0, 2.0, 2.0, True,
                                   trajectory=[(0.0, np.array([10.0, 0.0, 100.0])),
                                               (1.0, np.array([20.0, 0.0, 90.0])),
                                               (2.0, np.array([30.0, 0.0, 80.0]))])
    ap._bb_trajectory_epoch = 4.0
    ap.elapsed = 5.1
    ap._update_boostback_display(np.array([23.0, 1.0, 88.0]))
    assert ap.predicted_points == [(23.0, 1.0, 88.0), (30.0, 0.0, 80.0)]
    ap.elapsed = 5.3
    ap._update_boostback_display(np.array([25.0, 1.0, 86.0]))
    assert ap.predicted_points[0] == (25.0, 1.0, 86.0)


def test_delayed_prediction_requires_the_real_powered_state_to_match():
    p = Prediction(np.ones(3), np.zeros(2), 1.0, 1.0, 2.0, 1.0, True,
                   trajectory=[(0.0, np.array([0.0, 0.0, 100.0])),
                               (1.0, np.array([10.0, 0.0, 90.0])),
                               (2.0, np.array([20.0, 0.0, 80.0]))])
    assert prediction_matches_state(p, 1.5, np.array([15.0, 0.0, 85.0]), np.array([10.0, 0.0, -10.0]))
    assert not prediction_matches_state(p, 1.5, np.array([50.0, 0.0, 85.0]), np.array([10.0, 0.0, -10.0]))
    assert not prediction_matches_state(p, 1.5, np.array([15.0, 0.0, 85.0]), np.array([-20.0, 0.0, -10.0]))
    assert not prediction_matches_state(p, 3.0, np.array([15.0, 0.0, 85.0]), np.array([10.0, 0.0, -10.0]))


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


def test_hop_hands_over_only_when_braking_the_return():
    sim = Sim3D(cfd=False, wind_on=False)
    ap = sim.autopilot
    ap._hop = True
    up, still = np.array([0.0, 0.0, 1.0]), np.zeros(3)
    # Still flying away from the pad (+x): the retrograde burn that turns the
    # booster round is the return itself, not a braking handover.
    assert not braking_handover(ap, np.array([-200.0, 0.0, 50.0]), up, np.array([80.0, 0.0, -20.0]), still,
                                np.array([3000.0, 0.0, 800.0]))
    # Flying back towards the pad and the change wanted brakes the approach.
    assert braking_handover(ap, np.array([60.0, 0.0, 10.0]), up, np.array([-80.0, 0.0, -20.0]), still,
                            np.array([1500.0, 0.0, 800.0]))


def test_landing_burn_is_not_due_at_the_apogee_of_a_lofted_return():
    # 8.3 km up, 3.3 km out, 128 m/s towards the pad, just past the apogee:
    # the burn time over that height is about an hour, not the 21 s the
    # floored deceleration gave -- whose lateral term lit the burn up there.
    ap = Sim3D(cfd=False, wind_on=False).autopilot
    veh = Sim3D(cfd=False, wind_on=False).vehicle
    r = np.array([3337.0, 0.0, 8339.0])
    v = np.array([-128.0, 0.0, -3.3])
    frac, t_burn, direction = ap._burn_need(veh, r, v, veh.mass, 0.0)
    assert t_burn == pytest.approx(2.0 * (r[2] - 0.45 * 3.3) / (3.3 + ap.gate_speed), rel=1e-6)
    assert frac < ap._ignition_threshold(t_burn, direction)
    # where the deceleration floor does not bind, the burn time is unchanged
    v = np.array([0.0, 0.0, -150.0])
    r = np.array([0.0, 0.0, 1500.0])
    _, t_burn, _ = ap._burn_need(veh, r, v, veh.mass, 0.0)
    h_eff = r[2] - 0.45 * 150.0
    a_v = (150.0 ** 2 - ap.gate_speed ** 2) / (2.0 * h_eff)
    assert t_burn == pytest.approx((150.0 - ap.gate_speed) / a_v, rel=1e-9)
