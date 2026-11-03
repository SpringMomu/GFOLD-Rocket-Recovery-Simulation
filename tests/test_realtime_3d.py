"""Bounded boost-back work and the separate, latched flight clock."""

import json
import time

import numpy as np
import pytest

from lander3d.autopilot import Autopilot3D
from lander3d.boostback import Prediction
from lander3d.server import snapshot
from lander3d.sim import PHYSICS_DT, Sim3D


def _work(prediction):
    def generator():
        yield prediction
        while True:
            yield
    return generator()


def test_threadless_browser_queues_boostback_without_solving_in_a_step(monkeypatch):
    def no_threads(*args, **kwargs):
        raise RuntimeError("threads unavailable")

    monkeypatch.setattr("threading.Thread.start", no_threads)
    sim = Sim3D(cfd=False, scenario=3, async_plan=True)
    ap = sim.autopilot
    assert not ap.async_plan and ap._bb_cooperative
    sim.step(PHYSICS_DT)
    assert ap.phase == "BOOSTBACK" and ap._bb_work is not None
    assert ap._bb_prediction is None and ap._bb_evaluations == 0
    before = sim.time
    sim.advance_realtime(0.02)
    assert sim.time > before                    # physics runs while the solve is pending
    assert ap._bb_work is not None
    ap.reset()
    ap.advance_boostback_work()
    assert ap._bb_work is None and ap._bb_dv is None


def test_committed_candidate_is_corrected_to_the_live_state():
    # The solve requests propulsive velocity, not the navigation velocity
    # change caused by gravity and drag while the work was in progress.
    ap = Autopilot3D()
    ap._boostback = True
    ap._bb_freeze_vel = np.array([10.0, 0.0, 0.0])
    ap._bb_vel_now = np.array([13.0, 2.0, -1.0])
    ap._bb_work_delivered = 2.0
    ap._bb_delivered = 12.0
    prediction = Prediction(np.array([60.0, 0.0, 80.0]), np.zeros(2), 500.0, 900.0, 5.0, 20.0, True)
    ap._bb_work = _work(prediction)
    ap.advance_boostback_work(0.001)
    planned = np.array([60.0, 0.0, 80.0]) * 0.9
    assert np.allclose(ap._bb_dv, planned)
    assert ap._bb_prediction is prediction
    assert ap._bb_reserve == 900.0
    assert ap._bb_work is None                  # a verified candidate ends that solve


def test_coasting_during_the_solve_does_not_add_gravity_to_the_burn():
    ap = Autopilot3D()
    ap._boostback = True
    ap._bb_freeze_vel = np.array([100.0, -20.0, 0.0])
    ap._bb_vel_now = np.array([85.0, -10.0, -30.0])
    prediction = Prediction(np.array([-180.0, -25.0, 75.0]), np.zeros(2), 500.0, 900.0, 5.0, 20.0, True)
    ap._bb_work = _work(prediction)
    ap.advance_boostback_work(0.001)
    assert np.array_equal(ap._bb_dv, prediction.impulse)


def test_accepted_provisional_solution_ends_its_frozen_search():
    ap = Autopilot3D()
    ap._boostback = True
    ap._w10 = np.array([0.0, 16.0, 0.0])
    prediction = Prediction(np.array([-180.0, 0.0, 75.0]), np.array([100.0, 0.0]), 500.0, 900.0, 5.0, 20.0, True)
    ap._bb_work = _work(prediction)
    ap.advance_boostback_work(0.001)
    assert ap._bb_prediction is prediction and ap._bb_work is None


def test_stale_correction_is_rejected_in_moderate_wind_too():
    ap = Autopilot3D()
    ap._w10 = np.array([0.0, 16.0, 0.0])
    ap._bb_dv = np.array([-100.0, 20.0, 30.0])
    original = ap._bb_dv.copy()
    ap._bb_work_started = 1.0
    ap.elapsed = 2.0
    ap._queue_boostback_candidate(Prediction(np.array([-30.0, 90.0, 20.0]), np.zeros(2), 500.0, 900.0, 5.0, 20.0, True))
    assert np.array_equal(ap._bb_dv, original)


def test_inflight_wind_updates_do_not_move_the_frozen_shooting_problem():
    sim = Sim3D(cfd=False, scenario=3)
    ap = sim.autopilot
    ap._bb_cooperative = True
    sim.step(PHYSICS_DT)
    frozen = ap._bb_work_ap
    before = frozen._wind(2000.0).copy()
    ap._w10 += np.array([10.0, -5.0, 0.0])
    ap._wind_ref[0][:] = 100.0
    assert np.array_equal(frozen._wind(2000.0), before)
    assert not np.shares_memory(ap._w10, frozen._w10)


def test_boostback_work_is_a_fixed_quota_per_simulated_second(monkeypatch):
    # Guidance per simulated second must not depend on the CPU: with the
    # quota paid in time the clock runs at wall speed ...
    sim = Sim3D(cfd=False, scenario=3)
    ap = sim.autopilot
    ap._bb_cooperative = True
    sim.step(PHYSICS_DT)
    budgets = []

    def pay(budget, deadline=None):
        budgets.append(budget)
        ap._bb_work_debt = 0.0

    monkeypatch.setattr(ap, "advance_boostback_work", pay)
    before = sim.time
    sim.advance_realtime(0.1)
    assert sim.time - before == pytest.approx(0.1, abs=PHYSICS_DT)
    assert sum(budgets) == pytest.approx(sim.time - before)


def test_simulated_clock_waits_for_an_unpaid_boostback_quota(monkeypatch):
    # ... and a CPU too slow to pay it (a phone) runs the same flight in slow
    # motion instead of flying a command solved seconds late.
    sim = Sim3D(cfd=False, scenario=3)
    ap = sim.autopilot
    ap._bb_cooperative = True
    sim.step(PHYSICS_DT)
    assert ap._bb_work is not None
    monkeypatch.setattr(ap, "boostback_slice_wall", 0.0)
    before = sim.time
    sim.advance_realtime(0.1)
    assert ap.boostback_work_pending()
    assert sim.time - before <= 2 * PHYSICS_DT


@pytest.mark.parametrize("wind_speed", [0.0, 12.0])
def test_urgent_first_prediction_uses_the_fast_wind_aware_path(wind_speed):
    sim = Sim3D(cfd=False, wind_on=False)
    veh, ap = sim.vehicle, sim.autopilot
    veh.reset(position=(3800.0, 0.0, 1000.0), velocity=(60.0, 0.0, -40.0),
              propellant=5000.0, on_pad=False)
    veh.pos[2] += veh.rest_height
    ap.reset()
    ap._submit_boostback(veh, np.array([wind_speed, 0.0, 0.0]))
    assert ap._bb_urgent_work
    assert ap._bb_fast_work


def test_realtime_boostback_work_slices_stay_short():
    # The browser path feeds the solve in slices, so a complete solve never
    # blocks one call.  The slice must still be long enough to make progress:
    # at 4 ms the command used to arrive many simulated seconds too late.
    sim = Sim3D(cfd=False, scenario=3)
    ap = sim.autopilot
    ap._bb_cooperative = True
    sim.step(PHYSICS_DT)
    times = []
    for _ in range(4000):
        start = time.perf_counter()
        ap.advance_boostback_work(0.002)
        times.append(time.perf_counter() - start)
        if ap._bb_dv is not None:
            break
    assert ap._bb_dv is not None and ap._bb_prediction is not None
    assert ap._bb_prediction.valid and np.linalg.norm(ap._bb_prediction.miss) <= 150.0
    assert max(times) < 0.08, f"work slice blocked for {max(times) * 1000:.0f} ms"


def test_physically_feasible_initializer_that_misses_the_pad_is_not_flown():
    ap = Autopilot3D()
    ap._boostback = True
    prediction = Prediction(np.array([-120.0, 0.0, 0.0]), np.array([1500.0, 0.0]), 500.0, 900.0, 5.0, 20.0, True)

    ap._bb_work = _work(prediction)
    ap.advance_boostback_work(0.001)
    assert prediction.valid and not prediction.return_ready
    assert ap._bb_dv is None and ap._bb_prediction is None


def test_urgent_partial_return_must_make_substantial_progress():
    ap = Autopilot3D()
    ap._boostback = ap._bb_urgent_work = True
    ap._burn_miss_now = np.array([4000.0, 0.0])
    weak = Prediction(np.array([-100.0, 0.0, 50.0]), np.array([3200.0, 0.0]), 500.0, 900.0, 5.0, 20.0, True)
    ap._queue_boostback_candidate(weak)
    assert ap._bb_dv is None
    useful = Prediction(np.array([-200.0, 0.0, 100.0]), np.array([900.0, 0.0]), 500.0, 900.0, 5.0, 20.0, True)
    ap._queue_boostback_candidate(useful)
    assert ap._bb_prediction is useful


def test_urgent_wait_brakes_descent_without_freezing_the_clock(monkeypatch):
    sim = Sim3D(cfd=False, wind_on=False)
    veh, ap = sim.vehicle, sim.autopilot
    veh.reset(position=(4000.0, 0.0, 1000.0), velocity=(60.0, 0.0, -40.0), propellant=5000.0, on_pad=False)
    veh.pos[2] += veh.rest_height
    ap.reset()
    ap._bb_cooperative = True
    sim.autopilot_on = True
    monkeypatch.setattr(ap, "advance_boostback_work", lambda budget, deadline=None: None)
    for _ in range(40):
        sim.advance_realtime(0.01)
    assert sim.time == pytest.approx(0.4, abs=PHYSICS_DT)
    assert ap.phase == "BOOSTBACK" and ap._bb_work is not None
    assert veh.throttle > 0.2


def test_urgent_wait_eventually_hands_over_if_no_return_is_found(monkeypatch):
    sim = Sim3D(cfd=False, wind_on=False)
    veh, ap = sim.vehicle, sim.autopilot
    veh.reset(position=(4000.0, 0.0, 1000.0), velocity=(60.0, 0.0, -40.0), propellant=5000.0, on_pad=False)
    veh.pos[2] += veh.rest_height
    ap.reset()
    ap._bb_cooperative = True
    sim.autopilot_on = True
    monkeypatch.setattr(ap, "advance_boostback_work", lambda budget, deadline=None: None)
    # Far off and flying away, the booster does not wait at zero throttle:
    # it turns the outward motion round towards the pad while it waits, and
    # hands over once it is heading back at a conservative return speed.
    while sim.time < 10.0 and not ap._boostback_done:
        sim.advance_realtime(0.02)
    assert ap._boostback_done and ap._bb_work is None
    assert ap._bb_exit_reason == "no-command"
    assert veh.vel[0] < -50.0, veh.vel


@pytest.mark.slow
@pytest.mark.parametrize("wind_speed", [0.0, 10.0, 20.0])
def test_distant_descending_booster_lights_and_turns_back_in_wind(wind_speed):
    sim = Sim3D(cfd=False, wind_on=wind_speed > 0)
    sim.apply_action("windcfg:" + json.dumps({"from": [180, 180], "speed": [wind_speed, wind_speed], "shear": 1}))
    veh, ap = sim.vehicle, sim.autopilot
    veh.reset(position=(4000.0, 0.0, 1000.0), velocity=(60.0, 0.0, -40.0), propellant=5000.0, on_pad=False)
    veh.pos[2] += veh.rest_height
    ap.reset()
    ap._bb_cooperative = True
    sim.autopilot_on = True
    while sim.time < 12.0 and veh.state == "FLYING":
        sim.advance_realtime(0.016)
    assert veh.state == "FLYING" and veh.altitude > 500.0
    assert veh.vel[0] < -20.0 and veh.pos[0] < 4000.0
    assert ap._bb_prediction is not None and ap._bb_prediction.valid


def test_boostback_snapshot_updates_accepted_prediction_as_vehicle_flies():
    sim = Sim3D(cfd=False, wind_on=False)
    veh, ap = sim.vehicle, sim.autopilot
    veh.reset(position=(3800.0, 0.0, 1000.0), velocity=(60.0, 0.0, -40.0),
              propellant=5000.0, on_pad=False)
    veh.pos[2] += veh.rest_height
    ap.reset()
    ap._bb_cooperative = True
    sim.autopilot_on = True
    while sim.time < 3.0 and ap._bb_prediction is None:
        sim.advance_realtime(0.01)
    assert ap._bb_prediction is not None
    while sim.time < 1.0:
        sim.advance_realtime(0.01)
    first = snapshot(sim)["plan"]
    start_time = sim.time
    while sim.time < start_time + 0.5:
        sim.advance_realtime(0.01)
    second = snapshot(sim)["plan"]
    assert ap.phase == "BOOSTBACK"
    assert len(first) > 2 and len(second) > 2
    assert first[0] != second[0]
    assert np.linalg.norm(np.array(second[0][:2]) - veh.pos[:2]) < 20.0
    assert second[-1][2] == pytest.approx(ap.h_off + ap.gate_height)


@pytest.mark.slow
@pytest.mark.parametrize("outward_speed", [30.0, 60.0])
def test_low_far_realtime_boostback_ignites_before_losing_its_return_window(outward_speed):
    sim = Sim3D(cfd=False, wind_on=False)
    veh, ap = sim.vehicle, sim.autopilot
    veh.reset(position=(3800.0, 0.0, 1000.0), velocity=(outward_speed, 0.0, -40.0),
              propellant=5000.0, on_pad=False)
    veh.pos[2] += veh.rest_height
    ap.reset()
    ap._bb_cooperative = True
    sim.autopilot_on = True
    ignited = False
    while sim.time < 12.0 and veh.state == "FLYING":
        sim.advance_realtime(0.01)
        ignited |= ap.phase == "BOOSTBACK" and veh.throttle > 0.5
    assert ignited, (ap.phase, ap._bb_status, ap._bb_exit_reason)
    assert veh.altitude > ap.boostback_end_alt
    assert veh.vel[0] < -20.0, (veh.pos, veh.vel, ap._bb_exit_reason)


def test_old_strong_wind_solution_does_not_replace_the_flying_command():
    ap = Autopilot3D()
    ap._w10 = np.array([0.0, 21.0, 0.0])
    ap._bb_dv = np.array([-100.0, 20.0, 30.0])
    original = ap._bb_dv.copy()
    ap._bb_work_started = 1.0
    ap.elapsed = 2.0
    outdated = Prediction(np.array([-30.0, 90.0, 20.0]), np.zeros(2), 500.0, 900.0, 5.0, 20.0, True)
    ap._queue_boostback_candidate(outdated)
    assert np.array_equal(ap._bb_dv, original)
    assert ap._bb_prediction is None


@pytest.mark.parametrize("scenario", [3, 7], ids=["scenario-4", "scenario-8"])
@pytest.mark.parametrize("wind_on", [False, True], ids=["calm", "default-wind"])
def test_slow_guidance_returns_before_coasting(scenario, wind_on):
    # Checking only the final touchdown hid the old 1.7--2.3 km miss at
    # boost-back shutdown.  The work budget is in simulated seconds now (the
    # same solve spans the same simulated time on any machine), so the slow
    # machine is modelled by the tool's own frame loop, not by shrinking it.
    sim = Sim3D(cfd=False, scenario=scenario, wind_on=wind_on)
    ap = sim.autopilot
    ap.max_rate = np.radians(60.0)                  # live site's existing attitude setting
    ap._bb_cooperative = True
    while sim.time < 25.0 and not ap._boostback_done:
        sim.advance_realtime(0.01)
    assert ap._bb_exit_reason == "impulse-complete"
    assert np.linalg.norm(ap._bb_prediction.miss) < 150.0
    # Include the shutdown ramp. No powered landing correction has happened.
    end = sim.time + 1.2
    while sim.time < end:
        sim.advance_realtime(0.01)
    assert not ap.burn_locked
    r, v, m = ap._state(sim.vehicle)
    w = sim.wind.at(sim.vehicle.pos, sim.time)
    ignition = ap._coast(sim.vehicle, r, v, m, w)
    assert ignition is not None
    miss, _ = ap._burn_miss(sim.vehicle, ignition[1], ignition[2], m, w)
    # The boost-back leaves the glide some sideways error to take out (the
    # glide's own first second already moves the ballistic aim); the old
    # failure was 1.7--2.3 km at shutdown, so a few hundred metres proves the
    # boost-back really came back.
    assert np.linalg.norm(miss) < 250.0, (miss, ap._bb_exit_reason, r, v)


@pytest.mark.parametrize("state", ["LANDED", "CAUGHT"])
def test_flight_clock_latches_on_the_success_step_and_resets(state, monkeypatch):
    sim = Sim3D(cfd=False, scenario=0)
    sim.autopilot_on = False
    sim.step(PHYSICS_DT)
    assert sim.flight_time == sim.time > 0.0

    def succeed(*args, **kwargs):
        sim.vehicle.state = state

    monkeypatch.setattr(sim.vehicle, "step", succeed)
    sim.step(PHYSICS_DT)
    stopped = snapshot(sim)["flight_time"]
    assert stopped == round(sim.time, 3)
    for _ in range(100):
        sim.step(PHYSICS_DT)
    after = snapshot(sim)
    assert after["flight_time"] == stopped and after["t"] > stopped
    sim.load_scenario(0)
    assert sim.flight_time == 0.0 and not sim._flight_finished
    sim.reset_to_pad()
    assert snapshot(sim)["flight_time"] == 0.0
def test_scenario_8_comes_home_in_a_13_m_s_south_wind():
    # The reported failure: with a steady 13 m/s south wind the cooperative
    # boost-back never commanded ("no-command") and the booster fell 1-3 km
    # from the pad.  It must boost back, hand over near the tower and be
    # caught.
    sim = Sim3D(cfd=False, scenario=7, wind_on=True)
    sim.apply_action("windcfg:" + json.dumps({"from": [180, 180], "speed": [13, 13], "shear": 1}))
    ap = sim.autopilot
    ap._bb_cooperative = True
    while sim.time < 150.0 and sim.vehicle.state == "FLYING":
        sim.advance_realtime(1.0 / 60.0)
    assert ap._bb_exit_reason == "impulse-complete", ap._bb_exit_reason
    assert np.linalg.norm(ap._bb_prediction.miss) < 100.0, ap._bb_prediction.miss
    assert sim.vehicle.state == "CAUGHT", (sim.vehicle.state, sim.vehicle.crash_reason)


@pytest.mark.slow
@pytest.mark.parametrize("work_scale", [0.4, 1.0])
def test_scenario_8_16m_s_south_wind_returns_at_different_solver_speeds(work_scale):
    sim = Sim3D(cfd=False, scenario=7, wind_on=True)
    sim.apply_action("windcfg:" + json.dumps({"from": [180, 180], "speed": [16, 16], "shear": 1}))
    ap, veh = sim.autopilot, sim.vehicle
    ap._bb_cooperative = True
    original = ap.advance_boostback_work
    ap.advance_boostback_work = lambda budget, deadline=None: original(budget * work_scale, deadline)
    while sim.time < 30.0 and not ap._boostback_done and veh.state == "FLYING":
        sim.advance_realtime(0.016)
    assert ap._bb_exit_reason == "impulse-complete"
    while sim.time < 35.0 and veh.throttle > 0.01 and not ap.burn_locked:
        sim.advance_realtime(0.016)
    r, v, mass = ap._state(veh)
    wind = sim.wind.at(veh.pos, sim.time)
    ignition = ap._coast(veh, r, v, mass, wind)
    assert ignition is not None
    miss, _ = ap._burn_miss(veh, ignition[1], ignition[2], mass, wind)
    assert np.linalg.norm(miss) < 300.0, miss
    while sim.time < 150.0 and veh.state == "FLYING":
        sim.advance_realtime(0.016)
    assert veh.state == "CAUGHT", (veh.state, veh.crash_reason, veh.touchdown)
    assert veh.touchdown["error"] <= 1.0 and veh.touchdown["vh"] <= 1.0, veh.touchdown


@pytest.mark.parametrize("max_rate", [30.0, 60.0])
def test_scenario_8_boostback_in_a_21_m_s_south_wind(max_rate):
    sim = Sim3D(cfd=False, scenario=7, wind_on=True)
    sim.apply_action("windcfg:" + json.dumps({"from": [180, 180], "speed": [21, 21], "shear": 1}))
    ap = sim.autopilot
    ap.max_rate = np.radians(max_rate)
    ap._bb_cooperative = True
    while sim.time < 25.0 and not ap._boostback_done:
        sim.advance_realtime(0.016)
    assert ap._bb_exit_reason == "impulse-complete"
    assert ap._bb_prediction is not None and ap._bb_prediction.valid
    # Account for engine shutdown; check the physical handover, not merely a
    # candidate's optimistic prediction or the final landing result.
    end = sim.time + 1.2
    while sim.time < end:
        sim.advance_realtime(0.016)
    assert not ap.burn_locked
    r, v, mass = ap._state(sim.vehicle)
    wind = sim.wind.at(sim.vehicle.pos, sim.time)
    ignition = ap._coast(sim.vehicle, r, v, mass, wind)
    assert ignition is not None
    miss, _ = ap._burn_miss(sim.vehicle, ignition[1], ignition[2], mass, wind)
    assert np.linalg.norm(miss) < 150.0, miss


def test_scenario_8_boostback_in_a_21_m_s_north_wind():
    sim = Sim3D(cfd=False, scenario=7, wind_on=True)
    sim.apply_action("windcfg:" + json.dumps({"from": [0, 0], "speed": [21, 21], "shear": 1}))
    ap = sim.autopilot
    ap.max_rate = np.radians(60.0)
    ap._bb_cooperative = True
    while sim.time < 25.0 and not ap._boostback_done:
        sim.advance_realtime(0.016)
    assert ap._bb_exit_reason == "impulse-complete"
    end = sim.time + 1.2
    while sim.time < end:
        sim.advance_realtime(0.016)
    assert not ap.burn_locked
    r, v, mass = ap._state(sim.vehicle)
    wind = sim.wind.at(sim.vehicle.pos, sim.time)
    ignition = ap._coast(sim.vehicle, r, v, mass, wind)
    assert ignition is not None
    miss, _ = ap._burn_miss(sim.vehicle, ignition[1], ignition[2], mass, wind)
    assert np.linalg.norm(miss) < 250.0, miss
