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
    # The solve runs from a frozen state; the command must leave the booster
    # at the planned burnout velocity. The measured velocity change already
    # includes thrust delivered during the solve, so it is subtracted once.
    ap = Autopilot3D()
    ap._boostback = True
    ap._bb_freeze_vel = np.array([10.0, 0.0, 0.0])
    ap._bb_vel_now = np.array([13.0, 2.0, -1.0])
    ap._bb_work_delivered = 2.0
    ap._bb_delivered = 12.0
    prediction = Prediction(np.array([60.0, 0.0, 80.0]), np.zeros(2), 500.0, 900.0, 5.0, 20.0, True)
    ap._bb_work = _work(prediction)
    ap.advance_boostback_work(0.001)
    planned = np.array([60.0, 0.0, 80.0]) - np.array([3.0, 2.0, -1.0])
    assert np.allclose(ap._bb_dv, planned)
    assert ap._bb_prediction is prediction
    assert ap._bb_reserve == 900.0
    assert ap._bb_work is None                  # a verified candidate ends that solve


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


def test_a_slow_callback_does_not_expand_the_next_boostback_slice(monkeypatch):
    sim = Sim3D(cfd=False, scenario=3)
    sim.autopilot._bb_cooperative = True
    sim.step(PHYSICS_DT)
    budgets = []
    monkeypatch.setattr(sim.autopilot, "advance_boostback_work", budgets.append)
    before = sim.time
    sim.advance_realtime(0.1)
    assert sim.time - before <= 0.008 + PHYSICS_DT
    assert budgets == [0.008]


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
    assert ap._bb_prediction.return_ready
    assert max(times) < 0.08, f"work slice blocked for {max(times) * 1000:.0f} ms"


def test_physically_feasible_initializer_that_misses_the_pad_is_not_flown():
    ap = Autopilot3D()
    ap._boostback = True
    prediction = Prediction(np.array([-120.0, 0.0, 0.0]), np.array([1500.0, 0.0]), 500.0, 900.0, 5.0, 20.0, True)

    ap._bb_work = _work(prediction)
    ap.advance_boostback_work(0.001)
    assert prediction.valid and not prediction.return_ready
    assert ap._bb_dv is None and ap._bb_prediction is None


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
