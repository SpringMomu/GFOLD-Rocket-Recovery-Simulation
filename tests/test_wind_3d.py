"""3D wind model and the front end's wind settings (action "windcfg:<json>")."""

import math

import numpy as np

from lander3d.sim import Sim3D
from lander3d.vehicle import Wind


def _log_law_wind(pos, t, speed=6.0, toward_deg=90.0, gust=1.8):
    """The wind before the ranges / shear setting existed."""
    z = max(0.5, float(pos[2]))
    profile = speed * min(2.6, max(0.25, math.log(z / 0.05) / math.log(10.0 / 0.05)))
    g = gust * (0.55 * math.sin(0.31 * t + 0.002 * pos[0]) + 0.3 * math.sin(0.97 * t + 1.7) + 0.15 * math.sin(2.3 * t + 0.4))
    s = profile + g
    a = math.radians(toward_deg)
    return np.array([math.sin(a) * s, math.cos(a) * s + gust * 0.35 * math.sin(0.53 * t + 0.8), 0.25 * g])


def test_default_wind_is_unchanged():
    w = Wind()
    for t in np.linspace(0.0, 200.0, 41):
        for z in (0.3, 10.0, 300.0, 9000.0):
            p = np.array([150.0, -80.0, z])
            assert np.array_equal(w.at(p, t), _log_law_wind(p, t))


def test_wind_stays_inside_its_ranges():
    w = Wind()
    w.configure(from_deg=(300.0, 60.0), speed=(9.0, 2.0), shear=1.0)     # wraps through north
    assert w.config() == {"from": [300.0, 60.0], "speed": [2.0, 9.0], "shear": 1.0}
    frm, spd = [], []
    for t in np.linspace(0.0, 900.0, 3000):
        toward, s10 = w.mean_at(t)
        frm.append((toward + 180.0) % 360.0)
        spd.append(s10)
    assert all(f >= 300.0 - 1e-9 or f <= 60.0 + 1e-9 for f in frm)
    assert max(f for f in frm if f >= 300.0) > 350.0 and max(f for f in frm if f <= 60.0) > 50.0   # covers the arc
    assert 2.0 - 1e-9 <= min(spd) < 2.5 and 8.5 < max(spd) <= 9.0 + 1e-9


def test_no_shear_means_the_same_wind_at_every_height():
    w = Wind(gust=0.0)
    w.configure(shear=0.0)
    assert np.allclose(w.at(np.array([0.0, 0.0, 10.0]), 7.0), w.at(np.array([0.0, 0.0, 4000.0]), 7.0))
    w.configure(shear=2.0)
    hi, lo = (float(np.linalg.norm(w.at(np.array([0.0, 0.0, z]), 7.0))) for z in (1000.0, 10.0))
    assert abs(hi / lo - (1.0 + 2.0 * (math.log(1000.0 / 0.05) / math.log(10.0 / 0.05) - 1.0))) < 1e-9


def test_windcfg_action_sets_the_wind_and_ignores_bad_input():
    sim = Sim3D(cfd=False)
    sim.apply_action('windcfg:{"from": [200, 250], "speed": [4, 40], "shear": 9}')
    good = {"from": [200.0, 250.0], "speed": [4.0, Wind.SPEED_MAX], "shear": Wind.SHEAR_MAX}
    assert sim.wind.config() == good
    for bad in ("nonsense", "null", "[1, 2]", '{"from": [1]}', '{"from": 5}', '{"speed": ["a", 2]}',
                '{"shear": NaN}', '{"from": [1, Infinity]}'):
        sim.apply_action("windcfg:" + bad)
        assert sim.wind.config() == good


def test_autopilot_fits_the_shear_from_the_winds_it_has_measured():
    from lander3d.autopilot import Autopilot3D

    ap = Autopilot3D()
    w = Wind(gust=0.0)
    w.configure(from_deg=(270.0, 270.0), speed=(8.0, 8.0), shear=2.0)

    class V:
        altitude = 0.0

    veh = V()
    # samples over a wide height range (as measured near the ground): the
    # shear is found (nearby strengths give similar shapes, so roughly)
    for shear, lo, hi in ((2.0, 1.4, 2.2), (0.0, 0.0, 0.4)):
        ap.reset()
        w.configure(shear=shear)
        for i, z in enumerate(np.linspace(300.0, 2.0, 120)):
            ap.elapsed = 0.25 * i
            veh.altitude = z
            ap._update_wind_profile(veh, w.at(np.array([0.0, 0.0, z]), 0.0))
        assert lo <= ap.wind_shear_est <= hi
    w.configure(shear=2.0)
    # little height spread: the prior (standard log law) is kept
    ap.reset()
    for i in range(40):
        ap.elapsed = 0.25 * i
        veh.altitude = 100.0 + 0.1 * i
        ap._update_wind_profile(veh, w.at(np.array([0.0, 0.0, veh.altitude]), 0.0))
    assert ap.wind_shear_est == 1.0


def test_tilt_limit_is_centred_on_the_wind_trim():
    from lander3d.autopilot import Autopilot3D
    from lander3d.vehicle import G0, Vehicle

    ap = Autopilot3D()
    veh = Vehicle()
    veh.reset(position=(0.0, 0.0, 50.0), on_pad=False)
    lim = math.radians(2.0)
    trim = np.array([-1.0, 0.0])                 # lean ~5.8 deg west, into a west wind
    # a 5 deg westward command: inside the cone about the trim, kept
    ap._fly_accel(veh, np.array([-math.tan(math.radians(5.0)) * G0, 0.0, G0]), lim, center=trim)
    assert abs(math.degrees(math.atan2(-ap.cmd_accel[0], ap.cmd_accel[2])) - 5.0) < 1e-6
    # upright-centred, the same command is clipped to 2 deg
    ap._dir_prev = None
    ap._fly_accel(veh, np.array([-math.tan(math.radians(5.0)) * G0, 0.0, G0]), lim)
    assert abs(math.degrees(math.atan2(-ap.cmd_accel[0], ap.cmd_accel[2])) - 2.0) < 1e-6


def test_planner_tilt_cone_leans_into_the_wind():
    from lander3d.planner import PlanConfig3D, Planner3D

    P = Planner3D()
    if not P.available:
        return
    # hold position above the pad against a steady 1 m/s^2 eastward wind drag
    cfg = PlanConfig3D(thrust_min=0.2 * 360e3, thrust_max=0.85 * 360e3, isp=282.0, dry_mass=8150.0,
                       sigma_now=9.8, u_now=(-1.0, 0.0, 9.8), engine_on=True, max_tilt_deg=2.0, touchdown_speed=1.2,
                       sink_gain=0.5, drag_model=lambda t: np.tile([1.0, 0.0, 0.0], (len(t), 1)),
                       tilt_trim=(-1.0, 0.0), trim_fade=(0.0, 0.1))
    plan = P.solve(np.array([0.0, 0.0, 6.0]), np.array([0.0, 0.0, -1.2]), 11000.0, 4.0, cfg)
    assert plan is not None
    lean = np.degrees(np.arctan2(-plan.u[:-1, 0], plan.u[:-1, 2]))
    assert lean.min() > 3.0                        # leans west to hold against the drag (2 deg cone about ~5.8)
    assert plan.terminal_position_error < 0.5


def test_the_final_time_is_the_optimisers_and_it_does_not_loft():
    from lander3d.planner import PlanConfig3D, Planner3D

    P = Planner3D()
    if not P.available:
        return
    # low and 150 m off: the cost J = propellant + time picks the final time
    # (a longer burn costs ~g per second of gravity loss), and the cheapest
    # route does not climb -- no envelope makes it do anything else
    r0, v0 = np.array([-150.0, 0.0, 87.0]), np.array([0.0, 0.0, -5.0])
    cfg = PlanConfig3D(thrust_min=0.2 * 360e3, thrust_max=0.85 * 360e3, isp=282.0, dry_mass=8150.0,
                       sigma_now=3.6, u_now=(0.0, 0.0, 3.6), engine_on=True, max_tilt_deg=65.0, touchdown_speed=1.2,
                       sink_gain=0.5, ground_alt=0.0, time_weight=1.0)
    best = P.search(r0, v0, 12150.0, 8.0, 40.0, cfg)
    assert best.reaches_pad and 18.0 < best.t_f < 32.0
    rushed, slow = (P.solve(r0, v0, 12150.0, t, cfg) for t in (12.0, 40.0))
    assert rushed.objective > best.objective + 100.0 and slow.objective > best.objective
    assert float(best.r[:, 2].max()) < 88.0          # no loft: nothing in the cost asks for one


def test_aoa_limit_does_not_push_downwind_while_rising():
    from lander3d.autopilot import Autopilot3D
    from lander3d.vehicle import G0, Vehicle

    veh = Vehicle()
    veh.reset(position=(0.0, 0.0, 120.0), velocity=(0.0, 0.0, 12.0), on_pad=False)
    veh.last["q"] = 450.0
    wind = np.array([0.0, 22.0, 0.0])                    # from the south, the booster climbing
    accel = np.array([0.0, -2.0, G0 + 2.0])              # lean into the wind
    out = Autopilot3D._aoa_limit(veh, accel, wind, math.radians(75.0))
    assert out[1] <= 1e-9                                 # never turned downwind (+y)
    assert abs(out[2] - min(accel[2], G0)) < 1e-9         # and what it gives up does not go into a climb


def test_angle_of_attack_limit_comes_from_the_moment_the_controls_can_hold():
    sim = Sim3D(cfd=False)
    sim.load_scenario(9)
    ap = sim.autopilot
    ap.elapsed = 10.0
    qs, lim = ap._aoa_table(sim.vehicle)
    lim = dict(zip(qs, lim))
    assert lim[150.0] == 180.0 and lim[500.0] == 180.0  # a 30 m/s cross wind: any lean is holdable
    assert lim[20000.0] < 15.0                          # fast flight: close to engines-first only
    assert all(a >= b for a, b in zip(list(lim.values()), list(lim.values())[1:]))


def test_low_far_catch_burn_time_is_searched_not_predicted():
    from lander3d.sim import PHYSICS_DT

    sim = Sim3D(cfd=False)
    if not sim.autopilot.planner.available:
        return
    sim.load_scenario(9)                                 # behind the tower, 150 m off, 87 m above the catch point
    ap = sim.autopilot
    while not ap.burn_locked and sim.time < 3.0:         # lights almost at once (low, slow, far off)
        sim.step(PHYSICS_DT)
    assert ap.burn_locked and ap.plan is not None and ap.plan.reaches_pad
    # the vertical-burn estimate alone says ~34 s (locked to it, the plan
    # used to loft to ~250 m); the final time is the one minimising J
    assert ap.touchdown_time - ap.elapsed < 32.0
    assert float(ap.plan.r[:, 2].max()) < 130.0      # (a locked 34 s burn lofted it to 250-310 m)


def test_catch_hold_bias_includes_what_the_model_misses():
    from lander3d.autopilot import Autopilot3D
    from lander3d.vehicle import Vehicle

    ap = Autopilot3D()
    veh = Vehicle()
    veh.reset(position=(0.0, 0.0, 50.0), on_pad=False)
    m = veh.mass
    ap.cmd_accel = np.array([-0.7, 0.0, 9.8])
    # the thrust delivers only part of the commanded lean (gimbal holding the
    # attitude), the wind pushes east at 0.75 m/s^2
    veh.last["thrust"] = np.array([-0.45, 0.0, 9.8]) * m
    veh.last["aero"] = np.array([0.75, 0.0, 0.0]) * m
    for _ in range(2000):
        ap._update_hold_bias(veh, 1.0 / 240.0)
    assert np.allclose(ap._hold_bias, [-1.0, 0.0], atol=1e-3)


def test_the_plan_is_solvable_from_a_state_outside_its_limits():
    from lander3d.planner import PlanConfig3D, Planner3D

    # Near the catch in a strong cross wind the booster can be outside every
    # limit tied to its current state at once: full throttle (beyond the
    # plan's 85 % margin), leaning the wrong way, far past the tilt budget,
    # with the command flown elsewhere.  The plan must still come back (the
    # limits are elastic), and say how far outside they had to be.
    P = Planner3D()
    tmax = 0.85 * 360e3
    cfg = PlanConfig3D(thrust_min=0.2 * 360e3, thrust_max=tmax, isp=282.0, dry_mass=8150.0,
                       sigma_now=360e3 / 11000.0, u_now=(12.0, -6.0, 30.0), u_cmd=(-3.0, 2.0, 12.0), engine_on=True,
                       max_tilt_deg=2.0, touchdown_speed=1.2, sink_gain=0.5, tilt_trim=(-1.5, 0.0), trim_fade=(0.0, 0.1),
                       tilt_budget=((0.0, 3.0, 40.0), (0.6, 5.0, 20.0)), slew_deg=((2.0, 30.0), (8.0, 15.0)))
    plan = P.solve(np.array([2.0, -1.0, 8.0]), np.array([1.5, 0.0, -2.0]), 11000.0, 4.0, cfg)
    assert plan is not None
    assert plan.limit_violation > 0.5 and not plan.reaches_pad
    assert abs(plan.sigma[0] - tmax / 11000.0) < 0.2 * tmax / 11000.0    # starts where the engine can be
