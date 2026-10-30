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
    cfg = PlanConfig3D(mode="smooth", thrust_min=0.2 * 360e3, thrust_max=0.85 * 360e3, isp=282.0, dry_mass=8150.0,
                       sigma_now=9.8, u_now=(-1.0, 0.0, 9.8), engine_on=True, max_tilt_deg=2.0, touchdown_speed=1.2,
                       sink_gain=0.5, drag_model=lambda t: np.tile([1.0, 0.0, 0.0], (len(t), 1)),
                       tilt_trim=(-1.0, 0.0), trim_fade=(0.0, 0.1))
    plan = P.solve(np.array([0.0, 0.0, 6.0]), np.array([0.0, 0.0, -1.2]), 11000.0, 4.0, cfg)
    assert plan is not None
    lean = np.degrees(np.arctan2(-plan.u[:-1, 0], plan.u[:-1, 2]))
    assert lean.min() > 3.0                        # leans west to hold against the drag (2 deg cone about ~5.8)
    assert plan.terminal_position_error < 0.5
