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
