"""Aerodynamic database build-up: retro-propulsion, grid-fin wake shadowing,
jet damping and the CFD anchoring of the body coefficients."""

import math
import time

import numpy as np

import lander3d.aero as aero_mod
from lander3d.aero import AeroModel, ExternalCFD, grid_fin_forces, srp_axial_factor
from lander3d.vehicle import Vehicle, atmosphere


def _falling(speed=150.0, alt=1500.0, tilt_deg=0.0):
    veh = Vehicle()
    t = math.radians(tilt_deg)
    q = np.array([math.cos(t / 2), 0.0, math.sin(t / 2), 0.0])     # about body y
    veh.reset(position=(0.0, 0.0, alt), velocity=(0.0, 0.0, -speed), attitude=q, on_pad=False)
    return veh


def test_the_plume_shields_an_engines_first_booster():
    assert srp_axial_factor(0.0) == 1.0
    assert srp_axial_factor(0.2) > 0.85                       # small C_T: the drag is kept
    assert srp_axial_factor(3.0) < 0.35                       # the plume stands off the base
    veh = _falling()
    aero = AeroModel(veh.spec)
    rho, _, _, a = atmosphere(veh.pos[2])
    v_rel = veh.vel.copy()
    veh.last["thrust_mag"] = 0.0
    f_off, _, _ = aero(veh, v_rel, rho, a, veh.com_world())
    veh.last["thrust_mag"] = veh.spec.thrust_sl
    f_on, _, info = aero(veh, v_rel, rho, a, veh.com_world())
    assert info["ct"] > 1.0
    assert 0.0 < f_on[2] < 0.5 * f_off[2]
    # the guidance's prediction sees the same plume
    quick = aero.quick_force(v_rel, veh.axis, rho, a, 0.0, thrust=veh.spec.thrust_sl)
    many = aero.quick_force_many(v_rel[None], veh.axis[None], rho, a, 0.0, thrust=veh.spec.thrust_sl)[0]
    assert np.allclose(quick, many, rtol=1e-9, atol=1e-6)
    assert quick[2] < 0.5 * aero.quick_force(v_rel, veh.axis, rho, a, 0.0)[2]


def test_leeward_grid_fins_sit_in_the_body_wake(monkeypatch):
    veh = _falling(tilt_deg=25.0)
    s = veh.spec
    rho, _, _, a = atmosphere(veh.pos[2])
    _, zc, _, _ = veh.mass_properties()

    def fins(defl):
        return grid_fin_forces(s, veh.R, zc, np.zeros(3), veh.vel, rho, a, np.asarray(defl, float), 1.0)[2]

    # crossflow along body x: the 0 / 180 deg panels are windward / leeward
    v_b = veh.R.T @ veh.vel
    lee = 0 if -v_b[0] > 0 else 2
    push = np.zeros(4)
    push[lee] = 0.3
    lee_gain = np.linalg.norm(fins(push) - fins(np.zeros(4)))
    push = np.roll(push, 2)
    wind_gain = np.linalg.norm(fins(push) - fins(np.zeros(4)))
    assert lee_gain < 0.8 * wind_gain                  # a deflected leeward panel does less
    shadowed = np.linalg.norm(fins(np.zeros(4)))
    monkeypatch.setattr(aero_mod, "FIN_SHADOW", 0.0)
    assert shadowed < np.linalg.norm(fins(np.zeros(4)))


def test_the_exhaust_damps_pitch_rates():
    def rate_after(throttle):
        veh = _falling(speed=0.5, alt=3000.0)
        veh.omega[:] = [0.2, 0.0, 0.0]
        veh.controls.throttle = throttle
        veh.throttle = throttle
        for _ in range(240):
            veh.step(1.0 / 240.0)
        return veh.omega[0]

    assert rate_after(1.0) < rate_after(0.0) - 1e-3


def test_cfd_anchors_the_body_coefficients_within_bounds():
    veh = _falling(tilt_deg=20.0)
    aero = AeroModel(veh.spec)
    rho, _, _, a = atmosphere(veh.pos[2])
    veh.last["thrust_mag"] = 0.0
    f_model, _, _ = aero(veh, veh.vel, rho, a, veh.com_world())
    aero.cfd = ExternalCFD()
    aero.cfd_forces = True
    axis = veh.axis
    # a CFD sample with a huge normal force and the centre of pressure at the top
    side = np.cross(axis, [0.0, 1.0, 0.0])
    aero.cfd.update(list(-0.8 * axis + 40.0 * side), zcp=17.0)
    time.sleep(0.02)
    f_cfd, m_cfd, info = aero(veh, veh.vel, rho, a, veh.com_world())
    assert info["source"] == "CFD"
    k_a, k_n, dz = aero._anchor[1:]
    assert 0.5 <= k_a <= 2.0 and 0.5 <= k_n <= 2.0
    assert abs(dz) <= 0.15 * veh.spec.length + 1e-9
    assert np.all(np.isfinite(f_cfd)) and np.all(np.isfinite(m_cfd))
    # stale data falls back to the model
    aero.cfd.stamp -= 5.0
    f_back, _, info = aero(veh, veh.vel, rho, a, veh.com_world())
    assert info["source"] == "MODEL"
    assert np.allclose(f_back, f_model)
