"""Attitude control of the 3D booster.

Torque sources in priority order: grid fins (effectiveness measured on the
aero model at the current flow), engine gimbal while the engine burns, and
RCS for whatever is left (alone in near-vacuum or at idle thrust).
"""

from __future__ import annotations

import math

import numpy as np

from .aero import grid_fin_forces
from .vehicle import cross3, fin_mix, vnorm


def fin_effectiveness(veh, period=0.05):
    """Grid-fin moment model at the current flow, from the aero model:
    (M0, B) in body axes with M_fins(u) ~= M0 + B u for the (pitch, yaw,
    roll) command u in [-1, 1].  Cached for ``period`` seconds."""
    cache = veh.fin_model_cache
    if cache is not None and veh.time - cache[0] < period and cache[3] == round(veh.fins_deploy, 2):
        return cache[1], cache[2]
    s = veh.spec
    L = veh.last
    v_rel = L.get("v_rel")
    rho = float(L.get("rho", 0.0))
    if v_rel is None or rho <= 0.0 or veh.fins_deploy < 0.05:
        M0, B = np.zeros(3), np.zeros((3, 3))
    else:
        _, zc, _, _ = veh.mass_properties()
        R = veh.R
        a_snd = float(L.get("a", 340.0))

        def mom(u):
            d = np.clip(fin_mix(u) * s.fin_max_defl, -s.fin_max_defl, s.fin_max_defl)
            _, m_w, _ = grid_fin_forces(s, R, zc, veh.omega, v_rel, rho, a_snd, d, veh.fins_deploy)
            return R.T @ m_w

        M0 = mom(np.zeros(3))
        B = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = 0.5
            B[:, j] = (mom(e) - mom(-e)) / 1.0
    veh.fin_model_cache = (veh.time, M0, B, round(veh.fins_deploy, 2))
    return M0, B


FIN_MIN_EFF = 1500.0   # N*m per unit fin command: below this the grid fins stay neutral


def attitude_control(veh, desired_axis, k_angle, k_rate, max_rate, use_rcs, hold_roll=True, roll_rate_cmd=0.0,
                     rate_bias=None):
    """Drive the vehicle axis toward ``desired_axis`` (world); writes controls.

    Torque sources, in order: grid fins (free, strong at high dynamic
    pressure; their effectiveness is measured on the aero model at the
    current flow, so it also covers nose-first flow and Mach effects),
    engine gimbal (TVC) when the engine burns, RCS for the rest (and alone
    in near-vacuum)."""
    s = veh.spec
    c = veh.controls
    R = veh.R
    zd = np.asarray(desired_axis, float)
    zd = zd / (vnorm(zd) + 1e-12)
    bx = R[:, 0]
    xd = bx - np.dot(bx, zd) * zd
    if vnorm(xd) < 1e-6:
        xd = cross3([0.0, 1.0, 0.0], zd)
    xd /= vnorm(xd)
    yd = cross3(zd, xd)
    Rd = np.column_stack([xd, yd, zd])
    E = Rd.T @ R - R.T @ Rd
    e = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])  # body-frame attitude error
    w_cmd = -k_angle * e
    _, zc, ixx, izz = veh.mass_properties()
    M0, B = fin_effectiveness(veh)
    # Braking-distance limit: never command a rate that the actuators
    # (TVC at the current thrust + grid fins + RCS) cannot stop before the
    # target attitude -> no overshoot / wobble at low throttle.
    t_now = veh.throttle * s.thrust_sl
    tq = t_now * max(zc - s.gimbal_z, 0.5) * math.sin(s.gimbal_limit) * 0.7
    tq += 0.8 * s.rcs_torque[0] if (use_rcs or t_now <= 0.04 * s.thrust_sl) else 0.0
    tq += 0.5 * float(min(abs(B[0, 0]), abs(B[1, 1])))
    a_avail = max(0.02, tq / ixx)
    e_mag = float(vnorm(e[:2]))
    rate_cap = min(max_rate, 0.85 * math.sqrt(2.0 * a_avail * e_mag) + 0.004)
    n = float(vnorm(w_cmd[:2]))
    if n > rate_cap:
        w_cmd[:2] *= rate_cap / n
    w_cmd[2] = 0.0 if hold_roll else roll_rate_cmd
    if rate_bias is not None:
        w_cmd = w_cmd + rate_bias
    alpha = k_rate * (w_cmd - veh.omega)
    alpha[2] = 2.0 * (w_cmd[2] - veh.omega[2])
    tau = np.array([ixx, ixx, izz]) * alpha
    # Cancel the known aerodynamic moment of the body (the fins are handled
    # by the allocation below, including their own restoring moment M0).
    info = veh.last.get("aero_info") or {}
    m_body = info.get("moment_body")
    if m_body is None:
        m_body = veh.last.get("aero_moment")
        m_fins_now = info.get("moment_fins")
        if m_body is not None and m_fins_now is not None:
            m_body = np.asarray(m_body) - np.asarray(m_fins_now)
    if m_body is not None:
        tau = tau - R.T @ np.asarray(m_body, float)
    tau = tau - M0
    # Grid fins first (least squares on the measured effectiveness).
    # In near-vacuum (thin air high up) the fins can give only a few hundred
    # N*m: they would flap from stop to stop for nothing.  They take over
    # progressively once they are worth a few percent of the RCS.
    fins = np.zeros(3)
    b_max = float(np.abs(B).max())
    if b_max > FIN_MIN_EFF:
        try:
            fins, *_ = np.linalg.lstsq(B, tau, rcond=1e-3)
        except np.linalg.LinAlgError:
            fins = np.zeros(3)
        fins = np.clip(fins, -1.0, 1.0) * float(np.clip((b_max - FIN_MIN_EFF) / FIN_MIN_EFF, 0.0, 1.0))
        tau = tau - B @ fins
    c.fins = tuple(float(x) for x in fins)
    thrust = veh.throttle * s.thrust_sl
    gx = gy = 0.0
    rcs = np.zeros(3)
    if thrust > 0.04 * s.thrust_sl:
        lever = zc - s.gimbal_z
        fy = tau[0] / lever
        fx = -tau[1] / lever
        sy = float(np.clip(fx / thrust, -0.99, 0.99))
        gy = math.asin(sy)
        sx = float(np.clip(-fy / (thrust * max(0.2, math.cos(gy))), -0.99, 0.99))
        gx = math.asin(sx)
        lim = s.gimbal_limit
        # Whatever TVC cannot deliver, RCS adds.
        gx_c, gy_c = float(np.clip(gx, -lim, lim)), float(np.clip(gy, -lim, lim))
        if abs(gx_c - gx) > 1e-6 or abs(gy_c - gy) > 1e-6:
            rcs[0] = (tau[0] * (1 - gx_c / gx if gx else 0)) / s.rcs_torque[0]
            rcs[1] = (tau[1] * (1 - gy_c / gy if gy else 0)) / s.rcs_torque[1]
        gx, gy = gx_c, gy_c
    if use_rcs or thrust <= 0.04 * s.thrust_sl:
        rcs[0] = tau[0] / s.rcs_torque[0]
        rcs[1] = tau[1] / s.rcs_torque[1]
    rcs[2] = tau[2] / s.rcs_torque[2]
    c.gimbal = (gx, gy)
    c.rcs = tuple(float(v) for v in np.clip(rcs, -1.0, 1.0))
