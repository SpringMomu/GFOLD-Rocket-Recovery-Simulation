"""Sideways authority of the engines-first landing burn at high dynamic pressure.

Tilting the thrust off the flow gives it a sideways part, and the body (with
the grid fins as the attitude loop trims them) answers with a lift against
the tilt.  High up the lift wins or cancels it: a tilt towards the pad there
does not move the booster towards it.  The guidance has to model that the
same way everywhere: the trimmed-lift table, the plan's lift linearisation
and the coast's prediction of where the burn ends (the glide's aim point).
"""

import math

import numpy as np
import pytest

from lander3d.autopilot import SECANT_TILT_DEG
from lander3d.planner import Plan3D
from lander3d.sim import Sim3D
from lander3d.vehicle import atmosphere


def _ap():
    sim = Sim3D(cfd=False)
    sim.vehicle.reset(position=(0.0, 0.0, 1500.0), velocity=(0.0, 0.0, -180.0), on_pad=False)
    sim.autopilot.elapsed = 10.0
    return sim.autopilot, sim.vehicle


def _net(ap, veh, q, sigma, deg, rho):
    """Sideways acceleration along a tilt of ``deg`` off a vertical flow: thrust + trimmed lift."""
    sp = math.sqrt(2.0 * q / rho)
    t = math.radians(deg)
    axis = np.array([math.sin(t), 0.0, math.cos(t)])
    f = ap.aero.quick_force(np.array([0.0, 0.0, -sp]), axis, rho, 320.0, 0.0, trim=ap._lift_trim(veh))
    return sigma * math.sin(t) + f[0] / veh.mass


def test_trimmed_lift_curve_has_no_kink_at_the_flow():
    """The fins' share of the lift changes smoothly off the flow: the
    trimmed / untrimmed ratio is about the same over the first degrees."""
    ap, veh = _ap()
    rho = atmosphere(1000.0)[0]
    sp = math.sqrt(2.0 * 15000.0 / rho)
    ratios = []
    for d in (1.0, 2.0, 4.0):
        t = math.radians(d)
        axis = np.array([math.sin(t), 0.0, math.cos(t)])
        vr = np.array([0.0, 0.0, -sp])
        f_t = ap.aero.quick_force(vr, axis, rho, 320.0, 0.0, trim=ap._lift_trim(veh))
        f_0 = ap.aero.quick_force(vr, axis, rho, 320.0, 0.0)
        ratios.append(f_t[0] / f_0[0])
    assert max(ratios) - min(ratios) < 0.08, ratios


def test_burn_authority_vanishes_where_the_lift_beats_the_thrust():
    ap, veh = _ap()
    auth = ap._burn_authority(veh)
    m = veh.mass
    rho = atmosphere(1000.0)[0]
    # high up the drag does most of the braking: little thrust, a lot of lift
    assert auth(20000.0, 5.0, m) < 0.3
    assert all(_net(ap, veh, 20000.0, 5.0, d, rho) <= 0.05 for d in (2, 5, 8))
    # low down the thrust wins: the authority is that of a tilted thrust
    low = auth(500.0, 16.0, m)
    assert low > 3.0
    assert auth(500.0, 16.0, m) > auth(6000.0, 14.0, m) > auth(20000.0, 5.0, m)


def test_burn_end_drifts_when_the_burn_cannot_steer():
    """In a wind the burn would have to lean into it to stay put; high up
    leaning gives no sideways force (the lift answers the tilt), so the
    booster drifts with the wind until the air has thinned -- the end of
    the burn lies downwind, and the glide has to aim for that.  A burn that
    could steer all the way would end where it was lit."""
    ap, veh = _ap()
    ap.aoa_refresh = 1e9
    ap._wind_ref = (np.array([8.0, 0.0, 0.0]), 10.0)
    ap.wind_shear_est = 1.0
    p = np.array([0.0, 0.0, 1500.0])
    v = np.array([float(ap._wind(1500.0)[0]), 0.0, -185.0])      # drifting with the wind
    m = veh.mass
    miss, t_br = ap._burn_miss(veh, p, v, m, np.zeros(3))
    real = ap._burn_authority
    ap._burn_authority = lambda veh_: (lambda q, sigma, mm: math.inf)
    try:
        steer, _ = ap._burn_miss(veh, p, v, m, np.zeros(3))
    finally:
        ap._burn_authority = real
    assert t_br > 5.0
    assert abs(steer[0] - 0.5 * v[0] * t_br) < 0.05 * 0.5 * v[0] * t_br     # braked steadily from ignition on
    assert miss[0] > steer[0] + 3.0, (miss, steer)         # carried downwind first


def _reference_plan(ap, veh, sp, tilt_deg, sigma, n=20, t_f=8.0):
    """A plan to linearise about: falling at ``sp`` through still air, the
    thrust ``sigma`` tilted ``tilt_deg`` off the flow towards +x."""
    t = math.radians(tilt_deg)
    u = np.tile(sigma * np.array([math.sin(t), 0.0, math.cos(t)]), (n + 1, 1))
    r = np.tile(np.array([0.0, 0.0, float(veh.pos[2])]), (n + 1, 1))
    v = np.tile(np.array([0.0, 0.0, -sp]), (n + 1, 1))
    return Plan3D(t_f=t_f, dt=t_f / n, r=r, v=v, z=np.full(n + 1, math.log(veh.mass)), u=u,
                  sigma=np.full(n + 1, sigma), drag=np.zeros((n + 1, 3)), status="Solved", objective=0.0,
                  terminal_position_error=0.0, terminal_velocity_error=0.0, fuel_violation=0.0,
                  created_at=ap.elapsed)


def test_plan_lift_linearisation_follows_the_chord_along_the_tilt():
    """About a reference tilted off the flow, the plan's net sideways gain
    along the tilt has the sign of the net force really produced there
    (thrust plus lift), not of the local slope -- and the model is exact at
    the reference and along the flow."""
    ap, veh = _ap()
    ap.burn_locked = True                   # (the plan in hand is the reference once the burn is on)
    ap._wind_ref = (np.zeros(3), 10.0)
    ap.wind_shear_est = 1.0
    sp, sigma, m = 120.0, 14.0, veh.mass
    trim = ap._lift_trim(veh)
    for deg in (6.0, 12.0, 18.0):
        assert deg > SECANT_TILT_DEG
        plan = ap.plan = _reference_plan(ap, veh, sp, deg, sigma)
        times = np.array([2.0, 4.0])
        d0, G, _, _, _ = ap._drag_model(veh, np.zeros(3), 8.0)(times)
        t = math.radians(deg)
        u_ref = sigma * np.array([math.sin(t), 0.0, math.cos(t)])
        d_ref = u_ref / sigma
        for k in range(2):
            pos, vel, _ = plan.sample(plan.t_f * times[k] / 5.0)     # (the plan stretched over the horizon)
            rho, _, _, a_s = atmosphere(float(pos[2]))
            base = -vel / np.linalg.norm(vel)
            axis = base + (d_ref - base * float(np.dot(d_ref, base)))  # the reference attitude of the model
            axis /= np.linalg.norm(axis)
            f_al = ap.aero.quick_force(vel, base, rho, a_s, 0.0, trim=trim) / m
            f0 = ap.aero.quick_force(vel, axis, rho, a_s, 0.0, trim=trim) / m
            net_ref = sigma * math.sin(t) + (f0 - f_al)[0]
            assert np.allclose(d0[k] + G[k] @ u_ref, f0, atol=1e-6)           # exact at the reference
            assert np.allclose(d0[k], f_al, atol=1e-6)                         # ... and along the flow
            gain = 1.0 + float(G[k][0, 0])                                     # net d(sideways)/d(u_x)
            assert np.sign(gain) == np.sign(net_ref), (deg, gain, net_ref)


def test_axis_is_aimed_so_that_the_trimmed_thrust_points_as_commanded():
    """The gimbal holds the aerodynamic moment, so the thrust leans off the
    axis by its trim angle; the axis is aimed the other way by that much
    (at low dynamic pressure, where a tilt moves the booster one for one),
    not at all on the ground, and never beyond the gimbal's range."""
    sim = Sim3D(cfd=False)
    sim.vehicle.reset(position=(0.0, 0.0, 60.0), velocity=(0.0, 0.0, -2.0), on_pad=False)
    ap, veh = sim.autopilot, sim.vehicle
    ap._dt = 1.0 / 240.0
    veh.throttle = 0.0                              # (no new measurement: the offset is the one set here)
    d = np.array([0.0, 0.0, 1.0])
    ap._thr_off = np.array([math.radians(5.0), 0.0, 0.0])      # thrust 5 deg off the axis, towards +x
    aim = ap._thrust_aim(veh, d, 9.8)
    assert math.degrees(math.atan2(-aim[0], aim[2])) == pytest.approx(5.0, abs=0.3)
    ap._thr_off = np.array([math.radians(30.0), 0.0, 0.0])
    aim = ap._thrust_aim(veh, d, 9.8)
    assert math.degrees(math.atan2(-aim[0], aim[2])) <= math.degrees(veh.spec.gimbal_limit) + 0.1
    veh.pos[2] = veh.rest_height + 0.2                         # touching down: upright
    ap._thr_off = np.array([math.radians(5.0), 0.0, 0.0])
    assert np.allclose(ap._thrust_aim(veh, d, 9.8), d)
    ap.thrust_aim = False
    veh.pos[2] = 60.0
    assert np.allclose(ap._thrust_aim(veh, d, 9.8), d)
