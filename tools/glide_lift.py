"""How much of the quick aerodynamic model's lift the glide really gets
(read-only; changes no control).

    python -m tools.glide_lift

An engine-off booster, grid fins out, falls engines-first while the attitude
loop (the autopilot's own: attitude_control, same gains) holds a fixed angle
of attack alpha off the flow in a fixed horizontal direction.  Once settled,
the measured aerodynamic force across the flow is compared with what
``aero.quick_force`` predicts for the commanded attitude -- the ratio is what
the glide planner may count on (the fins deflect to hold the attitude, and
their force takes part of the body lift back).
"""

from __future__ import annotations

import math

import numpy as np

from lander3d.attitude import attitude_control
from lander3d.autopilot import Autopilot3D
from lander3d.sim import PHYSICS_DT, Sim3D
from lander3d.vehicle import atmosphere


def measure(height: float, speed: float, alpha_deg: float, settle: float = 6.0, window: float = 2.0):
    sim = Sim3D(cfd=False)
    sim.wind.configure((0.0, 0.0), (0.0, 0.0), 1.0)
    sim.vehicle.reset(position=(0.0, 0.0, height), velocity=(0.0, 0.0, -speed), propellant=3000.0, on_pad=False)
    sim.autopilot_on = False
    veh = sim.vehicle
    veh.controls.fins_deploy = True
    veh.fins_deploy = 1.0
    ap = Autopilot3D
    e = np.array([1.0, 0.0, 0.0])
    a = math.radians(alpha_deg)
    ratios, lifts = [], []
    t = 0.0
    while t < settle + window and veh.state == "FLYING" and veh.pos[2] > 200.0:
        vr = veh.vel - sim.wind.at(veh.pos, veh.time)
        u = vr / np.linalg.norm(vr)
        base = -u
        side = e - u * float(np.dot(e, u))
        side /= np.linalg.norm(side)
        axis = math.cos(a) * base + math.sin(a) * side
        veh.controls.throttle = 0.0
        attitude_control(veh, axis, ap.k_angle, ap.k_rate, ap.max_rate, True, hold_roll=True)
        veh.step(PHYSICS_DT, sim.aero, sim.wind)     # the vehicle alone (no manual-control layer)
        t += PHYSICS_DT
        if t > settle:
            m = veh.mass
            f = np.asarray(veh.last["aero"], float) / m
            perp = f - u * float(np.dot(f, u))
            rho, _, _, a_s = atmosphere(max(0.0, float(veh.pos[2])))
            fq = sim.autopilot.aero.quick_force(vr, axis, rho, a_s, veh.legs) / m
            f0 = sim.autopilot.aero.quick_force(vr, base, rho, a_s, veh.legs) / m
            dq = fq - f0
            perp_q = dq - u * float(np.dot(dq, u))
            if np.linalg.norm(perp_q) > 1e-3:
                ratios.append(float(np.dot(perp, perp_q) / np.dot(perp_q, perp_q)))
                lifts.append(float(np.linalg.norm(perp)))
    q = 0.5 * atmosphere(height)[0] * speed ** 2
    return q, (float(np.mean(ratios)) if ratios else float("nan")), (float(np.mean(lifts)) if lifts else float("nan"))


def main() -> int:
    print(f"{'h':>6s} {'speed':>6s} {'q Pa':>7s} {'alpha':>6s} {'lift m/s2':>10s} {'share of quick model':>21s}")
    for h, sp in ((6000.0, 200.0), (3000.0, 120.0), (2000.0, 80.0), (1500.0, 50.0)):
        for alpha in (5.0, 10.0, 15.0):
            q, ratio, lift = measure(h, sp, alpha)
            print(f"{h:6.0f} {sp:6.0f} {q:7.0f} {alpha:6.0f} {lift:10.2f} {ratio:21.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
