"""How well the boost-back and the glide bring the booster over the pad
(read-only; changes no control).

    python -m tools.recovery_sweep [--scenarios 3,5,7] [--winds default,15@0,15@90] [--jobs 4]

For each scenario and wind it flies the closed loop and reports, at the
landing-burn ignition: the horizontal distance from the gate, where the
predicted burn would end (``_burn_miss``), the height and speed; then the
landing burn's duration and propellant, the total time and the outcome.  A
good boost-back + glide lights the landing burn close above the pad, so the
burn needs only a short, nearly vertical flight.

Winds: ``default`` (the sim's own, westerly 6 m/s), ``calm``, or
``SPEED@FROM`` (steady, m/s at 10 m, from FROM degrees).
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from lander3d.regression import passed
from lander3d.sim import PHYSICS_DT, SCENARIOS, Sim3D

DEFAULT_SCENARIOS = (2, 3, 4, 5, 7, 8)
DEFAULT_WINDS = ("default", "15@0", "15@90", "15@180", "15@270")


def run(args) -> dict:
    index, wind = args
    sim = Sim3D(cfd=False)
    if wind == "calm":
        sim.wind.configure((0.0, 0.0), (0.0, 0.0), 1.0)
    elif wind != "default":
        speed, frm = (float(x) for x in wind.split("@"))
        sim.wind.configure((frm, frm), (speed, speed), 1.0)
    sim.load_scenario(index)
    ap, veh = sim.autopilot, sim.vehicle
    ign = None
    while sim.time < 260.0 and veh.state == "FLYING":
        sim.step(PHYSICS_DT)
        if ign is None and ap.burn_locked:
            r = veh.pos[:2] - ap._gate_xy
            ign = dict(t=round(sim.time, 1), dist=round(float(np.linalg.norm(r)), 1),
                       miss=round(float(np.linalg.norm(ap._burn_miss_now)), 1), h=round(float(ap._height(veh)), 0),
                       speed=round(float(np.linalg.norm(veh.vel)), 0), prop=round(veh.propellant, 0))
    out = dict(scenario=index, name=SCENARIOS[index]["name"], wind=wind, state=veh.state, reason=veh.crash_reason,
               ok=passed(veh, sim.tower is not None), time=round(sim.time, 1), prop_left=round(veh.propellant, 0))
    if ign is not None:
        out.update(ign_t=ign["t"], ign_dist=ign["dist"], ign_miss=ign["miss"], ign_h=ign["h"], ign_speed=ign["speed"],
                   burn_s=round(sim.time - ign["t"], 1), burn_prop=round(ign["prop"] - veh.propellant, 0))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default=",".join(str(i) for i in DEFAULT_SCENARIOS))
    parser.add_argument("--winds", default=",".join(DEFAULT_WINDS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="one JSON line per run instead of the table")
    a = parser.parse_args()
    jobs = [(int(s), w) for s in a.scenarios.split(",") for w in a.winds.split(",")]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        results = list(pool.map(run, jobs))
    if a.json:
        for r in results:
            print(json.dumps(r))
        return 0
    print(f"{'scenario':28s} {'wind':8s} {'result':9s} {'time':>6s} {'ign@':>6s} {'dist':>6s} {'miss':>6s} "
          f"{'h':>6s} {'burn s':>7s} {'burn kg':>8s} {'left':>6s}")
    for r in results:
        f = lambda k, w=6, r=r: f"{r[k]:>{w}}" if k in r else " " * (w - 1) + "-"  # noqa: E731
        res = "ok" if r["ok"] else (r["state"] if not r["reason"] else r["reason"][:9])
        print(f"{r['name'][:28]:28s} {r['wind']:8s} {res:9s} {r['time']:>6} {f('ign_t')} {f('ign_dist')} {f('ign_miss')} "
              f"{f('ign_h')} {f('burn_s', 7)} {f('burn_prop', 8)} {r['prop_left']:>6}")
    good = [r for r in results if "burn_prop" in r]
    if good:
        print(f"mean landing-burn propellant {np.mean([r['burn_prop'] for r in good]):.0f} kg, "
              f"mean ignition miss {np.mean([r['ign_miss'] for r in good]):.0f} m, "
              f"failures {sum(not r['ok'] for r in results)}/{len(results)}")
    return 0 if all(r["ok"] for r in results) and not any(math.isnan(r["time"]) for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
