"""Closed-loop boost-back measurements, including the landing handover.

    python -m tools.boostback_diagnostic --out tools/out/boostback.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np

from lander3d import sim as sim_module
from lander3d.regression import passed
from lander3d.sim import PHYSICS_DT, Sim3D


def run(case):
    sim = Sim3D(cfd=False, scenario=case.get("scenario", 3), wind_on=case.get("wind", 6.0) > 0)
    if "pos" in case:
        sim.vehicle.reset(position=case["pos"], velocity=case.get("vel", (0, 0, 0)),
                          propellant=5500, on_pad=False)
        sim.vehicle.pos[2] += sim.vehicle.rest_height
        sim.autopilot.reset()
    if case.get("wind", 6.0) != 6.0 or "from" in case:
        sim.apply_action("windcfg:" + json.dumps({"from": [case.get("from", 270)] * 2,
                                                "speed": [case.get("wind", 6)] * 2, "shear": 1}))
    vehicle, ap = sim.vehicle, sim.autopilot
    prop0 = vehicle.propellant
    transitions = []
    prev = None
    hover, rate, direction_step = 0.0, 0.0, 0.0
    prev_axis = None
    started = time.perf_counter()
    while sim.time < 260 and vehicle.state == "FLYING":
        sim.step(PHYSICS_DT)
        if ap.phase != prev:
            transitions.append({"phase": ap.phase, "time": round(sim.time, 3),
                                "height": round(ap._height(vehicle), 2),
                                "speed": vehicle.vel.tolist(), "xy": vehicle.pos[:2].tolist(),
                                "propellant": round(vehicle.propellant, 1)})
            prev = ap.phase
        if ap.phase == "BOOSTBACK":
            if prev_axis is not None:
                direction_step = max(direction_step, math.degrees(math.acos(np.clip(prev_axis @ ap.cmd_axis, -1, 1))))
            prev_axis = ap.cmd_axis.copy()
        if (vehicle.touchdown is None and ap.phase == "LANDING BURN"
                and ap._height(vehicle) - ap.gate_height > 1.5 and np.linalg.norm(vehicle.vel) < 1):
            hover += PHYSICS_DT
        if vehicle.touchdown is None and ap.phase in ("LANDING BURN",) and ap._height(vehicle) < 8:
            rate = max(rate, math.degrees(float(np.abs(vehicle.omega[:2]).max())))
    td = vehicle.touchdown or {}
    return {"case": case, "success": passed(vehicle, sim.tower is not None), "state": vehicle.state,
            "reason": vehicle.crash_reason, "time": round(sim.time, 2), "touchdown": td,
            "propellant_used": round(prop0 - vehicle.propellant, 1), "hover": round(hover, 3),
            "low_rate": rate, "max_boost_command_step_deg": direction_step,
            "wall_seconds": round(time.perf_counter() - started, 2), "transitions": transitions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tools/out/boostback.json"))
    parser.add_argument("--baseline", type=Path, help="saved autopilot source, for a paired comparison")
    parser.add_argument("--case", action="append", help="JSON case; repeat to run multiple cases")
    args = parser.parse_args()
    if args.baseline:
        spec = importlib.util.spec_from_file_location("lander3d._boostback_baseline", args.baseline)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sim_module.Autopilot3D = module.Autopilot3D
    cases = [json.loads(c) for c in args.case] if args.case else [
        {"scenario": 3, "wind": 0}, {"scenario": 3},
        {"scenario": 3, "wind": 15, "from": 180},
        {"scenario": 7}, {"pos": [2000, 0, 600]},
        {"pos": [150, 100, 2000], "vel": [0, 0, -30]},
    ]
    results = []
    for case in cases:
        result = run(case)
        results.append(result)
        print(json.dumps(result), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return int(not all(r["success"] and r["hover"] <= 2 and r["low_rate"] <= 12
                       and r["touchdown"].get("tilt", 99) <= 1.5 for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
