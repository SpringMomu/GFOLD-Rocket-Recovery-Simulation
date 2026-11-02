"""Check the browser's cooperative boost-back path without Python threads.

    python -m tools.check_realtime_3d --out tools/out/realtime/cooperative.json

Each callback advances 10 ms of flight and gives numerical guidance a bounded
work slice, as advance_realtime does in the browser. Checks landing/catch,
hover, low-altitude rates, and the latched flight clock; records callback cost.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from lander3d.regression import passed
from lander3d.sim import PHYSICS_DT, SCENARIOS, Sim3D
from tools.check_landing_envelope import MAX_HOVER, MAX_LOW_RATE, MAX_TOUCHDOWN_TILT


def check(index):
    sim = Sim3D(cfd=False, scenario=index)
    sim.autopilot._bb_cooperative = True
    metrics = {"hover": 0.0, "low_rate": 0.0}
    step = sim.step

    def measured_step(dt):
        step(dt)
        v, ap = sim.vehicle, sim.autopilot
        if (v.touchdown is None and ap.phase == "LANDING BURN" and ap._height(v) - ap.gate_height > 1.5
                and np.linalg.norm(v.vel) < 1.0):
            metrics["hover"] += dt
        if v.touchdown is None and ap.phase in ("LANDING BURN", "CATCH HOLD") and ap._height(v) < 8.0:
            metrics["low_rate"] = max(metrics["low_rate"], math.degrees(float(np.abs(v.omega[:2]).max())))

    sim.step = measured_step
    costs = []
    while sim.time < 260.0 and sim.vehicle.state == "FLYING":
        begin = time.perf_counter()
        sim.advance_realtime(0.01)
        if sim.autopilot.phase == "BOOSTBACK":
            costs.append(1000.0 * (time.perf_counter() - begin))
    td = sim.vehicle.touchdown or {}
    flight_time, physics_time = sim.flight_time, sim.time
    for _ in range(100):
        sim.step(PHYSICS_DT)
    clock_ok = sim.flight_time == flight_time and sim.time > physics_time
    ok = (passed(sim.vehicle, sim.tower is not None) and clock_ok and metrics["hover"] <= MAX_HOVER
          and metrics["low_rate"] <= MAX_LOW_RATE and td.get("tilt", 99.0) <= MAX_TOUCHDOWN_TILT)
    return {"scenario": index, "name": sim.scenario_name, "pass": ok, "state": sim.vehicle.state,
            "time": physics_time, "touchdown": td, **metrics, "clock_stopped": clock_ok,
            "boost_update_max_ms": max(costs, default=0.0),
            "boost_update_p95_ms": float(np.percentile(costs, 95)) if costs else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tools/out/realtime/cooperative.json"))
    args = parser.parse_args()
    rows = []
    for index in range(len(SCENARIOS)):
        rows.append(check(index))
        print(json.dumps(rows[-1]), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return int(not all(row["pass"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
