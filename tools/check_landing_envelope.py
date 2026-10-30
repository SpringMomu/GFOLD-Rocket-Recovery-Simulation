"""Terminal-flight envelope check for the 3D scenarios (pad landings and tower catches).

    python -m tools.check_landing_envelope [--out DIR]

Beyond the regression limits this checks that the booster never hovers (sink
rate stays >= 0.8 m/s above 1.5 m) and never rocks near the ground (pitch/yaw
rate <= 12 deg/s below 8 m, touchdown tilt <= 1.5 deg); a scenario may set a
lower "min_sink" (flying over the catch tower levels off above it).  Heights are above
the end of the burn: the pad, or for a catch the collar on the arms.  Writes
``landing-envelope.json`` to DIR (default ``tools/out``); exit code 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lander3d.regression import passed
from lander3d.sim import PHYSICS_DT, SCENARIOS, Sim3D

MIN_SINK = 0.8          # m/s, while higher than HOVER_HEIGHT
HOVER_HEIGHT = 1.5      # m
MAX_LOW_RATE = 12.0     # deg/s, pitch/yaw rate below LOW_HEIGHT
LOW_HEIGHT = 8.0        # m
MAX_TOUCHDOWN_TILT = 1.5


def check(index: int) -> dict:
    sim = Sim3D(cfd=False)
    sim.load_scenario(index)
    vehicle = sim.vehicle
    min_sink = math.inf
    sink_limit = SCENARIOS[index].get("min_sink", MIN_SINK)
    max_rate = 0.0
    while sim.time < 260 and vehicle.state == "FLYING":
        sim.step(PHYSICS_DT)
        if vehicle.touchdown is None and sim.autopilot.phase == "LANDING BURN":
            height = sim.autopilot._height(vehicle)
            if height > HOVER_HEIGHT:
                min_sink = min(min_sink, -float(vehicle.vel[2]))
            if height < LOW_HEIGHT:
                rate = math.degrees(max(abs(float(vehicle.omega[0])), abs(float(vehicle.omega[1]))))
                max_rate = max(max_rate, rate)
    td = vehicle.touchdown or {}
    ok = (
        passed(vehicle, sim.tower is not None)
        and td.get("tilt", 99) <= MAX_TOUCHDOWN_TILT
        and min_sink >= sink_limit
        and max_rate <= MAX_LOW_RATE
    )
    return {"scenario": sim.scenario_name, "pass": ok, "minimum_sink_above_1_5m": min_sink,
            "maximum_pitch_yaw_rate_below_8m": max_rate, "touchdown": td}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("out"))
    args = parser.parse_args()

    results = []
    for index in range(len(SCENARIOS)):
        results.append(check(index))
        print(json.dumps(results[-1]), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "landing-envelope.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(row["pass"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
