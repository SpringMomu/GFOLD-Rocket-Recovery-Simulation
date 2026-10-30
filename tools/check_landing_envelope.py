"""Terminal-flight envelope check for the 3D scenarios (pad landings and tower catches).

    python -m tools.check_landing_envelope [--out DIR]

Beyond the regression limits this checks that the booster does not hover
(at most 2 s of the landing burn spent above 1.5 m at a speed below 1 m/s --
levelling off, climbing over an obstacle or flying sideways is fine, standing
still in the air is not) and never rocks near the ground (pitch/yaw rate
<= 12 deg/s below 8 m, touchdown tilt <= 1.5 deg).  Heights are above the end
of the burn: the pad, or for a catch the collar on the arms (the deliberate
catch hold there is not a landing-burn hover).  Writes
``landing-envelope.json`` to DIR (default ``tools/out``); exit code 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from lander3d.regression import passed
from lander3d.sim import PHYSICS_DT, SCENARIOS, Sim3D

HOVER_SPEED = 1.0       # m/s: slower than this, higher than HOVER_HEIGHT, is hovering
HOVER_HEIGHT = 1.5      # m
MAX_HOVER = 2.0         # s
MAX_LOW_RATE = 12.0     # deg/s, pitch/yaw rate below LOW_HEIGHT
LOW_HEIGHT = 8.0        # m
MAX_TOUCHDOWN_TILT = 1.5


def check(index: int) -> dict:
    sim = Sim3D(cfd=False)
    sim.load_scenario(index)
    vehicle = sim.vehicle
    hover = 0.0
    max_rate = 0.0
    while sim.time < 260 and vehicle.state == "FLYING":
        sim.step(PHYSICS_DT)
        if vehicle.touchdown is None and sim.autopilot.phase == "LANDING BURN":
            # above the end of the burn (a catch ends in a hold just above the arms)
            height = sim.autopilot._height(vehicle) - sim.autopilot.gate_height
            if height > HOVER_HEIGHT and float(np.linalg.norm(vehicle.vel)) < HOVER_SPEED:
                hover += PHYSICS_DT
        low = sim.autopilot._height(vehicle) < LOW_HEIGHT
        if vehicle.touchdown is None and sim.autopilot.phase in ("LANDING BURN", "CATCH HOLD") and low:
            rate = math.degrees(max(abs(float(vehicle.omega[0])), abs(float(vehicle.omega[1]))))
            max_rate = max(max_rate, rate)
    td = vehicle.touchdown or {}
    ok = (
        passed(vehicle, sim.tower is not None)
        and td.get("tilt", 99) <= MAX_TOUCHDOWN_TILT
        and hover <= MAX_HOVER
        and max_rate <= MAX_LOW_RATE
    )
    return {"scenario": sim.scenario_name, "pass": ok, "hover_time": round(hover, 2),
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
