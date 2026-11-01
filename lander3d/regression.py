"""Headless landing regression for the 3D scenarios (pad landings and tower catches)."""

from __future__ import annotations

import json

from .sim import PHYSICS_DT, SCENARIOS, Sim3D

MAX_TIME = 260.0        # s
MAX_PAD_ERROR = 3.5     # m
MAX_SINK_SPEED = 2.5    # m/s
MAX_SIDE_SPEED = 1.5    # m/s
MAX_TILT_DEG = 5.0
# Tower catches: the collar must come down on the closed arms, centred
# between them, slowly and upright (the arms take it on shock absorbers).
MAX_CATCH_ERROR = 1.0   # m, catch axis at arm height from the catch point
MAX_CATCH_SINK = 2.0    # m/s
MAX_CATCH_SIDE = 1.0    # m/s
MAX_CATCH_TILT = 3.0    # deg


def passed(vehicle, catch: bool) -> bool:
    td = vehicle.touchdown or {}
    if catch:
        return (vehicle.state == "CAUGHT" and td.get("error", 99) <= MAX_CATCH_ERROR
                and abs(td.get("vz", 9)) <= MAX_CATCH_SINK and td.get("vh", 9) <= MAX_CATCH_SIDE
                and td.get("tilt", 90) <= MAX_CATCH_TILT)
    return (vehicle.state == "LANDED" and td.get("error", 99) <= MAX_PAD_ERROR
            and abs(td.get("vz", 9)) <= MAX_SINK_SPEED and td.get("vh", 9) <= MAX_SIDE_SPEED
            and td.get("tilt", 90) <= MAX_TILT_DEG)


def run_scenario(index: int) -> dict:
    sim = Sim3D(cfd=False)
    sim.load_scenario(index)
    vehicle = sim.vehicle
    while sim.time < MAX_TIME and vehicle.state == "FLYING":
        sim.step(PHYSICS_DT)
    td = vehicle.touchdown or {}
    success = passed(vehicle, sim.tower is not None)
    return {"scenario": sim.scenario_name, "success": success, "state": vehicle.state,
            "reason": vehicle.crash_reason, "time": round(sim.time, 1), "touchdown": td,
            "propellant_left": round(vehicle.propellant, 1)}


def run_tests() -> int:
    """Print one JSON line per scenario; return a process exit code."""
    ok = True
    for index in range(len(SCENARIOS)):
        result = run_scenario(index)
        ok &= result["success"]
        print(json.dumps(result), flush=True)
    return 0 if ok else 1
