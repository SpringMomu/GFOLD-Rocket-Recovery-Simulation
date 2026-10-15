"""Preset starting conditions and the headless landing regression."""

from __future__ import annotations

import json
import math
import sys

from common.conic import NATIVE_BACKEND_PREFIX

from .autopilot import AutonomousGuidance
from .rocket import Rocket

PHYSICS_HZ = 120
FIXED_DT = 1.0 / PHYSICS_HZ

# (name, initial altitude / horizontal position / horizontal speed / vertical speed)
PRESETS = [
    ("1_hover_slam_100m", {"initial_altitude": 100.0}),
    ("2_2000m_up_30mps", {"initial_altitude": 2_000.0, "vertical_speed": 30.0}),
    ("3_2000m_down_30mps", {"initial_altitude": 2_000.0, "vertical_speed": -30.0}),
    ("4_boostback_x+1000_vx+100", {"initial_altitude": 2_000.0, "horizontal_position": 1_000.0, "horizontal_speed": 100.0}),
    ("5_divert_x-1000_vx+100", {"initial_altitude": 2_000.0, "horizontal_position": -1_000.0, "horizontal_speed": 100.0}),
]

# Acceptance limits for a precision landing.
MAX_PAD_ERROR = 1.0     # m
MAX_SINK_SPEED = 2.0    # m/s
MAX_SIDE_SPEED = 1.0    # m/s
MAX_TILT_DEG = 5.0


def place_rocket(rocket: Rocket, initial_altitude: float, horizontal_position: float = 0.0,
                 horizontal_speed: float = 0.0, vertical_speed: float = 0.0) -> None:
    rocket.set_flight_state(initial_altitude, horizontal_position, horizontal_speed)
    rocket.velocity += rocket.local_up() * vertical_speed


def run_landing_scenario(initial_altitude: float, horizontal_position: float = 0.0, horizontal_speed: float = 0.0,
                         vertical_speed: float = 0.0, *, max_seconds: float = 180.0) -> dict:
    """Fly one autonomous landing without a window, through the same physics and guidance as the game."""
    rocket = Rocket()
    place_rocket(rocket, initial_altitude, horizontal_position, horizontal_speed, vertical_speed)
    guidance = AutonomousGuidance()
    ignition_time = ignition_altitude = math.nan
    max_tilt = touchdown_pitch = 0.0
    steps = 0
    for steps in range(1, max(1, math.ceil(max_seconds / FIXED_DT)) + 1):
        guidance.command(rocket, FIXED_DT)
        touchdown_pitch = rocket.local_pitch
        rocket.step(FIXED_DT)
        max_tilt = max(max_tilt, abs(rocket.local_pitch))
        if math.isnan(ignition_time) and guidance.burn_locked:
            ignition_time = steps * FIXED_DT
            ignition_altitude = rocket.altitude
        if rocket.state in {"CRASHED", "LANDED"}:
            break

    touchdown_vx, touchdown_vz = guidance.touchdown_velocity
    pad_error = abs(rocket.position.x - rocket.launch_x)
    backend = guidance.planner.backend
    solved = guidance.last_status in {"Solved", "AlmostSolved"} or guidance.phase == "LANDED"
    success = (
        rocket.state == "LANDED"
        and solved
        and backend != "feedback-fallback"
        and pad_error <= MAX_PAD_ERROR
        and abs(touchdown_vz) <= MAX_SINK_SPEED
        and abs(touchdown_vx) <= MAX_SIDE_SPEED
        and abs(math.degrees(touchdown_pitch)) <= MAX_TILT_DEG
    )
    return {
        "state": rocket.state,
        "success": success,
        "flight_time_s": round(steps * FIXED_DT, 2),
        "ignition_time_s": round(ignition_time, 2),
        "ignition_altitude_m": round(ignition_altitude, 1),
        "landing_pad_error_m": round(pad_error, 3),
        "touchdown_vertical_speed_mps": round(touchdown_vz, 3),
        "touchdown_horizontal_speed_mps": round(touchdown_vx, 3),
        "touchdown_pitch_deg": round(math.degrees(touchdown_pitch), 2),
        "max_pitch_deg": round(math.degrees(max_tilt), 1),
        "solver": guidance.last_status,
        "solver_backend": backend,
        "native_solver": backend.startswith(NATIVE_BACKEND_PREFIX),
        "solve_count": guidance.planner.solve_count,
    }


def run_landing_tests() -> int:
    """Run every preset; print one JSON line each and return a process exit code."""
    results = []
    for _, params in PRESETS:
        result = {"scenario": params, **run_landing_scenario(**params)}
        results.append(result)
        print(json.dumps(result), flush=True)
    if results and not results[0]["native_solver"]:
        print("note: native clarabel_hover.dll (ABI 2) not in use; ran on the Python clarabel package. "
              "Run native\\build_solver.ps1 to rebuild it.", file=sys.stderr)
    return 0 if all(result["success"] for result in results) else 1
