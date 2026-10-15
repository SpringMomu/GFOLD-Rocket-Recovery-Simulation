"""Closed-loop ignition diagnostics for the 3D autopilot (read-only; changes no control).

    python -m tools.ignition_diagnostic [SCENARIO_INDEX ...] [--out DIR]

Flies the scenarios until six seconds after ignition and records, around the
landing-burn start, how far the vehicle and the commanded attitude are from the
plan.  Writes ``ignition_diagnostic.json`` to DIR (default ``tools/out``) and
prints a one-line summary per scenario.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from lander3d.sim import PHYSICS_DT, SCENARIOS, Sim3D

SAMPLE_EVERY = 24          # physics steps between rows (plus one row per re-plan)
POST_IGNITION_SECONDS = 6.0
PRE_IGNITION_SECONDS = 5.0


def angle_deg(a, b) -> float:
    cosine = np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(float(np.clip(cosine, -1, 1))))


def diagnose(index: int) -> dict:
    sim = Sim3D(cfd=False)
    sim.load_scenario(index)
    ap, v = sim.autopilot, sim.vehicle
    rows = []
    ignition = None
    initial_plan = None
    old_plan = None
    prev_cmd = ap.cmd_axis.copy()
    n = 0
    while sim.time < 260 and v.state == "FLYING":
        old_vel = v.vel.copy()
        sim.step(PHYSICS_DT)
        if ap.burn_locked and ignition is None:
            ignition = sim.time
            initial_plan = ap.plan
        changed = ap.plan is not old_plan
        if n % SAMPLE_EVERY == 0 or changed:
            row = dict(
                t=sim.time, phase=ap.phase, h=v.altitude, r=v.pos[:2].tolist(), vel=v.vel.tolist(),
                axis=v.axis.tolist(), cmd_axis=ap.cmd_axis.tolist(),
                cmd_error=angle_deg(v.axis, ap.cmd_axis), cmd_jump=angle_deg(prev_cmd, ap.cmd_axis),
                omega=np.degrees(v.omega).tolist(), throttle=v.throttle, throttle_cmd=v.controls.throttle,
                gimbal=np.degrees(v.gimbal).tolist(), gimbal_cmd=np.degrees(v.controls.gimbal).tolist(),
                thrust=(v.last["thrust"] / v.mass).tolist(), aero=(v.last["aero"] / v.mass).tolist(),
                actual_acc=((v.vel - old_vel) / PHYSICS_DT).tolist(), ignition_in=ap.ignition_in,
                dir_rate=np.degrees(ap._dir_rate).tolist(), replan=changed,
            )
            position = np.array([*v.pos[:2], v.altitude])
            if ap.plan is not None:
                p, vel, u = ap.plan.sample(ap.elapsed - ap.plan.created_at)
                row.update(plan_u=u.tolist(), plan_angle=angle_deg(v.axis, u),
                           cmd_plan_angle=angle_deg(ap.cmd_axis, u), pos_error=(p - position).tolist(),
                           vel_error=(vel - v.vel).tolist(), plan_status=ap.plan.status,
                           reaches=ap.plan.reaches_pad, tf=ap.plan.t_f, end_time=ap.touchdown_time,
                           plan_drag=ap.plan.drag[0].tolist())
            if initial_plan is not None:
                p, vel, _ = initial_plan.sample(sim.time - ignition)
                row["initial_plan_pos_error"] = (p - position).tolist()
                row["initial_plan_vel_error"] = (vel - v.vel).tolist()
            rows.append(row)
        prev_cmd = ap.cmd_axis.copy()
        old_plan = ap.plan
        n += 1
        if ignition is not None and sim.time > ignition + POST_IGNITION_SECONDS:
            break
    rows = [r for r in rows if ignition is None or r["t"] > ignition - PRE_IGNITION_SECONDS]
    return dict(scenario=SCENARIOS[index]["name"], index=index, ignition=ignition, backend=ap.planner.backend, rows=rows)


def summarize(result: dict) -> dict:
    summary = dict(scenario=result["scenario"], ignition=result["ignition"], backend=result["backend"])
    burn = [r for r in result["rows"] if r["phase"] == "LANDING BURN"]
    if burn:
        summary.update(
            max_cmd_error=max(r["cmd_error"] for r in burn),
            max_plan_angle=max(r.get("plan_angle", 0) for r in burn),
            max_initial_pos_error=max(float(np.linalg.norm(r["initial_plan_pos_error"])) for r in burn),
            max_initial_vel_error=max(float(np.linalg.norm(r["initial_plan_vel_error"])) for r in burn),
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="*", type=int, default=list(range(len(SCENARIOS))))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("out"))
    args = parser.parse_args()

    results = []
    for index in args.scenarios:
        results.append(diagnose(index))
        print(json.dumps(summarize(results[-1])), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ignition_diagnostic.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
