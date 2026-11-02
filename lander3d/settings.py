"""User-adjustable guidance (G-FOLD) settings: the front end's "solver panel".

Every option's default is the value the guidance had before the panel existed,
so an untouched panel changes nothing (bit for bit).  The front end sends
``solvercfg:{json}`` (a dict of key -> value; unknown or malformed entries are
ignored, numbers are clamped to the range), and the snapshot reports the
values and the ranges back (``solver_cfg`` / ``solver_opts``), so the sliders
always show what the guidance really uses.

``apply`` writes the values onto the autopilot / planner instances (their
class attributes are the defaults).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Option:
    key: str
    kind: str           # "float" | "int" | "bool"
    default: float
    lo: float
    hi: float
    step: float


# (key, kind, default, lo, hi, step).  Groups and labels are the front end's.
OPTIONS = (
    # --- wind resistance: how much the plan counts on / fights the wind
    Option("wind_plan_gain", "float", 1.0, 0.0, 1.5, 0.05),      # wind assumed by the planner's predictions, x measured
    Option("aero_pred", "int", 2, 0, 2, 1),                      # 0 none, 1 drag only, 2 drag + lift linearisation
    Option("trim_gain", "float", 1.0, 0.0, 1.0, 0.05),           # lean into the wind (tilt-cone / tracker trim)
    Option("lateral_aero_share", "float", 1.0, 0.0, 1.0, 0.05),  # share of the aerodynamic force the sideways-effort cost nets off
    Option("shear_fit", "bool", 1, 0, 1, 1),                     # fit the wind shear from the winds measured on the way down
    Option("lift_trim", "bool", 1, 0, 1, 1),                     # lift as the attitude loop trims it (fins), calibrated in flight
    Option("aoa_margin", "float", 0.8, 0.3, 1.0, 0.05),          # share of the control moment the aerodynamic moment may use
    # --- planner / solver
    Option("time_weight", "float", 1.0, 0.0, 3.0, 0.1),          # cost of a second of flight (propellant units)
    Option("knots", "int", 40, 20, 60, 2),                       # number of plan nodes (>= this; long plans use ~1 per s)
    Option("lateral_accel_weight", "float", 1.0, 0.0, 5.0, 0.1),
    Option("jerk_weight", "float", 0.1, 0.01, 1.0, 0.01),
    Option("lateral_rate", "float", 8.0, 1.0, 16.0, 0.5),        # m/s^3, limit on the change of planned sideways thrust
    Option("divert_tilt_deg", "float", 65.0, 30.0, 80.0, 1.0),   # largest thrust tilt of the plan
    Option("burn_fraction", "float", 0.85, 0.6, 0.97, 0.01),     # thrust ceiling of the plan (share of full thrust)
    Option("time_search_evals", "int", 9, 5, 14, 1),             # solves of the touchdown-time search at ignition
    Option("retime_step", "float", 0.08, 0.02, 0.2, 0.01),       # touchdown time trial step (share of the time to go)
    Option("max_iter", "int", 100, 50, 300, 10),                 # interior-point iteration cap
)
BY_KEY = {o.key: o for o in OPTIONS}


def defaults() -> dict:
    return {o.key: (int(o.default) if o.kind != "float" else float(o.default)) for o in OPTIONS}


def sanitize(raw) -> dict:
    """The valid, clamped entries of ``raw`` (anything else is dropped)."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        o = BY_KEY.get(key)
        if o is None or isinstance(value, (str, bytes, list, dict)) or value is None:
            continue
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x):
            continue
        x = min(o.hi, max(o.lo, x))
        if o.kind == "float":
            x = round(round(x / o.step) * o.step, 6) if o.step > 0 else x
            out[key] = min(o.hi, max(o.lo, x))
        else:
            out[key] = round(x)
    return out


def options_snapshot() -> list:
    """Static description for the front end: [key, kind, lo, hi, step, default]."""
    return [[o.key, o.kind, o.lo, o.hi, o.step, o.default] for o in OPTIONS]


def apply(autopilot, values: dict) -> None:
    """Put the (full, sanitised) values on the autopilot and its planner."""
    ap, pl = autopilot, autopilot.planner
    ap.wind_plan_gain = float(values["wind_plan_gain"])
    ap.aero_pred = int(values["aero_pred"])
    ap.trim_gain = float(values["trim_gain"])
    ap.shear_fit = bool(values["shear_fit"])
    ap.lift_trim = bool(values["lift_trim"])
    ap.aoa_moment_margin = float(values["aoa_margin"])
    ap.time_weight = float(values["time_weight"])
    ap.divert_tilt_deg = float(values["divert_tilt_deg"])
    ap.burn_fraction_nominal = float(values["burn_fraction"])
    if not ap.burn_locked:
        ap.burn_fraction = ap.burn_fraction_nominal
    ap.time_search_evals = int(values["time_search_evals"])
    ap.retime_step = float(values["retime_step"])
    pl.lateral_aero_share = float(values["lateral_aero_share"])
    pl.knots = int(values["knots"])
    pl.lateral_accel_weight = float(values["lateral_accel_weight"])
    pl.jerk_weight = float(values["jerk_weight"])
    pl.lateral_rate = float(values["lateral_rate"])
    pl.max_iter = int(values["max_iter"])
    if not ap.shear_fit:
        ap.wind_shear_est = 1.0
