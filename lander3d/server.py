"""Local web server for the 3D simulator: JSON state out, keys / CFD forces in.

``GET /api/state`` returns :func:`snapshot`; ``POST /api/input`` takes held keys
and queued actions; ``POST /api/cfd3d`` takes the GPU CFD force coefficients.
Everything else is served from ``web3d/``.
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .sim import SCENARIOS, Sim3D
from .vehicle import G0, atmosphere

WEB_DIR = Path(__file__).resolve().parents[1] / "web3d"


def finite(o):
    """Replace NaN/inf by None so the browser can always parse the state."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, np.floating):
        return finite(float(o))
    return o


def _vec(a):
    return [round(float(x), 4) for x in a]


def snapshot(sim: Sim3D) -> dict:
    with sim.lock:
        v = sim.vehicle
        s = v.spec
        ap = sim.autopilot
        a = sim.aero.last or {}
        rho, p, _, _ = atmosphere(max(0.0, v.pos[2]))
        m, zc, _, _ = v.mass_properties()
        heading, pitch, roll = v.euler_deg()
        thrust = float(v.last.get("thrust_mag", 0.0))
        mdot = float(v.last.get("mass_flow", 0.0))
        burn_left = v.propellant / mdot if mdot > 1e-3 else math.inf
        wind = sim.wind.at(v.pos, v.time)
        plan = ap.predicted_points if sim.autopilot_on else []
        if len(plan) > 80:
            plan = plan[:: max(1, len(plan) // 80)] + [plan[-1]]
        trail = sim.trail[-600:]
        if len(trail) > 200:
            trail = trail[:: len(trail) // 200] + [trail[-1]]
        cfd = sim.cfd
        return {
            "t": round(sim.time, 3), "state": v.state, "crash": v.crash_reason, "scenario": sim.scenario_name,
            "autopilot": sim.autopilot_on, "sas": sim.sas, "paused": sim.paused, "warp": sim.time_warp,
            "phase": ap.phase if sim.autopilot_on else "MANUAL", "visited": ap.visited, "tgo": None if not math.isfinite(ap.time_to_go) else round(ap.time_to_go, 2),
            "solver": {"status": ap.last_status, "backend": ap.planner.backend, "solves": ap.planner.solve_count,
                       "margin": round(1.0 - ap.burn_fraction, 3), "locked": ap.burn_locked},
            "pos": _vec(v.pos), "vel": _vec(v.vel), "q": _vec(v.q), "omega": _vec(v.omega), "alt": round(v.altitude, 2),
            "heading": round(heading, 2), "pitch": round(pitch, 2), "roll": round(roll, 2), "tilt": round(v.tilt_deg(), 2),
            "throttle": round(v.throttle, 4), "cmd_throttle": round(ap.cmd_throttle, 4) if sim.autopilot_on else None,
            "cmd_axis": _vec(ap.cmd_axis) if sim.autopilot_on else None,
            "gimbal": _vec(np.degrees(v.gimbal)), "gimbal_limit": math.degrees(s.gimbal_limit), "rcs": _vec(v.rcs),
            "legs": round(v.legs, 3), "legs_cmd": v.controls.legs_down, "feet": v.feet_contact, "catch": v.catch_contact,
            "fins": {"defl": [round(float(math.degrees(x)), 2) for x in v.fin_defl], "deploy": round(float(v.fins_deploy), 3)},
            "fuel": {"lox": round(v.lox, 1), "rp1": round(v.rp1, 1), "lox_cap": s.lox_capacity, "rp1_cap": s.rp1_capacity,
                     "mass": round(m, 1), "dry": s.dry_mass, "mdot": round(mdot, 2), "isp": round(float(v.last.get("isp", s.isp_sl)), 1),
                     "dv": round(v.delta_v_remaining(p), 1), "burn_left": None if not math.isfinite(burn_left) else round(burn_left, 1),
                     "thrust": round(thrust, 1), "twr": round(thrust / (m * G0), 3), "com_z": round(zc, 3)},
            "aero": {"mach": round(float(a.get("mach", 0.0)), 3), "q": round(float(a.get("q", 0.0)), 1), "alpha": round(float(a.get("alpha", 0.0)), 2),
                     "drag": round(float(a.get("drag", 0.0)), 1), "normal": round(float(a.get("normal", 0.0)), 1),
                     "cp_z": round(float(a.get("cp_z", zc)), 3), "com_z": round(zc, 3), "source": a.get("source", "MODEL"),
                     "force": _vec(a.get("force", np.zeros(3))), "tail_first": bool(a.get("tail_first", True)),
                     "rho": round(rho, 4), "cfd_forces": sim.aero.cfd_forces},
            "wind": _vec(wind), "wind_on": sim.wind.enabled,
            "plan": [_vec(pt) for pt in plan], "trail": [_vec(pt) for pt in trail],
            "touchdown": v.touchdown, "tower": None if sim.tower is None else sim.tower.snapshot(), "events": [[round(t, 1), e] for t, e in sim.events[-6:]],
            "cfd": None if cfd is None else {"fresh": cfd.fresh},
            "spec": {"length": s.length, "diameter": s.diameter, "grid_fin_z": s.grid_fin_z, "nozzle_exit_z": s.nozzle_exit_z,
                     "rest": v.rest_height},
        }


class Runner:
    """Runs the simulation in real time on a background thread."""

    def __init__(self, sim: Sim3D) -> None:
        self.sim = sim
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        last = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            self.sim.advance_realtime(now - last)
            last = now
            time.sleep(0.002)


def make_handler(sim: Sim3D):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIR), **kwargs)

        def log_message(self, *args):
            pass

        def end_headers(self):
            # Front-end files change during development: never let the browser
            # reuse a stale cached copy of index.html / *.js.
            if not self.path.startswith("/api/"):
                self.send_header("Cache-Control", "no-store, must-revalidate")
            super().end_headers()

        def _json(self, obj, code=200):
            body = json.dumps(finite(obj), separators=(",", ":"), allow_nan=False,
                              default=lambda o: finite(float(o)) if isinstance(o, np.floating) else str(o)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                return self._json(snapshot(sim))
            if self.path.startswith("/api/scenarios"):
                return self._json([s["name"] for s in SCENARIOS])
            return super().do_GET()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                payload = {}
            if self.path.startswith("/api/cfd3d"):
                if sim.cfd is not None and isinstance(payload.get("c_world"), list):
                    sim.cfd.update(payload["c_world"])
                return self._json({"ok": True})
            if self.path.startswith("/api/input"):
                sim.handle_input(payload.get("keys", {}), payload.get("actions", []))
                return self._json(snapshot(sim))
            return self._json({"error": "unknown"}, 404)

    return Handler


def serve(port: int, open_browser: bool) -> None:
    sim = Sim3D(cfd=True)
    Runner(sim).thread.start()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(sim))
    url = f"http://127.0.0.1:{port}/"
    print(f"3D simulator running at {url}  (Ctrl+C to quit)")
    if open_browser:
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
