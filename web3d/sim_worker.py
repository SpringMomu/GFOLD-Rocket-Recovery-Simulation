"""Browser-side glue: runs the 3D simulator inside Pyodide (Web Worker).

Same physics, guidance and state as ``python main3d.py``; the worker's timer
plays the role of the server's ``Runner`` thread.
"""

import json
import sys

sys.path.insert(0, "/home/pyodide/sim")

from lander3d.server import finite, snapshot
from lander3d.sim import Sim3D

sim = Sim3D(cfd=True)


def advance(real_seconds: float) -> None:
    sim.advance_realtime(real_seconds)


def state() -> str:
    return json.dumps(finite(snapshot(sim)), separators=(",", ":"), allow_nan=False,
                      default=lambda o: finite(float(o)) if hasattr(o, "__float__") else str(o))


def handle_input(payload: str) -> None:
    p = json.loads(payload)
    sim.handle_input(p.get("keys", {}), p.get("actions", []))


def handle_cfd(payload: str) -> None:
    p = json.loads(payload)
    if sim.cfd is not None and isinstance(p.get("c_world"), list):
        sim.cfd.update(p["c_world"])
