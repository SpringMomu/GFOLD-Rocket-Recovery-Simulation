"""Simulation core of the 3D game: vehicle + aero + wind + autopilot + CFD input.

Used by the web server / browser worker and by the headless tests.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from .aero import AeroModel, ExternalCFD
from .attitude import attitude_control
from .autopilot import Autopilot3D
from .vehicle import Vehicle, Wind, quat_from_axis_angle

PHYSICS_DT = 1.0 / 240.0
MAX_STEPS_PER_CALL = 2400
LOGGED_PHASES = ("COAST", "BOOSTBACK", "LANDING BURN")

SCENARIOS = [
    {"name": "HOVER-SLAM 100 m", "pos": (0, 0, 100), "vel": (0, 0, 0), "tilt": 0.0, "prop": 2500},
    {"name": "2000 m / +30 m/s UP", "pos": (0, 0, 2000), "vel": (0, 0, 30), "tilt": 0.0, "prop": 4000},
    {"name": "2000 m / -30 m/s DOWN", "pos": (0, 0, 2000), "vel": (0, 0, -30), "tilt": 0.0, "prop": 4000},
    {"name": "BOOSTBACK +1000 m / 100 m/s", "pos": (1000, 300, 2000), "vel": (100, -20, 0), "tilt": 10.0, "prop": 5500,
     "fins_stowed": True},
    {"name": "DIVERT -1000 m / 3-D", "pos": (-1000, 600, 2000), "vel": (100, -40, 0), "tilt": 10.0, "prop": 5500,
     "fins_stowed": True},
    {"name": "ENTRY 12 km / 280 m/s", "pos": (1800, -1100, 12000), "vel": (-88, 54, -260), "tilt": 8.0, "prop": 6000},
]


class Sim3D:
    def __init__(self, cfd: bool = True) -> None:
        self.vehicle = Vehicle()
        self.aero = AeroModel(self.vehicle.spec)
        self.wind = Wind()
        self.autopilot = Autopilot3D(self.aero)
        self.cfd = ExternalCFD() if cfd else None
        self.aero.cfd = self.cfd
        self.lock = threading.RLock()
        self.autopilot_on = False
        self.sas = "OFF"          # OFF | STABILITY | RETROGRADE
        self.keys: dict = {}
        self.throttle_manual = 0.0
        self.time = 0.0
        self.time_warp = 1.0
        self.paused = False
        self.trail: list = []
        self.scenario_name = ""
        self.events: list = []
        self._hold_axis = None
        self._realtime_debt = 0.0
        self.load_scenario(3)

    # -- setup -----------------------------------------------------------------
    def load_scenario(self, index: int, autopilot: bool = True) -> None:
        with self.lock:
            sc = SCENARIOS[index % len(SCENARIOS)]
            tilt = math.radians(sc["tilt"])
            vel = np.array(sc["vel"], float)
            horizontal = np.array([vel[0], vel[1], 0.0])
            speed = np.linalg.norm(horizontal)
            axis = np.array([0.0, 1.0, 0.0]) if speed < 1e-6 else np.cross([0.0, 0.0, 1.0], horizontal / speed)
            q = quat_from_axis_angle(axis, -tilt)
            self.vehicle.reset(position=sc["pos"], velocity=sc["vel"], attitude=q, propellant=sc["prop"], on_pad=False)
            self.vehicle.pos[2] += self.vehicle.rest_height
            if sc.get("fins_stowed", False):
                # Before a boost-back the grid fins are still stowed.
                self.vehicle.fins_deploy = 0.0
                self.vehicle.controls.fins_deploy = False
            self.autopilot.reset()
            self.autopilot_on = autopilot
            self.sas = "OFF"
            self.throttle_manual = 0.0
            self.time = 0.0
            self.trail = [self.vehicle.pos.copy()]
            self.scenario_name = sc["name"]
            self.events = [(0.0, f"SCENARIO: {sc['name']}")]

    def reset_to_pad(self) -> None:
        with self.lock:
            self.vehicle.reset(on_pad=True)
            self.autopilot.reset()
            self.autopilot_on = False
            self.throttle_manual = 0.0
            self.time = 0.0
            self.trail = [self.vehicle.pos.copy()]
            self.scenario_name = "ON PAD - MANUAL"

    # -- manual control (KSP layout) ---------------------------------------------
    def _manual(self, dt: float) -> None:
        v = self.vehicle
        keys = self.keys
        controls = v.controls
        # Throttle: Shift up, Ctrl down; Z full and X cut are actions.
        if keys.get("shift"):
            self.throttle_manual = min(1.0, self.throttle_manual + 0.8 * dt)
        if keys.get("ctrl"):
            self.throttle_manual = max(0.0, self.throttle_manual - 0.8 * dt)
        controls.throttle = self.throttle_manual

        # Pitch W/S, yaw A/D, roll Q/E command body rates.
        def axis(positive: str, negative: str) -> float:
            return float(bool(keys.get(positive))) - float(bool(keys.get(negative)))

        w_cmd = np.array([axis("s", "w"), axis("d", "a"), axis("q", "e")]) * math.radians(18.0)
        steering = bool(np.any(w_cmd != 0.0))
        if not steering and self.sas == "RETROGRADE":
            vr = v.vel - self.wind.at(v.pos, v.time)
            speed = float(np.linalg.norm(vr))
            target = -vr / speed if speed > 3.0 else np.array([0.0, 0.0, 1.0])
            attitude_control(v, target, 1.2, 2.5, math.radians(20.0), use_rcs=True)
            return
        if not steering and self.sas == "STABILITY":
            if self._hold_axis is None:
                self._hold_axis = v.axis.copy()
            attitude_control(v, self._hold_axis, 1.2, 2.5, math.radians(20.0), use_rcs=True)
            return
        self._hold_axis = None

        _, zc, ixx, izz = v.mass_properties()
        alpha = 2.5 * (w_cmd - v.omega) if steering else np.zeros(3)
        torque = np.array([ixx, ixx, izz]) * alpha
        spec = v.spec
        thrust = v.throttle * spec.thrust_sl
        gx = gy = 0.0
        if thrust > 0.04 * spec.thrust_sl:
            lever = zc - spec.gimbal_z
            gy = math.asin(float(np.clip(-torque[1] / lever / thrust, -0.99, 0.99)))
            gx = math.asin(float(np.clip(-(torque[0] / lever) / thrust, -0.99, 0.99)))
        controls.gimbal = (gx, gy)
        controls.fins = (0.0, 0.0, 0.0)
        controls.rcs = tuple(float(x) for x in np.clip(torque / np.array(spec.rcs_torque), -1.0, 1.0))

    def apply_action(self, action: str) -> None:
        """Discrete commands from the front end."""
        if action == "autopilot":
            self.autopilot_on = not self.autopilot_on
            if self.autopilot_on:
                self.autopilot.reset()
            self.events.append((self.time, "AUTOPILOT ENGAGED" if self.autopilot_on else "AUTOPILOT OFF"))
        elif action == "sas":
            self.sas = {"OFF": "STABILITY", "STABILITY": "RETROGRADE", "RETROGRADE": "OFF"}[self.sas]
            self._hold_axis = None
            self.events.append((self.time, f"SAS {self.sas}"))
        elif action == "legs":
            controls = self.vehicle.controls
            controls.legs_down = not controls.legs_down
        elif action == "full":
            self.throttle_manual = 1.0
        elif action == "cut":
            self.throttle_manual = 0.0
        elif action.startswith("scenario:"):
            self.load_scenario(int(action.split(":")[1]))
        elif action == "pad":
            self.reset_to_pad()
        elif action == "pause":
            self.paused = not self.paused
        elif action == "warp+":
            self.time_warp = min(4.0, self.time_warp * 2)
        elif action == "warp-":
            self.time_warp = max(0.25, self.time_warp / 2)
        elif action == "cfd_forces":
            self.aero.cfd_forces = not self.aero.cfd_forces
            self.events.append((self.time, "AERO FORCES: " + ("LIVE 3D CFD" if self.aero.cfd_forces else "ENGINEERING MODEL")))
        elif action == "wind":
            self.wind.enabled = not self.wind.enabled
            self.events.append((self.time, "WIND " + ("ON" if self.wind.enabled else "OFF")))

    def handle_input(self, keys: dict, actions: list[str]) -> None:
        """Replace the held-key state and run the queued actions."""
        with self.lock:
            self.keys = {k: bool(v) for k, v in keys.items()}
            for action in actions:
                self.apply_action(action)

    # -- stepping --------------------------------------------------------------
    def step(self, dt: float) -> None:
        with self.lock:
            if self.paused:
                return
            v = self.vehicle
            previous_state = v.state
            wind_now = self.wind.at(v.pos, v.time)
            if self.autopilot_on:
                self.autopilot.command(v, wind_now, dt)
                self.throttle_manual = v.controls.throttle
            else:
                self._manual(dt)
            v.step(dt, self.aero, self.wind)
            self.time += dt
            if len(self.trail) == 0 or np.linalg.norm(v.pos - self.trail[-1]) > 4.0:
                self.trail.append(v.pos.copy())
                if len(self.trail) > 1500:
                    self.trail.pop(0)
            if v.state != previous_state:
                self.events.append((self.time, f"{v.state}{': ' + v.crash_reason if v.crash_reason else ''}"))
            if self.autopilot_on:
                self._log_phase()
            self.events = self.events[-40:]

    def _log_phase(self) -> None:
        phase = self.autopilot.phase
        if phase not in LOGGED_PHASES:
            return
        if self.events and self.events[-1][1].endswith(phase):
            return
        if any(text == f"PHASE {phase}" for _, text in self.events[-6:]):
            return
        self.events.append((self.time, f"PHASE {phase}"))

    def advance_realtime(self, real_seconds: float) -> None:
        """Advance by wall-clock time (scaled by the time warp).

        A slow step (a long solve) never makes the simulation fast-forward to
        catch up; it just runs slower for a moment.
        """
        self._realtime_debt += min(0.1, max(0.0, real_seconds)) * self.time_warp
        self._realtime_debt = min(self._realtime_debt, 0.1 * max(1.0, self.time_warp))
        steps = 0
        while self._realtime_debt >= PHYSICS_DT and steps < MAX_STEPS_PER_CALL:
            self.step(PHYSICS_DT)
            self._realtime_debt -= PHYSICS_DT
            steps += 1
        if steps >= MAX_STEPS_PER_CALL:
            self._realtime_debt = 0.0
