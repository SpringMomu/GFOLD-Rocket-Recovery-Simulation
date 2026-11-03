"""Simulation core of the 3D game: vehicle + aero + wind + autopilot + CFD input.

Used by the web server / browser worker and by the headless tests.
"""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np

from . import settings
from .aero import AeroModel, ExternalCFD
from .attitude import allocate_torque, attitude_control
from .autopilot import Autopilot3D
from .tower import CatchTower
from .vehicle import Vehicle, Wind, quat_from_axis_angle

PHYSICS_DT = 1.0 / 240.0
MAX_STEPS_PER_CALL = 2400
LOGGED_PHASES = ("COAST", "BOOSTBACK", "LANDING BURN", "CATCH HOLD")

SCENARIOS = [
    {"name": "HOVER-SLAM 100 m", "pos": (0, 0, 100), "vel": (0, 0, 0), "tilt": 0.0, "prop": 2500},
    {"name": "2000 m / +30 m/s UP", "pos": (0, 0, 2000), "vel": (0, 0, 30), "tilt": 0.0, "prop": 4000},
    {"name": "2000 m / -30 m/s DOWN", "pos": (0, 0, 2000), "vel": (0, 0, -30), "tilt": 0.0, "prop": 4000},
    {"name": "BOOSTBACK +1000 m / 100 m/s", "pos": (1000, 300, 2000), "vel": (100, -20, 0), "tilt": 10.0, "prop": 5500,
     "fins_stowed": True},
    {"name": "DIVERT -1000 m / 3-D", "pos": (-1000, 600, 2000), "vel": (100, -40, 0), "tilt": 10.0, "prop": 5500,
     "fins_stowed": True},
    {"name": "ENTRY 12 km / 280 m/s", "pos": (1800, -1100, 12000), "vel": (-88, 54, -260), "tilt": 8.0, "prop": 6000},
    # Tower catches: the burn ends with the catch collar on the chopstick arms.
    {"name": "TOWER CATCH 2000 m", "pos": (40, -30, 2000), "vel": (0, 0, -30), "tilt": 0.0, "prop": 4000,
     "catch": True},
    {"name": "TOWER CATCH BOOSTBACK", "pos": (1000, 300, 2000), "vel": (100, -20, 0), "tilt": 10.0, "prop": 5500,
     "fins_stowed": True, "catch": True},
    {"name": "TOWER CATCH ENTRY 12 km", "pos": (1800, -1100, 12000), "vel": (-88, 54, -260), "tilt": 8.0, "prop": 6000,
     "catch": True},
    # Low behind the tower: the straight way to the arms goes through it, the
    # plan goes over it (keep-out constraints).
    {"name": "TOWER CATCH OVER TOWER", "pos": (-150, 0, 100), "vel": (0, 0, -5), "tilt": 0.0, "prop": 4000,
     "catch": True},
]


class Sim3D:
    def __init__(self, cfd: bool = True, scenario: int = 3, wind_on: bool = True,
                 async_plan: bool = False) -> None:
        """``scenario`` is the one loaded at start, ``wind_on`` whether the wind
        blows at start (the headless regressions keep the defaults; the server
        and the browser simulator start in scenario 9 without wind).
        ``async_plan`` re-plans on a background thread (server / browser); the
        headless regression keeps the synchronous path, which is repeatable."""
        self.vehicle = Vehicle()
        self.aero = AeroModel(self.vehicle.spec)
        self.wind = Wind(enabled=wind_on)
        self.autopilot = Autopilot3D(self.aero)
        if async_plan:
            self.autopilot.enable_async_planning()
        self.solver_cfg = settings.defaults()      # guidance settings (the front end's solver panel)
        settings.apply(self.autopilot, self.solver_cfg)
        self.cfd = ExternalCFD() if cfd else None
        self.aero.cfd = self.cfd
        self.lock = threading.RLock()
        self.autopilot_on = False
        self.sas = "STABILITY"    # STABILITY | RETROGRADE | VERTICAL | OFF (rate damping only)
        self._stick = np.zeros(3)
        self._release_time = 0.0
        self._aoa_time = -1e9
        self.keys: dict = {}
        self.throttle_manual = 0.0
        self.time = 0.0
        self.flight_time = 0.0
        self._flight_finished = False
        self.time_warp = 1.0
        self.paused = False
        self.trail: list = []
        self.tower: CatchTower | None = None
        self.scenario_name = ""
        self.events: list = []
        self._hold_axis = None
        self._realtime_debt = 0.0
        self.load_scenario(scenario)

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
            self._set_tower(CatchTower() if sc.get("catch", False) else None)
            if sc.get("fins_stowed", False):
                # Before a boost-back the grid fins are still stowed.
                self.vehicle.fins_deploy = 0.0
                self.vehicle.controls.fins_deploy = False
            self.autopilot.reset()
            self.autopilot_on = autopilot
            self.sas = "STABILITY"
            self._hold_axis = None
            self._stick = np.zeros(3)
            self.throttle_manual = 0.0
            self.time = 0.0
            self.flight_time = 0.0
            self._flight_finished = False
            self.trail = [self.vehicle.pos.copy()]
            self.trail_axes = [self.vehicle.axis.copy()]
            self.history = []
            self._history_next = 0.0
            self._record_history()
            self.scenario_name = sc["name"]
            self.events = [(0.0, f"SCENARIO: {sc['name']}")]

    def reset_to_pad(self) -> None:
        with self.lock:
            self.vehicle.reset(on_pad=True)
            self._set_tower(None)
            self.autopilot.reset()
            self.autopilot_on = False
            self._hold_axis = None
            self.throttle_manual = 0.0
            self.time = 0.0
            self.flight_time = 0.0
            self._flight_finished = False
            self.trail = [self.vehicle.pos.copy()]
            self.trail_axes = [self.vehicle.axis.copy()]
            self.history = []
            self._history_next = 0.0
            self._record_history()
            self.scenario_name = "ON PAD - MANUAL"

    def _set_tower(self, tower: CatchTower | None) -> None:
        self.tower = tower
        self.vehicle.structure = tower
        if tower is None:
            self.autopilot.set_target()
        else:
            self.autopilot.set_target(tower.catch_base_z, self.vehicle.rest_height, legs=False,
                                      obstacles=tower.keep_out_boxes(self.vehicle.spec),
                                      catch_link=lambda: tower.secured)
            self.vehicle.controls.legs_down = False

    # -- manual control (KSP layout, fly-by-wire) --------------------------------
    MANUAL_RATE = math.radians(12.0)    # full-stick body rate
    STICK_TAU = 0.25                    # stick smoothing [s]
    SAS_MODES = ("STABILITY", "RETROGRADE", "VERTICAL", "OFF")

    def _manual(self, dt: float) -> None:
        """Rate-command / attitude-hold fly-by-wire.

        The keys command body rates (they never fire the actuators directly);
        released, the vehicle holds its attitude (STABILITY), points
        retrograde to the air flow, stands vertical, or (OFF) only damps its
        rates.  Torque goes through the same allocation as the autopilot: the
        known aerodynamic moment is cancelled, grid fins first, then the
        engine gimbal, then RCS.  Without that the bare stick left a free
        body that the aerodynamic moment turned over."""
        v = self.vehicle
        keys = self.keys
        controls = v.controls
        # Throttle: Shift up, Ctrl down; Z full and X cut are actions.
        if keys.get("shift"):
            self.throttle_manual = min(1.0, self.throttle_manual + 0.8 * dt)
        if keys.get("ctrl"):
            self.throttle_manual = max(0.0, self.throttle_manual - 0.8 * dt)
        controls.throttle = self.throttle_manual

        # Pitch W/S, yaw A/D, roll Q/E.
        def axis(positive: str, negative: str) -> float:
            return float(bool(keys.get(positive))) - float(bool(keys.get(negative)))

        raw = np.array([axis("s", "w"), axis("d", "a"), axis("q", "e")])
        self._stick += (raw - self._stick) * min(1.0, dt / self.STICK_TAU)
        self._stick[np.abs(self._stick) < 1e-3] = 0.0
        steering = bool(np.any(raw[:2] != 0.0))
        rolling = raw[2] != 0.0 or abs(self._stick[2]) > 0.02

        _, _, ixx, izz = v.mass_properties()
        aoa = self._manual_aoa(v)
        if not steering and self.sas != "OFF":
            target = None
            if self.sas == "RETROGRADE":
                vr = v.vel - self.wind.at(v.pos, v.time)
                speed = float(np.linalg.norm(vr))
                target = -vr / speed if speed > 3.0 else np.array([0.0, 0.0, 1.0])
            elif self.sas == "VERTICAL":
                target = np.array([0.0, 0.0, 1.0])
            else:
                # Capture the attitude once the stick rates have died away,
                # so releasing the keys does not snap back.
                # The aerodynamic moment may keep it turning: capture anyway
                # after a short damping phase.
                self._release_time += dt
                if self._hold_axis is None and (float(np.linalg.norm(v.omega[:2])) < math.radians(1.5)
                                                or self._release_time > 0.6):
                    self._hold_axis = v.axis.copy()
                target = self._hold_axis
            if target is not None and aoa is not None:
                target = self._clamp_to_flow(target, *aoa)
            if target is not None:
                attitude_control(v, target, 1.2, 2.5, math.radians(15.0), use_rcs=True,
                                 hold_roll=not rolling, roll_rate_cmd=float(self._stick[2]) * self.MANUAL_RATE)
                return
        if steering:
            self._hold_axis = None
            self._release_time = 0.0
        # Rate command (stick) or rate damping (released, hold not captured yet / SAS off).
        w_cmd = self._stick * self.MANUAL_RATE
        if not steering:
            w_cmd[:2] = 0.0
        elif aoa is not None:
            w_cmd = self._protect_rates(v, w_cmd, *aoa)
        alpha = 2.5 * (w_cmd - v.omega)
        allocate_torque(v, np.array([ixx, ixx, izz]) * alpha, use_rcs=True)

    def _manual_aoa(self, v):
        """Envelope protection for the stick: (flow direction w, current and
        allowed angle off engines-first) while falling through the air, from
        the autopilot's angle-of-attack table (the largest angle at which the
        fins + gimbal still hold the aerodynamic moment).  None when there is
        no limit (slow, thin air, or rising: the table is for engines-first
        descent)."""
        vr = v.vel - self.wind.at(v.pos, v.time)
        sp = float(np.linalg.norm(vr))
        q = float(v.last.get("q", 0.0))
        if sp < 3.0 or vr[2] >= 0.0 or q < 200.0:
            return None
        ap = self.autopilot
        if self.time - self._aoa_time > 2.0:
            ap._aoa_cache = (-1e9, ap._aoa_cache[1])       # recompute at the current mass
            self._aoa_time = self.time
        table = ap._aoa_table(v)
        allowed = math.radians(float(np.interp(q, *table)))
        if allowed >= math.radians(179.0):
            return None
        w = -vr / sp
        ang = math.acos(float(np.clip(np.dot(v.axis, w), -1.0, 1.0)))
        return w, ang, allowed

    @staticmethod
    def _clamp_to_flow(target, w, _ang, allowed):
        """Pull an attitude target into the cone ``allowed`` around the flow."""
        target = np.asarray(target, float)
        a = math.acos(float(np.clip(np.dot(target, w), -1.0, 1.0)))
        if a <= allowed:
            return target
        side = target - w * float(np.dot(target, w))
        n = float(np.linalg.norm(side))
        if n < 1e-9:
            return w.copy()
        return w * math.cos(allowed) + side / n * math.sin(allowed)

    def _protect_rates(self, v, w_cmd, w, ang, allowed):
        """Near the limit the stick cannot turn the axis further off the
        flow (that moment is more than the controls can stop); past it, the
        axis is brought back."""
        g = np.cross(v.axis, w)                 # body rates along g turn the axis toward the flow
        gn = float(np.linalg.norm(g))
        if gn < 1e-9:
            return w_cmd
        g_b = v.R.T @ (g / gn)
        g_b[2] = 0.0
        gbn = float(np.linalg.norm(g_b))
        if gbn < 1e-9:
            return w_cmd
        g_b /= gbn
        out = w_cmd.copy()
        margin = 0.85 * allowed
        if ang > margin:
            along = float(np.dot(out, g_b))
            # fade the outward rate to zero between 85 % and 100 % of the limit
            keep = float(np.clip((allowed - ang) / max(1e-6, allowed - margin), 0.0, 1.0))
            if along < 0.0:
                out -= (1.0 - keep) * along * g_b
            if ang > allowed:
                out += min(self.MANUAL_RATE, 1.5 * (ang - allowed)) * g_b
        return out

    def apply_action(self, action: str) -> None:
        """Discrete commands from the front end."""
        if action == "autopilot":
            self.autopilot_on = not self.autopilot_on
            if self.autopilot_on:
                self.autopilot.reset()
            else:
                self._hold_axis = None      # manual takes over holding the current attitude
            self.events.append((self.time, "AUTOPILOT ENGAGED" if self.autopilot_on else "AUTOPILOT OFF"))
        elif action == "sas":
            modes = self.SAS_MODES
            self.sas = modes[(modes.index(self.sas) + 1) % len(modes)] if self.sas in modes else modes[0]
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
        elif action == "replan":
            # Continuous planning off: the plan in hand (the last one made) is flown to touchdown by the tracker alone
            self.autopilot.replan_enabled = not self.autopilot.replan_enabled
            self.events.append((self.time, "REPLAN " + ("ON" if self.autopilot.replan_enabled else "OFF")))
        elif action == "wind":
            self.wind.enabled = not self.wind.enabled
            self.events.append((self.time, "WIND " + ("ON" if self.wind.enabled else "OFF")))
        elif action.startswith("windcfg:"):
            # {"from": [first, last] deg, "speed": [min, max] m/s, "shear": k}
            try:
                cfg = json.loads(action[len("windcfg:"):])
                self.wind.configure(cfg.get("from"), cfg.get("speed"), cfg.get("shear"))
            except (ValueError, TypeError, AttributeError, IndexError):
                return
            c = self.wind.config()
            self.events.append((self.time, "WIND CFG {:.0f}-{:.0f} {:.1f}-{:.1f} {:.1f}".format(*c["from"], *c["speed"], c["shear"])))
        elif action.startswith("solvercfg:"):
            # {"<option>": value, ...} (lander3d/settings.py); the guidance replans with them from now on
            try:
                changed = self.set_solver_cfg(json.loads(action[len("solvercfg:"):]))
            except ValueError:
                return
            if changed:
                self.events.append((self.time, "SOLVER CFG"))
        elif action == "solverreset":
            self.solver_cfg = settings.defaults()
            settings.apply(self.autopilot, self.solver_cfg)
            self.events.append((self.time, "SOLVER CFG DEFAULT"))

    def set_solver_cfg(self, raw) -> bool:
        """Merge the valid entries of ``raw`` into the guidance settings."""
        changes = settings.sanitize(raw)
        if not changes:
            return False
        self.solver_cfg.update(changes)
        settings.apply(self.autopilot, self.solver_cfg)
        return True

    def handle_input(self, keys: dict, actions: list[str]) -> None:
        """Replace the held-key state and run the queued actions."""
        with self.lock:
            self.keys = {k: bool(v) for k, v in keys.items()}
            for action in actions:
                self.apply_action(action)

    # -- stepping --------------------------------------------------------------
    def _record_history(self) -> None:
        """Time, vertical tilt, speed, altitude and horizontal target error."""
        v = self.vehicle
        self.history.append([round(self.time, 2), round(v.tilt_deg(), 2),
                             round(float(np.linalg.norm(v.vel)), 2), round(v.altitude, 2),
                             round(float(np.linalg.norm(v.pos[:2])), 2)])
        self._history_next = self.time + 0.5

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
            if self.tower is not None:
                self.tower.update(dt, v)
            v.step(dt, self.aero, self.wind)
            self.time += dt
            if not self._flight_finished:
                self.flight_time = self.time
                self._flight_finished = v.state in ("LANDED", "CAUGHT")
            if len(self.trail) == 0 or np.linalg.norm(v.pos - self.trail[-1]) > 4.0:
                self.trail.append(v.pos.copy())
                self.trail_axes.append(v.axis.copy())
            if ((not self._flight_finished and v.state != "CRASHED" and self.time >= self._history_next)
                    or (v.state in ("LANDED", "CAUGHT", "CRASHED") and v.state != previous_state)):
                self._record_history()
            if self.tower is not None and self.tower.events:
                self.events.extend((self.time, e) for e in self.tower.events)
                self.tower.events.clear()
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
        elapsed = min(0.1, max(0.0, real_seconds))
        self._realtime_debt += elapsed * self.time_warp
        self._realtime_debt = min(self._realtime_debt, 0.1 * max(1.0, self.time_warp))
        ap = self.autopilot
        # The boost-back solve gets a fixed number of steps per simulated
        # second (Autopilot3D.boostback_work_rate), interleaved with the
        # physics steps.  When the wall slice runs out first, the clock waits
        # for it: a slow CPU runs the return in slow motion, never a different
        # (late-solved) one.  Guidance per simulated second must not depend on
        # the machine's speed.
        deadline = time.monotonic() + ap.boostback_slice_wall
        steps = 0
        while steps < MAX_STEPS_PER_CALL:
            guided = not self.paused and self.autopilot_on
            if guided and ap.boostback_work_pending():
                # Reset/input/snapshot may run on another server thread. The
                # lock covers only this bounded slice, never a complete solve.
                with self.lock:
                    ap.advance_boostback_work(0.0, deadline)
                if ap.boostback_work_pending():
                    break                   # out of wall time: physics waits for the guidance
                continue
            if self._realtime_debt < PHYSICS_DT:
                break
            self.step(PHYSICS_DT)
            self._realtime_debt -= PHYSICS_DT
            steps += 1
            if guided:
                with self.lock:
                    ap.advance_boostback_work(PHYSICS_DT, deadline)
        if steps >= MAX_STEPS_PER_CALL:
            self._realtime_debt = 0.0
