"""Interactive pygame front end for the 2D rocket.

Keys: W/S throttle, A/D gimbal, Z full, Space cut, M autopilot, 1-5 preset
landing scenarios, T edit initial state, R reset, P pause, Tab force vectors,
F1 help, mouse wheel zoom.  The window only renders ``Rocket`` and the
telemetry the cockpit reads; guidance runs on the same fixed step as headless.
"""

from __future__ import annotations

import math
import os

import pygame

from .autopilot import AutonomousGuidance
from .backdrop import Backdrop
from .camera import Camera
from .cockpit import Cockpit, Telemetry
from .dialog import StateDialog
from .rocket import Rocket
from .scenarios import FIXED_DT, PRESETS, place_rocket
from .visuals import Visuals

SCREEN_SIZE = (1280, 720)
FPS = 120
TRAJECTORY_PERIOD = 0.20       # s between points of the flown trace
TRAJECTORY_MAX_POINTS = 2400


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


class RocketSimulator:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Vector Thrust - 2D Rocket Simulator")
        self.screen = pygame.display.set_mode(SCREEN_SIZE, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.rocket = Rocket()
        self.guidance = AutonomousGuidance()
        self.camera = Camera(SCREEN_SIZE)
        self.backdrop = Backdrop()
        self.visuals = Visuals()
        self.cockpit = Cockpit()
        self.dialog = StateDialog()
        self.autopilot_enabled = _env_flag("ROCKET_AUTOPILOT", "0")
        self.running = True
        self.paused = False
        self.show_help = False
        self.show_vectors = False
        self.scenario_title = ""
        self.accumulator = 0.0
        self.mission_time = 0.0
        self.touchdown: dict | None = None
        self.trajectory: list[pygame.Vector2] = []
        self.predicted_trajectory: list[pygame.Vector2] = []
        self.trajectory_timer = 0.0
        self._restart_trace()
        self.update_camera()
        # Start in a demo landing so the guidance is visible at once.
        if _env_flag("ROCKET_DEMO", "1"):
            self.launch_preset(3)

    # -- starting flights ------------------------------------------------------
    def _restart_trace(self) -> None:
        """Forget effects, mission clock and traces after the rocket was moved."""
        self.visuals.reset()
        self.cockpit.reset()
        self.mission_time = 0.0
        self.touchdown = None
        self.trajectory = [self.rocket.position.copy()]
        self.predicted_trajectory = []
        self.trajectory_timer = 0.0
        self.accumulator = 0.0

    def start_flight(self, initial_altitude: float, horizontal_position: float = 0.0,
                     horizontal_speed: float = 0.0, vertical_speed: float = 0.0) -> None:
        place_rocket(self.rocket, initial_altitude, horizontal_position, horizontal_speed, vertical_speed)
        self.guidance.reset()
        self._restart_trace()
        self.update_camera()

    def launch_preset(self, index: int) -> None:
        """Start preset ``index`` (0-4) with the autopilot engaged."""
        name, params = PRESETS[index]
        self.autopilot_enabled = True
        self.camera.enable_auto()
        self.start_flight(**params)
        self.scenario_title = f"DEMO {name.replace('_', ' ')}"

    # -- main loop -------------------------------------------------------------
    def run(self, max_frames: int | None = None) -> None:
        frames = 0
        while self.running and (max_frames is None or frames < max_frames):
            frame_dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            self._handle_events()
            if not self.paused and not self.dialog.active:
                # Manual keys and the autopilot are separate command sources;
                # feeding both would make the manual gimbal slew fight the
                # autopilot's every frame.
                if not self.autopilot_enabled:
                    self.rocket.apply_input(pygame.key.get_pressed(), frame_dt)
                self.accumulator += frame_dt
                while self.accumulator >= FIXED_DT:
                    self.advance_physics()
                    self.accumulator -= FIXED_DT
                self.visuals.update(frame_dt, self.rocket)
            self.update_camera()
            self.draw()
            pygame.display.flip()
            frames += 1
        pygame.quit()

    def advance_physics(self) -> None:
        """One fixed step of autopilot + physics, plus trace and touchdown bookkeeping."""
        rocket = self.rocket
        if self.autopilot_enabled:
            self.guidance.command(rocket, FIXED_DT)
        before = (rocket.position.x - rocket.launch_x, rocket.vertical_speed,
                  rocket.horizontal_speed, math.degrees(rocket.local_pitch))
        was_flying = rocket.state == "FLYING"
        rocket.step(FIXED_DT)
        if rocket.state == "FLYING":
            self.mission_time += FIXED_DT
        if was_flying and rocket.state == "LANDED" and self.touchdown is None:
            self.touchdown = {"x": before[0], "vz": before[1], "vx": before[2], "tilt": before[3]}
        self._sync_predicted_trajectory()
        self.trajectory_timer += FIXED_DT
        if self.trajectory_timer >= TRAJECTORY_PERIOD:
            self.trajectory_timer = 0.0
            self.trajectory.append(rocket.position.copy())
            if len(self.trajectory) > TRAJECTORY_MAX_POINTS:
                self.trajectory.pop(0)

    def _sync_predicted_trajectory(self) -> None:
        """Copy the planner's remaining trajectory (green line) for drawing."""
        points = self.guidance.predicted_points
        if not self.autopilot_enabled or not points:
            self.predicted_trajectory = []
            return
        raw = [pygame.Vector2(x, y) for x, y in points]
        raw[0] = self.rocket.position.copy()
        for point in raw:
            point.y = max(point.y, self.rocket.world_y_for_altitude(point.x, 0.0))
        self.predicted_trajectory = raw

    # -- input -----------------------------------------------------------------
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.dialog.active:
                state = self.dialog.handle_event(event, self.rocket)
                if state is not None:
                    self.start_flight(*state)
            elif event.type == pygame.MOUSEWHEEL:
                self.camera.wheel(event.y, self.autopilot_enabled)
                self.update_camera()
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

    def _handle_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_r:
            self.scenario_title = ""
            self.rocket.reset()
            self.guidance.reset()
            self._restart_trace()
        elif key == pygame.K_SPACE:
            self.rocket.throttle = 0.0
        elif key == pygame.K_z:
            self.rocket.throttle = 1.0
        elif key == pygame.K_t:
            self.dialog.open()
            self.accumulator = 0.0
        elif key == pygame.K_m:
            self.autopilot_enabled = not self.autopilot_enabled
            if self.autopilot_enabled:
                self.guidance.reset()
                self.camera.enable_auto()
            else:
                self.predicted_trajectory = []
            self._sync_predicted_trajectory()
        elif key == pygame.K_p:
            self.paused = not self.paused
            self.accumulator = 0.0
        elif pygame.K_1 <= key <= pygame.K_5:
            self.launch_preset(key - pygame.K_1)
        elif key == pygame.K_F1:
            self.show_help = not self.show_help
        elif key == pygame.K_TAB:
            self.show_vectors = not self.show_vectors

    # -- drawing ---------------------------------------------------------------
    def update_camera(self) -> None:
        frame_pad = self.autopilot_enabled and self.rocket.state != "ON PAD"
        self.camera.update(self.screen.get_size(), self.rocket, self.rocket.pad_position(), frame_pad)

    def draw(self) -> None:
        screen, camera, rocket = self.screen, self.camera, self.rocket
        screen.fill((3, 7, 15))
        self.backdrop.draw(screen, camera, rocket)
        self.visuals.draw_pad(screen, rocket, camera.world_to_screen, camera.scale, self.mission_time)
        self._draw_trajectory()
        self.visuals.draw_particles(screen, camera.world_to_screen, camera.scale)
        icon_scale = max(camera.scale, 26.0 / rocket.LENGTH)
        self.visuals.draw_plume(screen, rocket, camera.world_to_screen, camera.scale, icon_scale)
        self.visuals.draw_rocket(screen, rocket, camera.world_to_screen, icon_scale)
        if self.show_vectors:
            self._draw_force_vectors()
        self.cockpit.draw(screen, self._telemetry())
        if self.show_help:
            self.cockpit.help_overlay(screen)
        if self.dialog.active:
            self.dialog.draw(screen, self.cockpit)

    def _draw_trajectory(self) -> None:
        to_screen = self.camera.world_to_screen
        # Green: the planner's forecast.  Blue: the measured flight path.
        if len(self.predicted_trajectory) >= 2:
            forecast = [to_screen(point) for point in self.predicted_trajectory]
            pygame.draw.lines(self.screen, (18, 70, 40), False, forecast, 4)
            pygame.draw.aalines(self.screen, (90, 255, 150), False, forecast)
        if len(self.trajectory) >= 2:
            stride = max(1, math.ceil(len(self.trajectory) / 600))
            sampled = self.trajectory[::stride]
            if sampled[-1] is not self.trajectory[-1]:
                sampled = sampled + [self.trajectory[-1]]
            pygame.draw.lines(self.screen, (60, 150, 230), False, [to_screen(point) for point in sampled], 2)

    def _draw_force_vectors(self) -> None:
        origin = self.camera.world_to_screen(self.rocket.position)

        def arrow(vector: pygame.Vector2, color: tuple[int, int, int], scale: float) -> None:
            if vector.length_squared() < 1e-8:
                return
            screen_vector = pygame.Vector2(vector.x, -vector.y) * scale
            if screen_vector.length() > 150.0:
                screen_vector.scale_to_length(150.0)
            end = origin + screen_vector
            pygame.draw.line(self.screen, color, origin, end, 3)
            back = origin - end
            if back.length_squared() > 0.0:
                back.scale_to_length(9.0)
                side = pygame.Vector2(-back.y, back.x) * 0.45
                pygame.draw.polygon(self.screen, color, [end, end + back + side, end + back - side])

        arrow(self.rocket.last_thrust, (105, 255, 132), 0.00012)
        arrow(self.rocket.last_drag, (255, 202, 79), 0.00035)
        velocity = pygame.Vector2(self.rocket.velocity.x, -self.rocket.velocity.y) * 1.2
        if velocity.length_squared() > 1.0:
            if velocity.length() > 130.0:
                velocity.scale_to_length(130.0)
            pygame.draw.line(self.screen, (104, 205, 255), origin, origin + velocity, 2)

    def _telemetry(self) -> Telemetry:
        rocket, guidance = self.rocket, self.guidance
        a_max = rocket.MAX_THRUST / rocket.mass
        local_rate = rocket.angular_velocity - rocket.horizontal_speed / max(
            1.0, rocket.EARTH_RADIUS + rocket.geometric_altitude)
        cmd_x, cmd_z = guidance.last_command
        commanding = self.autopilot_enabled and rocket.state == "FLYING"
        flying = rocket.state == "FLYING"
        pad = rocket.pad_position()
        return Telemetry(
            mission_time=self.mission_time,
            autopilot=self.autopilot_enabled,
            phase=guidance.phase if self.autopilot_enabled else "MANUAL",
            state=rocket.state,
            crash_reason=rocket.crash_reason,
            altitude=rocket.altitude,
            x=rocket.position.x - rocket.launch_x,
            vx=rocket.horizontal_speed,
            vz=rocket.vertical_speed,
            tilt_deg=math.degrees(rocket.local_pitch),
            tilt_rate_deg=math.degrees(local_rate),
            gimbal_deg=math.degrees(rocket.gimbal),
            gimbal_limit_deg=math.degrees(rocket.GIMBAL_LIMIT),
            ap_gimbal_limit_deg=math.degrees(guidance.attitude_max_gimbal),
            throttle=rocket.throttle if flying else 0.0,
            throttle_cmd=min(1.0, math.hypot(cmd_x, cmd_z) / a_max) if commanding else None,
            cmd_tilt_deg=math.degrees(math.atan2(cmd_x, cmd_z)) if commanding and cmd_z > 0.5 else None,
            thrust_kn=rocket.last_thrust.length() / 1000.0 if flying else 0.0,
            twr=rocket.last_thrust.length() / (rocket.mass * rocket.G0) if flying else 0.0,
            rcs=rocket.rcs_command,
            t_go=guidance.time_to_go,
            solver_status=guidance.last_status,
            backend=guidance.planner.backend,
            solves=guidance.planner.solve_count,
            burn_locked=guidance.burn_locked,
            thrust_margin=1.0 - guidance.burn_fraction,
            fps=self.clock.get_fps(),
            scenario=self.scenario_title if self.autopilot_enabled else "",
            legs=self.visuals.legs,
            nav_pad=(pad.x, pad.y),
            nav_rocket=(rocket.position.x, rocket.position.y),
            nav_trajectory=[(p.x, p.y) for p in self.trajectory[-400:]],
            nav_plan=[(p.x, p.y) for p in self.predicted_trajectory[::4]],
            touchdown=self.touchdown,
            paused=self.paused,
        )
