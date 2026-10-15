"""Planar rigid-body rocket: gravity on a round Earth, drag, TVC and RCS.

SI units throughout.  Coordinates are the launch-pad tangent plane with the
Earth's centre at ``(0, -EARTH_RADIUS)``; "local up" is the radial direction.
The vehicle has infinite propellant (constant wet mass).
"""

from __future__ import annotations

import math

import pygame

from .mathutil import clamp, move_towards, wrap_angle


class Rocket:
    # Flight region.
    EARTH_RADIUS = 6_371_000.0
    MAP_LEFT = -2_000_000.0
    MAP_RIGHT = 2_000_000.0
    MAP_BOTTOM = -400_000.0
    MAP_TOP = 500_000.0
    ATMOSPHERE_TOP = 120_000.0

    # Geometry and mass.
    LENGTH = 18.0
    WIDTH = 3.0
    DRY_MASS = 8_000.0
    FUEL_CAPACITY = 12_000.0

    # Engine: a finite throttle response keeps manual starts and cutoffs readable.
    MAX_THRUST = 360_000.0
    GIMBAL_LIMIT = math.radians(30.0)
    GIMBAL_RATE = math.radians(42.0)
    THROTTLE_RATE = 1.20
    THRUST_LEVER_ARM = 7.4
    # Cold-gas RCS, still available with the engine off; sized for roughly
    # 40 deg/s^2 on the nominal 20 t vehicle.
    RCS_MAX_TORQUE = 1_300_000.0

    # Environment and aerodynamics.
    G0 = 9.80665
    SEA_LEVEL_DENSITY = 1.225
    ATMOSPHERE_SCALE_HEIGHT = 8_500.0
    DRAG_COEFFICIENT = 0.38
    FRONTAL_AREA = math.pi * (WIDTH * 0.5) ** 2
    SIDE_AREA = LENGTH * WIDTH

    # Touchdown counts as LANDED only inside this envelope.
    LANDING_PAD_RADIUS = 8.0
    LANDING_MAX_SPEED = 4.0
    LANDING_MAX_TILT = math.radians(12.0)
    LANDING_MAX_RATE = math.radians(20.0)

    def __init__(self) -> None:
        self.launch_x = 0.0
        self.pad_center_height = self.LENGTH * 0.5 + 0.04
        self.reset()

    def reset(self) -> None:
        """Park the vehicle upright on the launch pad."""
        self.position = self.pad_position()
        self.velocity = pygame.Vector2()
        up = self.local_up()
        self.angle = math.atan2(up.x, up.y)
        self.angular_velocity = 0.0
        self.throttle = 0.0
        self.gimbal = 0.0  # positive moves the nozzle to the right
        self.rcs_command = 0.0
        self.state = "ON PAD"
        self.crash_reason = ""
        self.last_drag = pygame.Vector2()
        self.last_thrust = pygame.Vector2()

    # -- derived quantities ----------------------------------------------------
    @property
    def mass(self) -> float:
        return self.DRY_MASS + self.FUEL_CAPACITY

    @property
    def altitude(self) -> float:
        """Height of the vehicle above the pad surface (0 when standing on it)."""
        return max(0.0, self.geometric_altitude - self.pad_center_height)

    @property
    def geometric_altitude(self) -> float:
        return (self.position - self.earth_center()).length() - self.EARTH_RADIUS

    @property
    def vertical_speed(self) -> float:
        return self.velocity.dot(self.local_up())

    @property
    def horizontal_speed(self) -> float:
        return self.velocity.dot(self.local_right())

    @property
    def local_pitch(self) -> float:
        """Angle of the body axis from local vertical, positive to the right."""
        up = self.local_up()
        return wrap_angle(self.angle - math.atan2(up.x, up.y))

    @property
    def moment_of_inertia(self) -> float:
        """Uniform-body approximation about the centre of mass."""
        return self.mass * (self.LENGTH**2 + self.WIDTH**2) / 12.0

    @property
    def engine_running(self) -> bool:
        return self.throttle > 0.001 and self.state not in {"CRASHED", "LANDED"}

    # -- geometry --------------------------------------------------------------
    @classmethod
    def earth_center(cls) -> pygame.Vector2:
        return pygame.Vector2(0.0, -cls.EARTH_RADIUS)

    @classmethod
    def surface_y(cls, x: float, altitude: float = 0.0) -> float:
        radius = cls.EARTH_RADIUS + altitude
        inside = max(0.0, radius * radius - x * x)
        return -cls.EARTH_RADIUS + math.sqrt(inside)

    def world_y_for_altitude(self, x: float, altitude: float) -> float:
        return self.surface_y(x, self.pad_center_height + altitude)

    def local_up(self, point: pygame.Vector2 | None = None) -> pygame.Vector2:
        radial = (point or self.position) - self.earth_center()
        if radial.length_squared() <= 1e-12:
            return pygame.Vector2(0.0, 1.0)
        return radial.normalize()

    def local_right(self, point: pygame.Vector2 | None = None) -> pygame.Vector2:
        up = self.local_up(point)
        return pygame.Vector2(up.y, -up.x)

    def pad_position(self) -> pygame.Vector2:
        """Centre of the landing pad surface (where the vehicle's centre sits when parked)."""
        return pygame.Vector2(self.launch_x, self.world_y_for_altitude(self.launch_x, 0.0))

    def altitude_at(self, point: pygame.Vector2) -> float:
        return (point - self.earth_center()).length() - self.EARTH_RADIUS

    def body_up(self) -> pygame.Vector2:
        """Unit vector along the body axis, engine to nose."""
        return pygame.Vector2(math.sin(self.angle), math.cos(self.angle))

    def local_to_world(self, local_x: float, local_y: float) -> pygame.Vector2:
        """Body frame (+y engine to nose, +x the vehicle's right) to world."""
        right = pygame.Vector2(math.cos(self.angle), -math.sin(self.angle))
        return self.position + right * local_x + self.body_up() * local_y

    def outline(self) -> list[pygame.Vector2]:
        half_width = self.WIDTH * 0.5
        half_length = self.LENGTH * 0.5
        return [
            self.local_to_world(0.0, half_length),
            self.local_to_world(half_width, half_length - 3.0),
            self.local_to_world(half_width, -half_length),
            self.local_to_world(-half_width, -half_length),
            self.local_to_world(-half_width, half_length - 3.0),
        ]

    def collision_points(self) -> list[pygame.Vector2]:
        """Hull outline plus the tips of the landing fins."""
        half_width = self.WIDTH * 0.5
        half_length = self.LENGTH * 0.5
        points = self.outline()
        points.append(self.local_to_world(-half_width - 1.45, -half_length - 0.1))
        points.append(self.local_to_world(half_width + 1.45, -half_length - 0.1))
        return points

    # -- control and state -----------------------------------------------------
    def apply_input(self, keys: pygame.key.ScancodeWrapper, dt: float) -> None:
        """Manual flight: W/S (or up/down) throttle, A/D (or left/right) gimbal."""
        if self.state in {"CRASHED", "LANDED"}:
            return
        throttle_axis = float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(keys[pygame.K_s] or keys[pygame.K_DOWN])
        self.throttle = clamp(self.throttle + throttle_axis * self.THROTTLE_RATE * dt, 0.0, 1.0)
        gimbal_axis = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT])
        self.gimbal = move_towards(self.gimbal, gimbal_axis * self.GIMBAL_LIMIT, self.GIMBAL_RATE * dt)

    def set_flight_state(self, altitude: float, horizontal_position: float, horizontal_speed: float) -> None:
        """Place the vehicle upright at the given height, offset and horizontal speed."""
        world_x = self.launch_x + horizontal_position
        self.position = pygame.Vector2(world_x, self.world_y_for_altitude(world_x, altitude))
        up = self.local_up()
        self.velocity = self.local_right() * horizontal_speed
        self.angle = math.atan2(up.x, up.y)
        self.angular_velocity = 0.0
        self.throttle = 0.0
        self.gimbal = 0.0
        self.rcs_command = 0.0
        on_pad = altitude <= 1e-9 and abs(horizontal_position) <= 1e-9 and abs(horizontal_speed) <= 1e-9
        self.state = "ON PAD" if on_pad else "FLYING"
        self.crash_reason = ""
        self.last_drag = pygame.Vector2()
        self.last_thrust = pygame.Vector2()

    # -- dynamics --------------------------------------------------------------
    def step(self, dt: float) -> None:
        if self.state in {"CRASHED", "LANDED"}:
            return

        mass = self.mass
        thrust = self.MAX_THRUST * self.throttle

        # The nozzle deflects with the gimbal; thrust acts the opposite way.
        thrust_angle = self.angle - self.gimbal
        thrust_force = pygame.Vector2(math.sin(thrust_angle), math.cos(thrust_angle)) * thrust
        local_up = self.local_up()
        gravity = self.G0 * (
            self.EARTH_RADIUS / max(self.EARTH_RADIUS, self.EARTH_RADIUS + self.geometric_altitude)
        ) ** 2
        gravity_force = -local_up * mass * gravity

        # Drag interpolates between the end-on and broadside reference areas.
        air_altitude = max(0.0, self.geometric_altitude)
        density = 0.0
        if air_altitude < self.ATMOSPHERE_TOP:
            density = self.SEA_LEVEL_DENSITY * math.exp(-air_altitude / self.ATMOSPHERE_SCALE_HEIGHT)
        drag_force = pygame.Vector2()
        speed = self.velocity.length()
        if speed > 0.01:
            direction = self.velocity / speed
            body_up = self.body_up()
            area = self.FRONTAL_AREA * abs(direction.dot(body_up)) + self.SIDE_AREA * abs(direction.cross(body_up))
            drag_force = -direction * (0.5 * density * self.DRAG_COEFFICIENT * area * speed * speed)

        # The engine sits below the centre of mass, so a positive gimbal (thrust
        # pushed left) gives a negative, left-rotating torque.  The atmosphere
        # adds no pitch torque or damping.
        gimbal_torque = -self.THRUST_LEVER_ARM * thrust * math.sin(self.gimbal)
        rcs_torque = self.rcs_command * self.RCS_MAX_TORQUE if self.state == "FLYING" else 0.0

        acceleration = (thrust_force + gravity_force + drag_force) / mass
        angular_acceleration = (gimbal_torque + rcs_torque) / self.moment_of_inertia
        self.last_drag = drag_force
        self.last_thrust = thrust_force

        # Held on the pad until thrust exceeds weight.
        if self.state == "ON PAD":
            if thrust_force.dot(local_up) <= mass * gravity * 1.001:
                self.velocity.update(0.0, 0.0)
                self.angular_velocity = 0.0
                self.position = self.earth_center() + local_up * (self.EARTH_RADIUS + self.pad_center_height)
                self.angle = math.atan2(local_up.x, local_up.y)
                return
            self.state = "FLYING"

        # Semi-implicit Euler at the fixed 120 Hz step.
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        self.angular_velocity += angular_acceleration * dt
        self.angle = wrap_angle(self.angle + self.angular_velocity * dt)

        self._resolve_ground_contact()
        if self.state != "CRASHED":
            self._resolve_map_boundaries()

    def _stop(self) -> None:
        self.velocity.update(0.0, 0.0)
        self.angular_velocity = 0.0
        self.throttle = 0.0

    def _resolve_ground_contact(self) -> None:
        points = self.collision_points()
        contact_point = min(points, key=self.altitude_at)
        lowest_altitude = self.altitude_at(contact_point)
        local_up = self.local_up(contact_point)
        if lowest_altitude > 0.0 or self.velocity.dot(local_up) > 0.0:
            return

        self.position += local_up * -lowest_altitude
        soft_touchdown = (
            abs(self.position.x - self.launch_x) <= self.LANDING_PAD_RADIUS
            and abs(self.horizontal_speed) <= self.LANDING_MAX_SPEED
            and abs(self.vertical_speed) <= self.LANDING_MAX_SPEED
            and abs(self.local_pitch) <= self.LANDING_MAX_TILT
            and abs(self.angular_velocity) <= self.LANDING_MAX_RATE
        )
        self._stop()
        self.gimbal = 0.0
        self.rcs_command = 0.0
        if soft_touchdown:
            self.state = "LANDED"
            self.crash_reason = ""
        else:
            self.state = "CRASHED"
            self.crash_reason = "GROUND IMPACT"

    def _resolve_map_boundaries(self) -> None:
        points = self.collision_points()
        min_x = min(point.x for point in points)
        max_x = max(point.x for point in points)
        highest_altitude = max(self.altitude_at(point) for point in points)
        hit_boundary = False

        if min_x < self.MAP_LEFT:
            self.position.x += self.MAP_LEFT - min_x
            hit_boundary = True
        if max_x > self.MAP_RIGHT:
            self.position.x -= max_x - self.MAP_RIGHT
            hit_boundary = True
        if highest_altitude > self.MAP_TOP:
            self.position -= self.local_up() * (highest_altitude - self.MAP_TOP)
            hit_boundary = True

        if hit_boundary:
            self._stop()
            self.state = "CRASHED"
            self.crash_reason = "MAP BOUNDARY IMPACT"
