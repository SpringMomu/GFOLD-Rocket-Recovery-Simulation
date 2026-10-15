"""World <-> screen mapping for the side view."""

from __future__ import annotations

import math

import pygame

from .mathutil import clamp
from .rocket import Rocket


class Camera:
    """Centres on the rocket, or frames rocket and pad together in auto mode.

    The mouse wheel hands control to the user (manual zoom); toggling the
    autopilot or loading a scenario re-enables auto framing.
    """

    MAX_SCALE = 7.0
    MAX_AUTO_SCALE = 5.0
    MIN_ZOOM = 0.0001
    WHEEL_STEP = 1.18

    def __init__(self, size: tuple[int, int]) -> None:
        self.center = pygame.Vector2()
        self.screen_center = pygame.Vector2(size[0] * 0.5, size[1] * 0.5)
        self.zoom = 1.0           # manual zoom, pixels per metre
        self.scale = self.zoom    # pixels per metre actually used
        self.auto = True
        self._auto_scale: float | None = None

    def world_to_screen(self, point: pygame.Vector2) -> pygame.Vector2:
        return pygame.Vector2(
            self.screen_center.x + (point.x - self.center.x) * self.scale,
            self.screen_center.y - (point.y - self.center.y) * self.scale,
        )

    def screen_to_world(self, point: pygame.Vector2) -> pygame.Vector2:
        return pygame.Vector2(
            self.center.x + (point.x - self.screen_center.x) / self.scale,
            self.center.y - (point.y - self.screen_center.y) / self.scale,
        )

    def enable_auto(self) -> None:
        self.auto = True
        self._auto_scale = None

    def wheel(self, steps: int, autopilot_enabled: bool) -> None:
        if self.auto and autopilot_enabled:
            self.zoom = self.scale
        self.auto = False
        self.zoom = clamp(self.zoom * self.WHEEL_STEP**steps, self.MIN_ZOOM, self.MAX_SCALE)

    def update(self, size: tuple[int, int], rocket: Rocket, pad: pygame.Vector2, frame_pad: bool) -> None:
        width, height = size
        self.center = rocket.position.copy()
        self.screen_center = pygame.Vector2(width * 0.5, height * 0.5)
        full_map_scale = min(
            width / (rocket.MAP_RIGHT - rocket.MAP_LEFT),
            height / (rocket.MAP_TOP - rocket.MAP_BOTTOM),
        )
        self.scale = clamp(self.zoom, full_map_scale, self.MAX_SCALE)
        if frame_pad and self.auto:
            # Frame the vehicle and the pad together with a log-smoothed zoom.
            span_x = abs(rocket.position.x - pad.x) + 70.0
            span_y = abs(rocket.position.y - pad.y) + 70.0
            target = clamp(min(width * 0.62 / span_x, height * 0.70 / span_y), full_map_scale, self.MAX_AUTO_SCALE)
            if self._auto_scale is None:
                self._auto_scale = target
            else:
                self._auto_scale = math.exp(0.92 * math.log(self._auto_scale) + 0.08 * math.log(target))
            self.scale = self._auto_scale
            self.center = (rocket.position + pad) * 0.5
            self.screen_center = pygame.Vector2(width * 0.56, height * 0.5)
