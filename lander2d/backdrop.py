"""Sky gradient, stars and the curved ground line."""

from __future__ import annotations

import math
import random

import numpy as np
import pygame

from .camera import Camera
from .mathutil import clamp
from .rocket import Rocket

SKY_ALTITUDES = np.array([0.0, 600.0, 2_500.0, 8_000.0, 20_000.0, 45_000.0, 85_000.0, 120_000.0])
SKY_COLORS = np.array([
    (176, 208, 228), (128, 184, 226), (88, 150, 212), (50, 106, 184),
    (22, 52, 112), (8, 20, 54), (3, 8, 24), (1, 3, 11),
], dtype=np.float64)


class Backdrop:
    def __init__(self, seed: int = 17) -> None:
        rng = random.Random(seed)
        # (x, y) as fractions of the screen, and the star radius in pixels.
        self.stars = [(rng.random(), rng.random(), rng.choice((1, 1, 1, 2))) for _ in range(110)]
        self._sky: pygame.Surface | None = None
        self._sky_center = pygame.Vector2(math.inf, math.inf)
        self._sky_scale = -1.0

    def draw(self, screen: pygame.Surface, camera: Camera, rocket: Rocket) -> None:
        self._draw_sky(screen, camera, rocket)
        self._draw_ground(screen, camera, rocket)

    # -- sky ---------------------------------------------------------------------
    def _draw_sky(self, screen: pygame.Surface, camera: Camera, rocket: Rocket) -> None:
        width, height = screen.get_size()
        centre_shift = math.inf
        if math.isfinite(self._sky_center.x):
            centre_shift = (camera.center - self._sky_center).length() * camera.scale
        scale_change = math.inf
        if self._sky_scale > 0.0:
            scale_change = abs(math.log(camera.scale / self._sky_scale))
        if (self._sky is None or self._sky.get_size() != (width, height)
                or centre_shift >= 3.0 or scale_change >= 0.018):
            self._sky = self._render_sky(width, height, camera, rocket)
            self._sky_center = camera.center.copy()
            self._sky_scale = camera.scale
        screen.blit(self._sky, (0, 0))

        # Stars fade in through the upper atmosphere; they are placed in screen
        # space so the backdrop stays stable while zooming.
        for x_ratio, y_ratio, radius in self.stars:
            position = pygame.Vector2(x_ratio * width, y_ratio * height)
            altitude = rocket.altitude_at(camera.screen_to_world(position))
            visibility = clamp((altitude - 18_000.0) / 65_000.0, 0.0, 1.0)
            if visibility <= 0.02:
                continue
            brightness = round(90 + 165 * visibility)
            pygame.draw.circle(screen, (brightness, brightness, min(255, brightness + 18)), position, radius)

    @staticmethod
    def _render_sky(width: int, height: int, camera: Camera, rocket: Rocket) -> pygame.Surface:
        """Colour every sample by its radial altitude, at 1/3 resolution."""
        reduction = 3
        low_width = max(1, math.ceil(width / reduction))
        low_height = max(1, math.ceil(height / reduction))
        screen_x = (np.arange(low_width, dtype=np.float64) + 0.5) * (width / low_width)
        screen_y = (np.arange(low_height, dtype=np.float64) + 0.5) * (height / low_height)
        world_x = camera.center.x + (screen_x - camera.screen_center.x) / camera.scale
        world_y = camera.center.y - (screen_y - camera.screen_center.y) / camera.scale
        radial_y = world_y + rocket.EARTH_RADIUS
        altitude = np.sqrt(world_x[:, np.newaxis] ** 2 + radial_y[np.newaxis, :] ** 2) - rocket.EARTH_RADIUS

        clipped = np.clip(altitude, SKY_ALTITUDES[0], SKY_ALTITUDES[-1])
        pixels = np.empty((low_width, low_height, 3), dtype=np.uint8)
        for channel in range(3):
            pixels[:, :, channel] = np.interp(clipped, SKY_ALTITUDES, SKY_COLORS[:, channel]).astype(np.uint8)
        low_surface = pygame.surfarray.make_surface(pixels)
        return pygame.transform.smoothscale(low_surface, (width, height)).convert()

    # -- ground ------------------------------------------------------------------
    @staticmethod
    def _draw_ground(screen: pygame.Surface, camera: Camera, rocket: Rocket) -> None:
        width, height = screen.get_size()
        step = 3
        surface_points = []
        for screen_x in range(-step, width + step * 2, step):
            world_x = camera.screen_to_world(pygame.Vector2(screen_x, 0.0)).x
            world_x = clamp(world_x, -rocket.EARTH_RADIUS, rocket.EARTH_RADIUS)
            surface_points.append(camera.world_to_screen(pygame.Vector2(world_x, rocket.surface_y(world_x))))

        earth = surface_points + [
            pygame.Vector2(width + step, height + step),
            pygame.Vector2(-step, height + step),
        ]
        pygame.draw.polygon(screen, (40, 44, 38), earth)
        pygame.draw.lines(screen, (84, 98, 74), False, surface_points, 3)
        pygame.draw.aalines(screen, (128, 146, 108), False, surface_points)
        if camera.scale < 0.02:
            pygame.draw.aalines(screen, (92, 164, 196), False, surface_points)

        # Tick marks at a round spacing make scale and curvature readable.
        spacing = _round_spacing(95.0 / camera.scale)
        left = max(camera.screen_to_world(pygame.Vector2(0.0, height * 0.5)).x, rocket.MAP_LEFT)
        right = min(camera.screen_to_world(pygame.Vector2(width, height * 0.5)).x, rocket.MAP_RIGHT)
        marker_x = math.floor(left / spacing) * spacing
        while marker_x <= right + spacing:
            if abs(marker_x) < rocket.EARTH_RADIUS:
                surface = pygame.Vector2(marker_x, rocket.surface_y(marker_x))
                base = camera.world_to_screen(surface)
                tip = camera.world_to_screen(surface + rocket.local_up(surface) * (8.0 / camera.scale))
                pygame.draw.line(screen, (96, 108, 86), base, tip, 1)
            marker_x += spacing


def _round_spacing(value: float) -> float:
    """Round up to 1, 2, 5 or 10 times a power of ten."""
    value = max(value, 1.0)
    exponent = 10.0 ** math.floor(math.log10(value))
    fraction = value / exponent
    for nice in (1.0, 2.0, 5.0):
        if fraction <= nice:
            return nice * exponent
    return 10.0 * exponent
