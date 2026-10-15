"""Modal editor for the initial flight state (opened with T).

Implemented in pygame rather than a platform dialog so it also works in
headless runs.
"""

from __future__ import annotations

import math

import pygame

from .cockpit import Cockpit
from .rocket import Rocket

FIELD_LABELS = (
    "ALTITUDE (m, >= 0)",
    "HORIZONTAL POSITION (m, +/-)",
    "HORIZONTAL SPEED (m/s, +/-)",
)


class StateDialog:
    def __init__(self) -> None:
        self.active = False
        self.fields = ["", "", ""]
        self.index = 0
        self.error = ""
        self._font_small = pygame.font.SysFont("consolas", 15)
        self._font_medium = pygame.font.SysFont("consolas", 18)

    def open(self) -> None:
        self.active = True
        self.fields = ["", "", ""]
        self.index = 0
        self.error = ""

    def close(self) -> None:
        self.active = False
        self.error = ""

    def handle_event(self, event: pygame.event.Event, rocket: Rocket) -> tuple[float, float, float] | None:
        """Process a key press; return (altitude, x, vx) once the user applies valid values."""
        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_ESCAPE:
            self.close()
        elif event.key in (pygame.K_TAB, pygame.K_DOWN):
            self.index = (self.index + 1) % 3
        elif event.key == pygame.K_UP:
            self.index = (self.index - 1) % 3
        elif event.key == pygame.K_BACKSPACE:
            self.fields[self.index] = self.fields[self.index][:-1]
            self.error = ""
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return self._validate(rocket)
        else:
            # KEYDOWN.unicode covers digits, decimal point and signs on any layout.
            character = getattr(event, "unicode", "")
            if character and (character.isdigit() or character in ".+-eE"):
                self.fields[self.index] += character
                self.error = ""
        return None

    def _validate(self, rocket: Rocket) -> tuple[float, float, float] | None:
        try:
            altitude, position, speed = (float(value.strip()) for value in self.fields)
        except ValueError:
            self.error = "Enter three valid numbers (use +/- for signed values)."
            return None
        if not all(math.isfinite(value) for value in (altitude, position, speed)):
            self.error = "Values must be finite numbers."
            return None
        if altitude < 0.0:
            self.error = "Altitude must be zero or greater."
            return None
        max_altitude = max(0.0, rocket.MAP_TOP - rocket.LENGTH * 0.5 - 1.0)
        if altitude > max_altitude:
            self.error = f"Altitude must stay within 0..{max_altitude:.0f} m."
            return None
        if not rocket.MAP_LEFT <= rocket.launch_x + position <= rocket.MAP_RIGHT:
            self.error = f"Horizontal position must stay within {rocket.MAP_LEFT:.0f}..{rocket.MAP_RIGHT:.0f} m."
            return None
        self.close()
        return altitude, position, speed

    def draw(self, screen: pygame.Surface, cockpit: Cockpit) -> None:
        width, height = screen.get_size()
        dimmer = pygame.Surface((width, height), pygame.SRCALPHA)
        dimmer.fill((0, 0, 0, 150))
        screen.blit(dimmer, (0, 0))

        panel_width = min(570, max(360, width - 32))
        panel_height = 314
        panel = pygame.Rect((width - panel_width) // 2, max(16, (height - panel_height) // 2), panel_width, panel_height)
        cockpit.panel(screen, panel, "SET FLIGHT STATE  -  INITIAL CONDITIONS", (40, 214, 255), alpha=240)

        left = panel.left + 22
        for i, (label, value) in enumerate(zip(FIELD_LABELS, self.fields)):
            y = panel.top + 62 + i * 62
            active = i == self.index
            self._text(screen, label, self._font_small, (215, 231, 240) if active else (155, 181, 196), (left, y))
            box = pygame.Rect(left, y + 20, panel.width - 44, 32)
            pygame.draw.rect(screen, (15, 29, 44), box, border_radius=3)
            pygame.draw.rect(screen, (111, 218, 255) if active else (74, 105, 124), box, 2 if active else 1,
                             border_radius=3)
            self._text(screen, value, self._font_medium, (240, 246, 249), (box.left + 9, box.top + 5))
            if active and pygame.time.get_ticks() % 1000 < 550:
                caret_x = box.left + 9 + self._font_medium.size(value)[0]
                pygame.draw.line(screen, (240, 246, 249), (caret_x, box.top + 6), (caret_x, box.bottom - 6), 1)

        footer = (self.error, (255, 124, 108)) if self.error else (
            "TAB / UP / DOWN: FIELD     ENTER: APPLY     ESC: CANCEL", (171, 199, 213))
        self._text(screen, footer[0], self._font_small, footer[1], (left, panel.bottom - 38))

    @staticmethod
    def _text(screen: pygame.Surface, text: str, font: pygame.font.Font, color, position) -> None:
        screen.blit(font.render(text, True, color), position)
