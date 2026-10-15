"""Small scalar helpers used across the 2D package."""

from __future__ import annotations

import math


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def move_towards(value: float, target: float, max_delta: float) -> float:
    """Move ``value`` toward ``target`` by at most ``max_delta``."""
    if value < target:
        return min(value + max_delta, target)
    return max(value - max_delta, target)


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def move_angle(value: float, target: float, max_delta: float) -> float:
    """Like :func:`move_towards`, along the shorter way around the circle."""
    delta = wrap_angle(target - value)
    if abs(delta) <= max_delta:
        return target
    return value + math.copysign(max_delta, delta)
