"""Offscreen video capture of the preset landings with the game's own renderer."""

from __future__ import annotations

import json
import math
import os

import pygame

from .game import RocketSimulator
from .scenarios import FIXED_DT, PHYSICS_HZ, PRESETS

SETTLE_SECONDS = 2.5       # keep recording after touchdown


def record_landing_video(path: str, title: str, initial_altitude: float, horizontal_position: float = 0.0,
                         horizontal_speed: float = 0.0, vertical_speed: float = 0.0, *,
                         fps: int = 30, max_seconds: float = 150.0) -> dict:
    """Render one autonomous landing to ``path`` (MP4 via imageio-ffmpeg, else JPEG frames).

    Physics runs at the normal fixed step; one frame is drawn every
    ``PHYSICS_HZ / fps`` steps.  A telemetry CSV is written next to the video.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    simulator = RocketSimulator()
    simulator.show_help = False
    simulator.autopilot_enabled = True
    simulator.scenario_title = title
    simulator.start_flight(initial_altitude, horizontal_position, horizontal_speed, vertical_speed)
    rocket = simulator.rocket

    width, height = simulator.screen.get_size()
    writer = None
    frame_dir = None
    try:
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            path, (width, height), fps=fps, codec="libx264", quality=7,
            pix_fmt_out="yuv420p", macro_block_size=8,
        )
        writer.send(None)
    except (ImportError, RuntimeError, OSError):  # no usable encoder
        frame_dir = os.path.splitext(path)[0] + "_frames"
        os.makedirs(frame_dir, exist_ok=True)

    steps_per_frame = max(1, PHYSICS_HZ // fps)
    total_steps = int(max_seconds * PHYSICS_HZ)
    settle_frames = int(SETTLE_SECONDS * fps)
    to_bytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
    telemetry = []
    frame = step = 0
    while step < total_steps and settle_frames > 0:
        for _ in range(steps_per_frame):
            if rocket.state not in {"LANDED", "CRASHED"}:
                simulator.advance_physics()
            step += 1
        if rocket.state in {"LANDED", "CRASHED"}:
            settle_frames -= 1
            simulator.predicted_trajectory = []
        simulator.visuals.update(1.0 / fps, rocket)
        simulator.update_camera()
        simulator.draw()
        if writer is not None:
            writer.send(to_bytes(simulator.screen, "RGB"))
        else:
            pygame.image.save(simulator.screen, os.path.join(frame_dir, f"f{frame:05d}.jpg"))
        frame += 1
        telemetry.append(
            f"{step * FIXED_DT:.3f},{simulator.guidance.phase},{rocket.position.x - rocket.launch_x:.3f},"
            f"{rocket.altitude:.3f},{rocket.horizontal_speed:.3f},{rocket.vertical_speed:.3f},"
            f"{rocket.throttle:.3f},{math.degrees(rocket.local_pitch):.2f},{math.degrees(rocket.gimbal):.2f}"
        )
    if writer is not None:
        writer.close()
    with open(os.path.splitext(path)[0] + "_telemetry.csv", "w", encoding="utf-8") as handle:
        handle.write("t,phase,x,h,vx,vz,throttle,pitch_deg,gimbal_deg\n" + "\n".join(telemetry) + "\n")
    pygame.quit()
    return {
        "video": path if frame_dir is None else frame_dir,
        "frames": frame,
        "state": rocket.state,
        "pad_error_m": round(abs(rocket.position.x - rocket.launch_x), 3),
        "backend": simulator.guidance.planner.backend,
    }


def record_presets(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, params in PRESETS:
        info = record_landing_video(os.path.join(out_dir, name + ".mp4"), name.replace("_", " "), **params)
        print(json.dumps({"scenario": name, **info}), flush=True)
