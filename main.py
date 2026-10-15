"""2D rocket landing simulator.

    python main.py                  play (window)
    python main.py --landing-test   headless regression of the five preset landings
    python main.py --record DIR     render the presets to video (or JPEG frames) in DIR
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--landing-test", action="store_true", help="run the headless landing regression")
    parser.add_argument("--record", metavar="DIR", nargs="?", const="videos", help="record the presets into DIR")
    args = parser.parse_args()

    if args.landing_test or os.environ.get("ROCKET_LANDING_TEST", "").lower() in {"1", "true", "yes"}:
        from lander2d.scenarios import run_landing_tests

        raise SystemExit(run_landing_tests())
    if args.record is not None:
        from lander2d.recording import record_presets

        record_presets(args.record)
        return

    from lander2d.game import RocketSimulator

    frames = os.environ.get("ROCKET_SIM_TEST_FRAMES")
    RocketSimulator().run(max_frames=int(frames) if frames else None)


if __name__ == "__main__":
    main()
