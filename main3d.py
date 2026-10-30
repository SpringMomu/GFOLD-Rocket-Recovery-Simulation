"""3D rocket landing simulator: local server + browser (WebGL2) front end.

    python main3d.py                 start the server and open the browser
    python main3d.py --port 8765 --no-browser
    python main3d.py --test          headless regression of all scenarios (pad landings + tower catches)
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="run the headless landing regression")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    if args.test:
        from lander3d.regression import run_tests

        raise SystemExit(run_tests())
    from lander3d.server import serve

    serve(args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
