"""Visual QA of the browser CFD view with a headless Chromium (needs ``playwright``).

    python main3d.py --port 8766 --no-browser &
    python -m tools.check_airflow [--url http://127.0.0.1:8766] [--out DIR]

Loads scenario 4 paused, serves that fixed state to the page, lets the fluid
solver run (``?cfd=low`` for software GL) and saves a screenshot per flow
display mode (default, vorticity, speed, dye) to DIR (default ``tools/out``).
Set ``CHROME_PATH`` to use a specific Chrome binary instead of Playwright's.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

GL_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--disable-web-security"]
MODES = ["airflow-vorticity", "airflow-speed", "airflow-dye"]   # order of the B key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("out"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    request = Request(args.url + "/api/input", data=json.dumps({"actions": ["scenario:3", "pause"]}).encode(),
                      headers={"Content-Type": "application/json"})
    urlopen(request).read()
    state = json.load(urlopen(args.url + "/api/state"))
    state["paused"] = False
    state["cfd_forces"] = False

    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROME_PATH"), headless=True, args=GL_ARGS)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.route("**/api/state", lambda r: r.fulfill(json=state))
        page.route("**/api/cfd3d", lambda r: r.fulfill(json={"ok": True}))
        page.goto(args.url + "/?cfd=low")
        page.wait_for_function("window.__app?.fluid?.steps > 130", timeout=180000)
        page.screenshot(path=str(args.out / "airflow-default.png"))
        print(json.dumps(page.evaluate(
            "({steps: __app.fluid.steps, ok: __app.fluid.ok, glError: __app.fluid.gl.getError(),"
            " renderer: __app.fluid.gl.getParameter(__app.fluid.gl.RENDERER)})")), flush=True)
        for name in MODES:
            page.keyboard.press("b")
            page.wait_for_function("(start) => __app.fluid.steps > start + 9", arg=page.evaluate("__app.fluid.steps"),
                                   timeout=30000)
            page.screenshot(path=str(args.out / f"{name}.png"))
        browser.close()
    (args.out / "airflow-browser-check.json").write_text(json.dumps({"errors": errors}, indent=2), encoding="utf-8")
    print(json.dumps({"errors": errors}, ensure_ascii=False), flush=True)
    assert not errors


if __name__ == "__main__":
    main()
