"""Real Pyodide boost-back regression, including the unpowered handover.

    python deploy/build_web.py
    python -m tools.check_browser_boostback --scenario 4 --scenario 8

Set CHROME_PATH to a Chrome binary if Playwright's Chromium is not installed.
By default serves dist/web3d locally and checks both calm and default wind.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import threading
from itertools import pairwise
from pathlib import Path

import numpy as np
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

INIT = """
const NativeWorker = window.Worker;
window.Worker = class extends NativeWorker {
    constructor(...args) {
        super(...args); window.__simWorker = this;
        this.addEventListener('message', e => {
            if (e.data.type === 'state') window.__latest = JSON.parse(e.data.json);
            if (e.data.type === 'error') window.__simError = e.data.text;
        });
    }
};
localStorage.setItem('booster3d.fold.v1', JSON.stringify({camPanel:true}));
"""

RECORD = """() => {
    self.__samples = []; self.__start = 0; self.__exit = null; self.__handover = null;
    const oldInput = self.onmessage, oldPost = self.postMessage;
    self.onmessage = function(ev) {
        if (ev.data.type === 'input' && ev.data.actions?.some(x => x.startsWith('scenario:')))
            self.__start = performance.now();
        oldInput.call(this, ev);
    };
    self.postMessage = function(msg, ...args) {
        if (msg.type === 'state' && self.__start) {
            const st = JSON.parse(msg.json);
            const previous = self.__samples.at(-1);
            self.__samples.push({ms:performance.now()-self.__start, t:st.t, phase:st.phase,
                                 pos:st.pos, vel:st.vel, tilt:st.tilt, cmd_axis:st.cmd_axis,
                                 tower:st.tower?.mode});
            if (!self.__exit && previous?.phase === 'BOOSTBACK' && st.phase !== 'BOOSTBACK') {
                const ap = glue.advance.__globals__.get('sim').autopilot;
                self.__exit = {t:st.t, reason:ap._bb_exit_reason, pos:st.pos, vel:st.vel};
            }
            if (self.__exit && !self.__handover && st.t >= self.__exit.t + 1.2 && st.throttle < .01) {
                const ap = glue.advance.__globals__.get('sim').autopilot;
                const miss = ap._burn_miss_now;
                self.__handover = {t:st.t, phase:st.phase, miss:Array.from(miss.toJs()), pos:st.pos, vel:st.vel};
                miss.destroy();
            }
        }
        return oldPost.call(this, msg, ...args);
    };
}"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def check(browser, url, index, wind, overlay, out, wind_from=None, wind_speed=None, miss_limit=100.0):
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.add_init_script(INIT)
    if overlay:
        source = overlay.read_text(encoding="utf-8")
        page.route("**/py/lander3d/autopilot.py",
                   lambda route, request: route.fulfill(body=source, content_type="text/plain"))
    try:
        page.goto(url.rstrip("/") + "/?local&cfd=low", wait_until="domcontentloaded")
        page.wait_for_function("window.__latest || window.__simError", timeout=90000)
        if page.evaluate("window.__simError"):
            raise RuntimeError(page.evaluate("window.__simError"))
        worker = next(w for w in page.workers if "sim_worker.js" in w.url)
        worker.evaluate(RECORD)
        actions = [f"scenario:{index - 1}"]
        if bool(page.evaluate("window.__latest.wind_on")) != wind:
            actions.append("wind")
        if wind_from is not None or wind_speed is not None:
            direction = 270 if wind_from is None else wind_from
            speed = 6 if wind_speed is None else wind_speed
            actions.append("windcfg:" + json.dumps({"from": [direction] * 2, "speed": [speed] * 2, "shear": 1}))
        page.evaluate("actions => window.__simWorker.postMessage({type:'input', keys:{}, actions})", actions)
        timed_out = False
        try:
            page.wait_for_function("['LANDED','CAUGHT','CRASHED'].includes(window.__latest?.state) || window.__simError",
                                   timeout=180000)
        except PlaywrightTimeoutError:
            timed_out = True
        st = page.evaluate("window.__latest")
        recording = worker.evaluate("({samples:self.__samples, exit:self.__exit, handover:self.__handover})")
        td = st.get("touchdown") or {}
        catch = index == 8
        handover = recording["handover"]
        miss = float(np.linalg.norm(handover["miss"])) if handover else None
        ok = (st["state"] == ("CAUGHT" if catch else "LANDED") and td.get("error", 999) <= (1.0 if catch else 3.5)
              and abs(td.get("vz", 99)) <= 2.5 and td.get("vh", 99) <= 1.5 and td.get("tilt", 99) <= 3.0
              and handover is not None and handover["phase"] == "COAST" and miss < miss_limit
              and recording["exit"]["reason"] == "impulse-complete" and not errors
              and not page.evaluate("window.__simError"))
        boost = [r for r in recording["samples"] if r["phase"] == "BOOSTBACK"]
        gaps = [b["ms"] - a["ms"] for a, b in pairwise(boost)]
        result = {"scenario": index, "wind": wind, "pass": bool(ok), "state": st["state"], "timeout": timed_out,
                  "crash": st.get("crash"), "touchdown": td,
                  "exit": recording["exit"], "handover": handover, "handover_miss": miss,
                  "boost_p95_ms": float(np.percentile(gaps, 95)) if gaps else None,
                  "boost_max_ms": max(gaps, default=None), "errors": errors}
        name = f"scenario-{index}-{'wind' if wind else 'calm'}"
        page.screenshot(path=str(out / f"{name}.png"))
        (out / f"{name}.json").write_text(json.dumps({"result": result, "recording": recording}, indent=2))
        return result
    finally:
        page.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="otherwise serve the local static build")
    parser.add_argument("--scenario", type=int, choices=[4, 8], action="append")
    parser.add_argument("--wind", choices=["off", "on", "both"], default="both")
    parser.add_argument("--wind-from", type=float)
    parser.add_argument("--wind-speed", type=float)
    parser.add_argument("--miss-limit", type=float, default=100.0)
    parser.add_argument("--overlay-ap", type=Path, help="validate a deployment merge with its existing settings")
    parser.add_argument("--out", type=Path, default=ROOT / "tools/out/browser-boostback")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    server = None
    if not args.url:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                 functools.partial(QuietHandler, directory=str(ROOT / "dist/web3d")))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        args.url = f"http://127.0.0.1:{server.server_port}"
    rows = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=os.environ.get("CHROME_PATH"), headless=True,
                      args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            try:
                for index in args.scenario or [4, 8]:
                    for wind in ([False, True] if args.wind == "both" else [args.wind == "on"]):
                        print(f"Running scenario {index}, wind={wind}", flush=True)
                        rows.append(check(browser, args.url, index, wind, args.overlay_ap, args.out,
                                          args.wind_from, args.wind_speed, args.miss_limit))
                        print(json.dumps(rows[-1]), flush=True)
                        (args.out / "results.json").write_text(json.dumps(rows, indent=2))
            finally:
                browser.close()
    finally:
        if server:
            server.shutdown()
            server.server_close()
    return int(not all(row["pass"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
