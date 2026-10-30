// On-screen controls: the menu panel (scenario selector + the keyboard
// commands as buttons) and the touch flight pad (virtual stick = W/A/S/D,
// roll, throttle, autopilot / SAS / legs).  Both only send the same inputs as
// the keyboard; the state they show is read from the snapshot.
import * as ZH from "./zh.js";

const $ = (id) => document.getElementById(id);
// Same order as lander3d/sim.py SCENARIOS (action "scenario:<index>").
const SCENARIOS = ["HOVER-SLAM 100 m", "2000 m / +30 m/s UP", "2000 m / -30 m/s DOWN", "BOOSTBACK +1000 m / 100 m/s",
  "DIVERT -1000 m / 3-D", "ENTRY 12 km / 280 m/s"];
const PAD_KEY = "booster3d.pad.v1";

export function initControls({ press, command, act }) {
  // ---- touch pad visibility: on by default on touch screens, remembered.
  const touch = window.matchMedia("(pointer: coarse)").matches || navigator.maxTouchPoints > 0;
  let padOn = touch;
  try { const v = localStorage.getItem(PAD_KEY); if (v !== null) padOn = v === "1"; } catch (e) { /* ignore */ }
  const setPad = (on) => {
    padOn = on;
    document.body.classList.toggle("pad", on);
    try { localStorage.setItem(PAD_KEY, on ? "1" : "0"); } catch (e) { /* ignore */ }
  };
  document.body.classList.toggle("pad", padOn);

  // ---- scenario selector
  const sel = $("scenSel");
  const opt = (value, text) => { const o = document.createElement("option"); o.value = value; o.textContent = text; sel.appendChild(o); };
  opt("", "切换场景…");
  SCENARIOS.forEach((name, i) => opt(String(i), `${i + 1}  ${ZH.scenario(name)}`));
  opt("pad", "发射台 · 手动起飞");
  sel.addEventListener("change", () => {
    const v = sel.value;
    if (v === "pad") act("pad"); else if (v !== "") act("scenario:" + v);
    sel.value = "";
    sel.blur();
  });

  // ---- buttons with a one-shot command (same names as the keys)
  for (const b of document.querySelectorAll("#toolbar [data-cmd], #pad [data-cmd]")) {
    b.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const k = b.dataset.cmd;
      if (k === "pad") setPad(!padOn); else command(k);
      b.blur();
    });
  }

  // ---- hold buttons (roll, throttle up / down)
  const hold = (el, key) => {
    const on = (ev) => { ev.preventDefault(); el.setPointerCapture(ev.pointerId); el.classList.add("held"); press(key, true); };
    const off = () => { el.classList.remove("held"); press(key, false); };
    el.addEventListener("pointerdown", on);
    el.addEventListener("pointerup", off);
    el.addEventListener("pointercancel", off);
    el.addEventListener("lostpointercapture", off);
  };
  for (const b of document.querySelectorAll("#pad [data-hold]")) hold(b, b.dataset.hold);
  $("pad").addEventListener("contextmenu", (ev) => ev.preventDefault());

  // ---- virtual stick: up = W (nose down, KSP), down = S, left = A, right = D
  const stick = $("stick"), knob = $("knob");
  let sid = null;
  const DIRS = { w: false, s: false, a: false, d: false };
  const setDir = (k, on) => { if (DIRS[k] !== on) { DIRS[k] = on; press(k, on); } };
  const moveStick = (ev) => {
    const r = stick.getBoundingClientRect();
    const R = r.width / 2 - 16;
    let dx = ev.clientX - (r.left + r.width / 2), dy = ev.clientY - (r.top + r.height / 2);
    const n = Math.hypot(dx, dy);
    if (n > R) { dx *= R / n; dy *= R / n; }
    knob.style.transform = `translate(${dx}px, ${dy}px)`;
    const T = 0.35 * R;
    setDir("w", dy < -T); setDir("s", dy > T); setDir("a", dx < -T); setDir("d", dx > T);
  };
  const release = () => {
    sid = null; knob.style.transform = "";
    for (const k of Object.keys(DIRS)) setDir(k, false);
  };
  stick.addEventListener("pointerdown", (ev) => { ev.preventDefault(); sid = ev.pointerId; stick.setPointerCapture(sid); moveStick(ev); });
  stick.addEventListener("pointermove", (ev) => { if (ev.pointerId === sid) moveStick(ev); });
  stick.addEventListener("pointerup", release);
  stick.addEventListener("pointercancel", release);
  stick.addEventListener("lostpointercapture", () => { if (sid !== null) release(); });

  // ---- live state of the toggle buttons
  const on = (id, v) => { const el = $(id); if (el) el.classList.toggle("on", !!v); };
  return {
    update(st, view) {
      if (!st) return;
      on("tbAP", st.autopilot); on("padAP", st.autopilot);
      on("tbPause", st.paused);
      on("padSAS", st.sas && st.sas !== "OFF");
      on("padLegs", st.legs > 0.5);
      on("tbPad", padOn);
      const w = $("tbWarp"); const txt = st.paused ? "暂停" : st.warp + "×";
      if (w.textContent !== txt) w.textContent = txt;
      const m = $("padMode"); const mt = st.autopilot ? "自动驾驶中" : "手动";
      if (m.textContent !== mt) m.textContent = mt;
      for (const b of document.querySelectorAll("#toolbar [data-cmd]")) {
        const k = b.dataset.cmd;
        if (k === "f") b.classList.toggle("on", !!view.flowMode);
        else if (k === "k") b.classList.toggle("on", !!view.showTraj);
        else if (k === "c") b.classList.toggle("on", !!(st.aero && st.aero.cfd_forces));
        else if (k === "y") b.classList.toggle("on", !!st.wind_on);
      }
    },
  };
}
