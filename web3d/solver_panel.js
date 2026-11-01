// Solver settings panel: the guidance (G-FOLD) options of lander3d/settings.py.
// The simulator owns the ranges and the clamping (snapshot solver_opts =
// [key, kind, lo, hi, step, default], solver_cfg = the values in use); this
// file only adds labels and the layout.  A change is sent as the action
// "solvercfg:<json>" ({key: value}), the full settings are remembered in
// localStorage and sent again on load (the browser simulator starts afresh).

const $ = (id) => document.getElementById(id);
const KEY = "booster3d.solver.v1";

const pct = (v) => `${Math.round(v * 100)}%`;
const times = (v) => `×${v.toFixed(2)}`;
const num = (d, u = "") => (v) => `${v.toFixed(d)}${u}`;

// group: wind = wind resistance, plan = planner / solver
const DEFS = [
  { key: "wind_plan_gain", group: "wind", label: "规划假设风速", fmt: times,
    hint: "规划预测气动力、攻角和配平用的风 = 实测风 × 该系数。1 = 完全按实测风（默认），0 = 按无风规划，风只靠反馈修正" },
  { key: "aero_pred", group: "wind", label: "气动力预测", choices: ["不预测", "仅阻力", "阻力+升力"],
    hint: "规划把箭体气动力算进动力学的程度：不预测 / 只算姿态不变时的阻力 / 再加上倾斜产生的升力线性化（默认）" },
  { key: "trim_gain", group: "wind", label: "迎风配平", fmt: pct,
    hint: "倾角锥和跟踪器的倾角预算以“迎风倾斜”为中心的程度。100% = 完全迎风（默认），0 = 以竖直为中心" },
  { key: "lateral_aero_share", group: "wind", label: "顶风推力不计代价", fmt: pct,
    hint: "侧向机动代价里扣除气动力的比例。100% = 为顶风付出的推力不算代价（默认），0% = 侧向推力一律计代价，规划不愿意硬顶风" },
  { key: "shear_fit", group: "wind", label: "风切变在线估计", toggle: true,
    hint: "用下降途中各高度实测风拟合风切变强度。关闭 = 固定按标准对数风廓线（强度 1）外推" },
  { key: "lift_trim", group: "wind", label: "升力按配平后计算", toggle: true,
    hint: "规划与滑翔用的箭体升力按“栅格舵为抵消气动力矩偏转后”的实际升力算（零偏转的模型会高估 2–4 倍），并用飞行中实测的气动力在线标定。关闭 = 零偏转的原模型" },
  { key: "aoa_margin", group: "wind", label: "攻角力矩余量", fmt: pct,
    hint: "气动力矩最多占用控制力矩的比例，决定高动压下允许的攻角（规划与跟踪器共用）。越小越保守" },

  { key: "time_weight", group: "plan", label: "时间权重 λ", fmt: num(1),
    hint: "目标 J = 推进剂 + λ·触地时间。λ 越大越倾向快落，越小越省推进剂" },
  { key: "knots", group: "plan", label: "规划节点数", fmt: num(0),
    hint: "规划离散节点数（长规划约每秒一个，最多 70）。越多越精细，求解也越慢" },
  { key: "lateral_accel_weight", group: "plan", label: "侧向机动代价", fmt: num(1),
    hint: "净侧向加速度（推力加气动力）的二次代价权重，抑制猛打侧向推力" },
  { key: "jerk_weight", group: "plan", label: "推力变化率代价", fmt: num(2),
    hint: "推力矢量变化率的二次代价权重，越大规划推力越平滑" },
  { key: "lateral_rate", group: "plan", label: "侧向推力变化率上限", fmt: num(1, " m/s³"),
    hint: "规划的侧向推力每秒最多变化多少（还会再被跟踪器的转动速率上限收紧）" },
  { key: "divert_tilt_deg", group: "plan", label: "最大推力倾角", fmt: num(0, "°"),
    hint: "规划允许的推力偏离竖直的最大角度（越接近触地越收紧）" },
  { key: "burn_fraction", group: "plan", label: "推力上限", fmt: pct,
    hint: "着陆燃烧规划使用的最大推力占满推力的比例，其余留给反馈余量" },
  { key: "time_search_evals", group: "plan", label: "时间搜索求解次数", fmt: num(0),
    hint: "点火时搜索触地时间的求解次数，越多越准，点火瞬间越卡" },
  { key: "retime_step", group: "plan", label: "重定时试探步长", fmt: pct,
    hint: "飞行中每秒试探触地时间的步长（占剩余时间的比例）" },
  { key: "max_iter", group: "plan", label: "求解器迭代上限", fmt: num(0),
    hint: "Clarabel 内点法单次求解的最大迭代数；失败时仍会用 400 次重解一次" },
];

const PRESETS = {
  weak: { wind_plan_gain: 0.5, trim_gain: 0.5, lateral_aero_share: 0.5, aero_pred: 1 },
  none: { wind_plan_gain: 0, trim_gain: 0, lateral_aero_share: 0, aero_pred: 0 },
};

export function initSolverPanel({ act }) {
  const body = $("soBody");
  if (!body) return { update() {} };
  let opts = null;                       // key -> {kind, lo, hi, step, def}
  const rows = {};                       // key -> {set(value)}
  let cur = {};                          // values shown
  let pending = null, pendingUntil = 0, holdUntil = 0;

  const store = () => { try { localStorage.setItem(KEY, JSON.stringify(cur)); } catch (e) { /* ignore */ } };
  const send = (part) => {
    Object.assign(cur, part);
    act("solvercfg:" + JSON.stringify(part));
    pending = part; pendingUntil = performance.now() + 5000;
    store(); paintTag();
  };
  const paintTag = () => {
    if (!opts) return;
    const n = Object.keys(opts).filter((k) => k in cur && Math.abs(cur[k] - opts[k].def) > 1e-9).length;
    $("soTag").textContent = n ? `已改 ${n} 项` : "默认";
    for (const k of Object.keys(rows)) rows[k].mark(k in cur && Math.abs(cur[k] - opts[k].def) > 1e-9);
  };

  const build = (list) => {
    opts = {};
    for (const [key, kind, lo, hi, step, def] of list) opts[key] = { kind, lo, hi, step, def };
    const groups = { wind: "抗风", plan: "规划与求解" };
    const subs = { wind: "规划对风的预测与应对", plan: "" };
    for (const g of Object.keys(groups)) {
      const h = document.createElement("div");
      h.className = "sg";
      h.innerHTML = `${groups[g]}<small>${subs[g]}</small>`;
      body.appendChild(h);
      for (const d of DEFS.filter((x) => x.group === g && opts[x.key])) {
        const o = opts[d.key];
        const box = document.createElement("div");
        box.className = "so"; box.title = d.hint;
        const head = document.createElement("div");
        head.className = "wl";
        const name = document.createElement("span"); name.textContent = d.label;
        const val = document.createElement("b");
        head.append(name, val);
        box.appendChild(head);
        let set;
        if (d.choices || d.toggle) {
          const labels = d.choices || ["关", "开"];
          const seg = document.createElement("div");
          seg.className = "seg";
          const btns = labels.map((t, i) => {
            const b = document.createElement("button");
            b.type = "button"; b.textContent = t;
            b.addEventListener("click", (ev) => { ev.stopPropagation(); send({ [d.key]: i }); set(i); b.blur(); });
            b.addEventListener("pointerdown", (ev) => ev.stopPropagation());
            seg.appendChild(b);
            return b;
          });
          box.appendChild(seg);
          set = (v) => { btns.forEach((b, i) => b.classList.toggle("on", i === Math.round(v))); val.textContent = labels[Math.round(v)] || ""; };
        } else {
          const inp = document.createElement("input");
          inp.type = "range"; inp.min = o.lo; inp.max = o.hi; inp.step = o.step; inp.value = o.def;
          inp.setAttribute("aria-label", d.label);
          inp.addEventListener("input", () => { holdUntil = performance.now() + 1500; set(Number(inp.value)); });
          inp.addEventListener("change", () => send({ [d.key]: Number(inp.value) }));
          inp.addEventListener("pointerdown", (ev) => ev.stopPropagation());
          box.appendChild(inp);
          set = (v) => { inp.value = v; val.textContent = d.fmt(Number(v)); };
        }
        body.appendChild(box);
        rows[d.key] = { set, mark: (m) => box.classList.toggle("mod", m) };
      }
    }
    for (const b of document.querySelectorAll("#soPresets [data-preset]")) {
      b.addEventListener("pointerdown", (ev) => ev.stopPropagation());
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const part = b.dataset.preset === "default"
          ? Object.fromEntries(Object.entries(opts).map(([k, o]) => [k, o.def])) : PRESETS[b.dataset.preset];
        for (const [k, v] of Object.entries(part)) if (rows[k]) rows[k].set(v);
        send(part);
        b.blur();
      });
    }
    // The last settings are remembered; send them once to a freshly started simulator.
    let stored = null;
    try { stored = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) { stored = null; }
    if (stored && typeof stored === "object") {
      const part = {};
      for (const [k, v] of Object.entries(stored)) if (opts[k] && typeof v === "number" && Number.isFinite(v)) part[k] = v;
      if (Object.keys(part).length) send(part);
    }
  };

  return {
    update(st) {
      if (!st || !st.solver_opts || !st.solver_cfg) return;
      if (!opts) build(st.solver_opts);
      const now = performance.now();
      const sv = st.solver_cfg;
      if (pending && (Object.keys(pending).every((k) => k in sv && Math.abs(sv[k] - pending[k]) < 1e-6) || now > pendingUntil)) pending = null;
      const dragging = $("solverPanel").contains(document.activeElement) && document.activeElement.tagName === "INPUT";
      if (pending || now < holdUntil || dragging) return;
      cur = Object.assign({}, sv);
      for (const k of Object.keys(rows)) if (k in sv) rows[k].set(sv[k]);
      paintTag();
    },
  };
}
