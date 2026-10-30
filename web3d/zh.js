// Display names for identifiers that come from the simulator (phases,
// states, scenarios, events).  The server keeps its English identifiers;
// only what is shown on screen is translated.

export const PHASE = {
  STANDBY: "待命", MANUAL: "手动", COAST: "无动力滑行", BOOSTBACK: "反推返航",
  "LANDING BURN": "着陆点火", "CATCH HOLD": "悬停待夹", TERMINAL: "末端下降", LANDED: "已着陆", FALLBACK: "备份制导",
  CRASHED: "箭体损毁", "ON PAD": "停放台上", FLYING: "飞行中", CAUGHT: "筷子夹住",
};

export const SCENARIO = {
  "HOVER-SLAM 100 m": "100 m 悬停急停",
  "2000 m / +30 m/s UP": "2000 m · 上升 30 m/s",
  "2000 m / -30 m/s DOWN": "2000 m · 下降 30 m/s",
  "BOOSTBACK +1000 m / 100 m/s": "反推返航 · +1000 m · 100 m/s",
  "DIVERT -1000 m / 3-D": "三维偏置 · −1000 m",
  "ENTRY 12 km / 280 m/s": "12 km 再入 · 280 m/s",
  "TOWER CATCH 2000 m": "筷子夹箭 · 2000 m 下降",
  "TOWER CATCH BOOSTBACK": "筷子夹箭 · 反推返航 +1000 m",
  "TOWER CATCH ENTRY 12 km": "筷子夹箭 · 12 km 再入",
  "TOWER CATCH OVER TOWER": "筷子夹箭 · 塔后低空越塔",
  "RTLS 30 km / 450 m/s": "返场 · 30 km 分离 · 450 m/s",
  "RTLS 35 km / CROSSWIND": "返场 · 35 km 分离 · 强侧风",
  "RTLS 30 km / LOW FUEL": "返场 · 30 km 分离 · 推进剂 3 t",
  "RTLS 40 km / GUST + TUMBLE": "返场 · 40 km 分离 · 阵风 + 翻滚",
  "ON PAD - MANUAL": "发射台 · 手动",
};

export const CRASH = {
  "NUMERICAL FAULT (NON-FINITE FORCE)": "数值故障（受力出现非有限值）",
  "LEG FAILURE (HARD LANDING)": "着陆腿损坏（硬着陆）",
  "HULL IMPACT": "箭体撞地",
  "IMPACT - LEGS NOT DEPLOYED": "撞地（着陆腿未展开）",
  "TIPPED OVER": "倾倒",
  "TOWER COLLISION": "撞上发射塔",
  "ARM COLLISION": "撞上筷子臂",
  "HIT LAUNCH MOUNT": "未夹住，坠落在发射台上",
};

const SAS = { OFF: "关", STABILITY: "姿态保持", RETROGRADE: "对准逆速度" };

export const SOLVER_STATUS = {
  idle: "空闲", Solved: "已收敛", AlmostSolved: "近似收敛", MaxIterations: "达到迭代上限",
  PrimalInfeasible: "原问题不可行", DualInfeasible: "对偶不可行", nonfinite: "解含非有限值", "no-solver": "无求解器",
};
export const BACKEND = { "clarabel-python": "Clarabel（Python）", "feedback-fallback": "反馈备份" };

export const phase = (p) => PHASE[p] || p || "";
export const scenario = (s) => SCENARIO[s] || s || "";
export const crash = (s) => CRASH[s] || s || "";
export const solverStatus = (s) => SOLVER_STATUS[s] || s;
export const backend = (b) => BACKEND[b] || (b ? `原生库 ${b}` : "-");

export function event(e) {
  if (typeof e !== "string") return String(e);
  let m;
  if ((m = e.match(/^SCENARIO: (.*)$/))) return "载入场景：" + scenario(m[1]);
  if ((m = e.match(/^PHASE (.*)$/))) return "进入阶段：" + phase(m[1]);
  if (e === "AUTOPILOT ENGAGED") return "自动驾驶接通";
  if (e === "AUTOPILOT OFF") return "自动驾驶断开";
  if ((m = e.match(/^SAS (.*)$/))) return "增稳：" + (SAS[m[1]] || m[1]);
  if (e === "AERO FORCES: LIVE 3D CFD") return "气动力改由实时 3D CFD 给出";
  if (e === "AERO FORCES: ENGINEERING MODEL") return "气动力改由工程模型给出";
  if (e === "WIND ON") return "风场接通";
  if (e === "ARMS READY") return "筷子臂预合拢，等待箭体";
  if (e === "ARMS CLOSING") return "筷子臂合拢";
  if (e === "CATCH CONTACT") return "挂点落上筷子臂，发动机关机";
  if (e === "WIND OFF") return "风场断开";
  if ((m = e.match(/^WIND CFG (\d+)-(\d+) ([\d.]+)-([\d.]+) ([\d.]+)$/))) {
    const dir = m[1] === m[2] ? `来向 ${m[1]}°` : `来向 ${m[1]}°→${m[2]}°`;
    const spd = m[3] === m[4] ? `${m[3]} m/s` : `${m[3]}–${m[4]} m/s`;
    return `风场设置：${dir}，风速 ${spd}，切变 ×${m[5]}`;
  }
  if ((m = e.match(/^CRASHED(?:: (.*))?$/))) return "箭体损毁" + (m[1] ? "：" + crash(m[1]) : "");
  if (PHASE[e]) return PHASE[e];
  return e;
}
