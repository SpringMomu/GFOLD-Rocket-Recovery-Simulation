// Four independently scaled time series; samples come from the simulation,
// so opening or folding the panel never interrupts the flight record.
let lastKey = "";
export function drawHistory(canvas, history) {
  const last = history[history.length - 1];
  const key = `${history.length}:${last ? last.join(",") : ""}`;
  if (key === lastKey) return;
  lastKey = key;
  const g = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
  g.clearRect(0, 0, W, H);
  g.fillStyle = "rgba(0, 0, 0, .26)"; g.fillRect(0, 0, W, H);
  const plots = [
    ["姿态倾角", "°", "#70d7ff"], ["速度", "m/s", "#ffca6e"],
    ["离地高度", "m", "#7df1a2"], ["水平误差", "m", "#f28eb9"],
  ];
  const left = 116, right = W - 24, row = 94, top = 12;
  const duration = last ? Math.max(1, last[0]) : 1;
  g.font = '18px "Segoe UI", "Microsoft YaHei", sans-serif';
  plots.forEach(([label, unit, color], k) => {
    const y = top + row * k, bottom = y + 65;
    const max = Math.max(1, ...history.map((sample) => Math.max(0, sample[k + 1])));
    g.fillStyle = "#a8bdcb"; g.textAlign = "left"; g.fillText(label, 8, y + 20);
    g.fillStyle = color;
    g.fillText(last ? `${last[k + 1].toFixed(1)} ${unit}` : `-- ${unit}`, 8, y + 46);
    g.strokeStyle = "rgba(170,200,220,.18)"; g.lineWidth = 1;
    for (let j = 0; j <= 2; j++) {
      const gy = y + j * 32.5; g.beginPath(); g.moveTo(left, gy); g.lineTo(right, gy); g.stroke();
    }
    if (history.length > 1) {
      g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
      history.forEach((s, i) => {
        const x = left + (right - left) * s[0] / duration;
        const py = bottom - Math.max(0, s[k + 1]) / max * 65;
        if (i === 0) g.moveTo(x, py); else g.lineTo(x, py);
      });
      g.stroke();
    }
    g.textAlign = "right"; g.fillStyle = "#8397a5"; g.font = '15px "Segoe UI", sans-serif';
    g.fillText(`${max.toFixed(max < 10 ? 1 : 0)} ${unit}`, right, y + 15);
    g.font = '18px "Segoe UI", "Microsoft YaHei", sans-serif';
  });
  g.fillStyle = "#8397a5"; g.font = '16px "Segoe UI", sans-serif';
  g.textAlign = "left"; g.fillText("T+0 s", left, H - 12);
  g.textAlign = "right"; g.fillText(`T+${last ? last[0].toFixed(1) : "0"} s`, right, H - 12);
}
