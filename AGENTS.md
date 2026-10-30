<INSTRUCTIONS>
项目条件：
- 项目实现 SpaceX（Falcon 9）式火箭自动定点回收：从任意空中状态开启自动模式后，经滑行 →（需要时）反推返航 → 最优时刻点火的一次连续着陆燃烧直接烧到触地（2D 与 3D 均不设单独的末端下降阶段），落到发射台中心。
- 制导使用凸优化动力下降规划（Açıkmeşe–Ploen 无损凸化 / G-FOLD 结构）：推力锥 ‖u‖ ≤ Γ、ρ_min ≤ Γ ≤ ρ_max、随剩余时间收紧的推力倾角约束、以台面为顶点的滑翔锥（软约束）、油门与侧向推力变化率约束、预测气动阻力、终端状态用精确 L1 罚函数保证问题恒可行。
- 飞行时间为自由终端时间：点火前以燃料最优搜索决定点火时机；点火后锁定触地时间、以平滑目标重规划；无法按时到达时只能移到“最早可达”时刻，不得单纯延长时间（会导致贴地悬停）。
- 规划时域覆盖“从现在到触地”（N = 40 节点、推力一阶保持），不再使用固定 600 秒时域；绿线显示规划轨迹，蓝线显示实际轨迹。
- 求解器使用 Clarabel：矩阵在 Python（lander2d/planner.py、lander3d/planner.py）中组装，经 common/conic.py 交给求解器：原生 DLL（native/src/lib.rs，ABI 2，文件名 clarabel_hover.dll）只做通用锥规划求解，Python clarabel 包作为同一问题的后备。修改 lib.rs 后运行 native\build_solver.ps1 重新编译。
- 规划推力上限留出反馈余量（点火规划 80%，着陆燃烧 85%）；内环为规划推力前馈 + PD 修正 + 级联 TVC 姿态环（限幅角速度、±12° 摆角、角速度前馈）；发动机关闭时允许用 RCS 调姿。
- 保留普通地面碰撞和地图边界碰撞；只有落点误差 ≤ 8 m、速度 ≤ 4 m/s、姿态 ≤ 12°、角速度 ≤ 20°/s 的接触才判定为 LANDED。
- 必须用无窗口闭环仿真实测（python main.py --landing-test）：100 m 静止、2000 m 垂直速度 +/-30 m/s、2000 m 水平位置 +/-1000 m 且水平速度 +100 m/s。验收标准：LANDED、落点误差 ≤ 1 m、触地下沉 ≤ 2 m/s、水平速度 ≤ 1 m/s、姿态 ≤ 5°，由 Clarabel 求解。
- python main.py --record <目录> 用游戏自身渲染器离屏录制上述场景的视频（需要 imageio-ffmpeg，缺失时输出 JPEG 帧）。
- 画面：lander2d/visuals.py 负责箭体、尾焰、烟尘和着陆台；lander2d/cockpit.py 负责模拟舱风格仪表（PFD、自动驾驶面板、发动机油门滑杆、NAV、最后 50 m 的着陆区精度监视器）。所有仪表只读 Telemetry，不参与控制。
</INSTRUCTIONS>
- 3D 版是独立入口 main3d.py（不得改动 2D 的 main.py 与 lander2d/ 行为）：lander3d/vehicle.py 6 自由度 + 推进剂/质量/质心/惯量变化 + TVC/栅格舵/RCS，lander3d/aero.py 工程气动 + ExternalCFD（接收浏览器 GPU 3D CFD 的力系数，按 C 驱动受力，限制在工程模型 0.5–2 倍内，>1 s 无数据退回模型），lander3d/planner.py + autopilot.py 3D G-FOLD（与 2D 共用 common/conic.py 的 Clarabel 后端，气动法向力对推力方向逐次线性化进规划），lander3d/sim.py 场景与 KSP 式手控，lander3d/server.py 本地服务，web3d/ 为 WebGL2 前端（fluid.js 是 3D GPU 流体；前端只读状态，经 /api/input 发按键，经 /api/cfd3d 发 CFD 力系数）。
- 3D 自动驾驶结构：滑行（迎风 + 攻角升力，ZEM 导引把“竖直着陆燃烧终点”_burn_miss 引到台面，横向误差在空中消除）→ 仅当弹道落点偏离 > 350 m 且空速 < 180 m/s 时反推返航（在阻力弹道上打靶求速度增量，使 _burn_miss 落在台面，即真正回到台面上方；够不着或点火下落角 > 10° 时加爬升，上限 80 m/s）→ 点火预测器（自适应极限点火：所需推力比 ≥ 0.76 − 0.14·侧向占比，预计燃烧时间 2/5/8 s 时阈值为 0.48/0.62/全值，短燃烧保持原裕度）→ 锁定时间的一次连续着陆燃烧，G-FOLD 直接规划到台面（竖直、1.2 m/s 下沉、近地下沉包络 1.2 + 0.5·h）→ 一直烧到触地即关机且不再点火（无末端下降阶段）。栅格舵为 4 片物理舵面（lander3d/vehicle.py 舵机 ±20°、40°/s、展开 1.5 s，反推时收起；lander3d/aero.py 的 grid_fin_forces 按每片当地来流算法向力/阻力），姿态环按数值效能矩阵分配舵面 → TVC → RCS；着陆燃烧倾角上限随动压收紧（2.5 kPa 30° → 8 kPa 12°，防止尾向来流箭体升力与推力侧向分量反向造成的倾角发散）；点火后按实测推力比 thrust_scale 缩放规划推力上下限；预测中的风按对数风廓线由当前风外推；前端 snapshot 含 fins（四片偏转角°、展开度），scene.js 按其转动/折叠舵面。禁止任何末段悬停/减速等待修正（下沉速度只能单调减小）；剩余时间 ≤ 5 s 时不再重新定时。
- 3D 必须用 python main3d.py --test 实测 6 个场景：LANDED、落点 ≤ 3.5 m、下沉 ≤ 2.5 m/s、水平速度 ≤ 1.5 m/s、姿态 ≤ 5°，并检查没有悬停（着陆燃烧中离地 > 1.5 m 时下沉速度不得小于 0.8 m/s）和没有接地前摇摆（离地 < 8 m 时俯仰/偏航角速度 ≤ 12°/s，触地姿态 ≤ 1.5°）。姿态环的角速度指令必须受执行机构可用角加速度限制，推力方向指令有转动速率上限。前端改动后用无头 Chromium（WebGL2/SwiftShader，?cfd=low）截图检查；服务端 JSON 不得含 NaN（server.finite 过滤），数值发散时判 NUMERICAL FAULT。
- 前端：相机不得自动旋转（只随鼠标/手指）；仪表面板可拖动（指针事件，鼠标与触屏通用）并记住位置，每个面板都可折叠成小标签、折叠状态也记住（L 复位）；菜单面板可切换场景；手机兼容（web3d/controls.js：单指旋转、双指缩放、虚拟摇杆与油门/自动/增稳/着陆腿按钮，手机默认只展开飞行数据条与操控面板）；姿态球为阿波罗 FDAI 风格、着陆区为带软键的 MFD 式 ND（web3d/instruments.js）；着陆成功不弹 LANDED 横幅；RCS 为白色冷气喷流。
- 网页版：web3d/sim_worker.js + sim_worker.py 在浏览器 Web Worker 里用 Pyodide 运行同一套 3D Python 仿真（common/ 与 lander3d/，无 /api 后端时自动启用，?local 强制）；deploy/build_web.py 生成纯静态包 dist/web3d/（py/manifest.json 列出打包的源文件），deploy/nginx-rockt3d.conf 静态托管。修改 common/、lander3d/ 或 web3d/ 后需重新构建并同步到 /var/www/rockt3d。
- 代码布局：main.py / main3d.py 只是入口；2D 在 lander2d/，3D 在 lander3d/，共用求解管道在 common/conic.py，检查与研究脚本在 tools/（python -m tools.<名称>），pytest 在 tests/（pytest 默认跳过 slow，完整场景用 pytest -m slow）。提交前运行 ruff check .。
