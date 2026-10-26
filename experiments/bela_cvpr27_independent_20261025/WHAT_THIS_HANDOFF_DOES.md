# 这一版 Independent Handoff 主要做什么

权威入口：`BELA_CVPR27_INDEPENDENT_CURSOR/00_START_CURSOR.md`（取代 136-arm 扫参）。

## 一句话
在**相同剪枝日程**下，分清精度差距来自 **Stage1 入口（COVER vs STAR QR）** 还是 **Stage2 退休选择器**；不要堆新模块。E3 作 W2048 实用基线；E1 仅作机制参考（更慢）。

## 主实验（P1）— 同日程 2×2
| | BELA Stage2 | STAR Stage2 |
|---|---|---|
| BELA Stage1 | **BB**（可复用 L2/E5） | **BS**（需合法 ID 移植 → 当前 BLOCKED） |
| STAR QR Stage1 | **QB**（BLOCKED） | **QS**（官方 STAR 端到端，可复用/已跑） |

预算：W1024=`64×12+17×8+10×12`；W2048=`128×12+37×8+18×12`。

## 第二实验（P2）— 不抄对手的瓶颈定位
固定 **E3** 日程与 Stage2，只改 Stage1：dual COVER（已有）vs penultimate-only vs Layer9-only。  
另有最终切点选择器对照（QAttn / RDRIFT / random）— 需源码审计后开。

## 明确不做
- 无因果证据不实装新 Guard / provenance 模块  
- 混合交叉移植不合法则 **BLOCK**，不造分  
- 不以测试集调参；不把 STAR 日程形状当 BELA 创新

## 本机执行状态（摘要）
见 `runs/RESULTS_MASTER.csv` 与 `runs/STOP_GO.md`。
