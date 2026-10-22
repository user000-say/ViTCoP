# BELA CVPR2027 — 两份 Zip 交接状态汇总（交付实验 Agent）

**日期：** 2026-10-22  
**当前 Git：** `83b5237`（fork/main）  
**GPU 约定：** 物理卡 0/2/4/7  
**本轮 ACC 任务集（用户覆盖）：**  
`gqa, pope, mme, mmbench_en_dev, scienceqa_img, textvqa_val_ocr`（6-task；**非**完整 R9；mmmu/ai2d/vizwiz **未跑**）  
**Scorer：** `elva_scores.py` Dense refs（GQA 61.9 / POPE 85.8 / MME **1852** / MMB 64.1 / SciQA 69.51 / TextVQA-OCR 58.20）  
**注意：** 外部论文 Dense MME 常为 1506.5；跨论文绝对值比较前须先统一口径。

---

## 0. 两份 Zip 身份（勿混用臂名）

| Zip | 路径 | 性质 |
|---|---|---|
| **Zip-1** 研究+代码 handoff | `BELA_ELVA_CVPR2027_Cursor_Handoff_研究与代码_20261022.zip` → 解压于 `experiments/bela_cvpr2027_20261022/` | 动态 Stage1/Stage2 研究包 + G1 静态入口臂；**不是**可直接覆盖现役 COVER/RDRIFT 的生产补丁 |
| **Zip-2** 研究协议/决策 | `BELA_CVPR_2027_RESEARCH_RULES_CURSOR_20261022.zip` → `experiments/bela_cvpr2027_rules_20261022/` | 强制「先验证日程分布、再加模块」；定义 Gate **T1/T2/T3/T4** |

**臂名冲突（实验 Agent 必须分清）：**

| 名称 | Zip-1 含义 | Zip-2 含义 | 本轮实际 |
|---|---|---|---|
| **A0–A4** | G2/G3 复杂度动态预算 / shuffle / early-drift（`experiment_matrix.csv`） | **T1** 固定 COVER@172+RDRIFT，只改 mid/deep（172/90/32 … 172/140/12） | 只跑了 **Zip-2 的 T1-A0/A2/A3**；Zip-1 的 A* **全部 NOT_RUN** |
| **F0–F3** | P0/G1 静态对照 | （无同名） | **已跑完 6-task** |
| **B0–B2 / C0–C3** | （Zip-1 无；G4 是 STAR-Pro X0） | T2 / T3 | **全部 NOT_RUN** |

---

## 1. Zip-1 要求对照（`BELA_ELVA_CVPR_Cursor_Handoff`）

### 1.1 工程 / 交付物

| 要求 | 状态 | 证据 / 缺口 |
|---|---|---|
| 不覆盖现役源码；先审计生产路径 | **DONE** | `BASELINE_LOCK.md`（包内）；生产 dual-COVER + E018 RDRIFT 仍由 host 持有 |
| 离线 `unittest` + plan/control/pope/trace 脚本 | **DONE（包内 CPU）** | 包测试 PASS；examples 为合成数据，**非**论文指标 |
| 产出 `BASELINE_LOCK.md` | **DONE（部分）** | 有锁文件；HEAD 已前移，复用时请 `git rev-parse HEAD` 再锁一次 |
| Stage1 COVER 接受 `budget=K0∈{96,128,160,172}`，slots=32 | **DONE** | `SOURCE_DIFF_STAGE1_COVER_K0.md`；`e013_stage1_support` / `native_cover` / `cover_binding` / `llava_stage1` |
| 只读 Trace：greedy_order、逐层真实 W、KV/mask/position | **NOT_RUN / 缺口** | 无生产级 `SMOKE_AUDIT.jsonl`；无逐样本 `actual_K_per_layer` 全量导出 |
| `BELA_NEXT_*` 动态 Stage1/Stage2 接线 | **NOT_RUN** | `BELA_NEXT_ENABLE` 仍 False；未进 clip_encoder/E018 |
| `SOURCE_DIFF.patch`（完整新分支） | **部分** | 仅有 COVER@K0 说明文档，**无**统一 patch 文件 |
| `SMOKE_AUDIT.jsonl` | **NOT_RUN** | — |
| `RESULTS.csv`（含 TTFT/W/overhead） | **NOT_RUN** | 仅有 metrics.json 目录树 |
| `POPE_DIAG.json`（random/popular/adversarial confusion） | **NOT_RUN** | 仅有官方 POPE F1 总分 |
| `STOP_GO_DECISION.md` | **NOT_RUN** | — |
| 同机 Dense / 官方 STAR-Pro T64 对照 + TTFT/Fixed32 | **NOT_RUN** | — |
| 完整九任务 / W1024 / 跨模型 | **NOT_RUN** | 用户本轮限 6-task |

### 1.2 Zip-1 实验矩阵臂

| 臂 | Gate | 含义 | ACC 状态 | 备注 |
|---|---|---|---|---|
| **F0** COVER@172 + Q-Attn | P0 | 同系统冻结参照 | **DONE 6/6** | 见 §3 |
| **F1** COVER@172 + RDRIFT | P0 | BELA 锚点 | **DONE 6/6** | ≡ Zip-2 **T1-A0** |
| **F2** COVER@160 + RDRIFT `160/96/32` @4/8/20 | G1 | 入口预算重分配 | **DONE 6/6** | 矩阵写 FIXED，实现为 **RDRIFT**（与 F1 同 Stage2） |
| **F3** COVER@128 + RDRIFT `128/64/32` @8/8/16 | G1 | 入口+生命周期多因子 | **DONE 6/6** | 同上 |
| **A0** 复杂度 profile 动态入口 | G2 | QuantilePolicy EASY/NORMAL/HARD | **NOT_RUN** | 缺生产接线 |
| **A1** shuffle budget 负控 | G2 | | **NOT_RUN** | |
| **A2** random budget 负控 | G2 | | **NOT_RUN** | **≠** T1-A2 |
| **A3** early-drift 选 cut | G3 | | **NOT_RUN** | **≠** T1-A3 |
| **A4** shuffle drift 负控 | G3 | | **NOT_RUN** | |
| **X0** 官方 STAR-Pro T64 | G4 | 外部强参照 | **NOT_RUN** | |

**日志根：** `logs/e014_aee/full/bela_cvpr2027_g1_6task/{F0,F1,F2,F3}_*/{task}/metrics.json`

---

## 2. Zip-2 要求对照（`BELA_CVPR_RESEARCH_PROTOCOL` / DECISION）

### 2.1 流程 / 协议

| 要求 | 状态 | 说明 |
|---|---|---|
| 安装 `AGENTS.md` + `.cursor/rules/bela_cvpr_research.mdc` | **DONE** | rules 目录已装 |
| 离线 unittest + `schedule_audit.py` | **DONE** | 6/6 tests；12 arms schedule PASS |
| P0 只读审计 + A0 与历史绑定 | **部分 DONE** | A0≡F1 复用历史分数；未重跑独立 A0 GPU 复现；无完整逐层 W 绑定 |
| T1：只改 `(k_mid,k_final)`，固定 COVER@172 + RDRIFT + cut 4/12 | **部分 DONE** | A0/A2/A3 ACC 完成；A1/A4 配置已写 **未排队** |
| 小样本 smoke 后再全量 | **DONE** | A2 smoke 后扩 allowlist `final K∈{…,20,24}` |
| 逐样本字段（ids / W_verified / TTFT / POPE_TPFP…） | **NOT_RUN** | 仅有官方 metrics.json |
| T1 后：同 K 深层选择器 RDRIFT vs Q-Attn vs random/stride（≥3 seed） | **NOT_RUN** | DECISION T4 / CURSOR 阶段三 |
| T2 B0–B2（少入口、长前期） | **NOT_RUN** | |
| T3 C0–C3（W1024） | **NOT_RUN** | |
| Stage1×Stage2 交互消融 | **NOT_RUN** | |
| 动态入口（仅当难度信号胜出 shuffle） | **禁止启动**（按协议）直到 T1/T2 证据充分 | Zip-1 G2 亦未接线 |

### 2.2 Zip-2 T1–T3 臂状态

| 臂 | Schedule (K×L) | Stage1/2 | ACC | 说明 |
|---|---|---|---|---|
| **A0** | 172×4+90×8+32×20 | COVER/RDRIFT | **DONE（复用 F1）** | 锚点 |
| **A1** | 172×4+110×8+24×20 | COVER/RDRIFT | **NOT_RUN** | 配置已写 |
| **A2** | 172×4+120×8+20×20 | COVER/RDRIFT | **DONE 6/6** | **本轮最佳 R4** |
| **A3** | 172×4+130×8+16×20 | COVER/RDRIFT | **DONE 6/6** | |
| **A4** | 172×4+140×8+12×20 | COVER/RDRIFT | **NOT_RUN** | 配置已写；DECISION：视 A2/A3 再开 |
| **B0–B2** | T2 STAR-like shapes | COVER/RDRIFT | **NOT_RUN** | |
| **C0–C3** | T3 W1024 | PCOV/QAttn 或 COVER/QAttn | **NOT_RUN** | |

**日志根：** `logs/e014_aee/full/bela_cvpr2027_t1_6task/{A0,A2,A3}_COVER_RDRIFT/{task}/metrics.json`  
**结果表：** `experiments/bela_cvpr2027_rules_20261022/RESULTS_T1_6task.md`

**T1 假说读数（ACC only，无 TTFT）：** A2/A3 相对 A0 的 R4 提升支持「少留 deep-32、多留 mid」；GQA/TextVQA 略降。正式 Pareto / 是否扩 A1/A4 与 T2 仍待实验 Agent 决策。

---

## 3. 已测完：6-task ACC 数字（abs / relative%）

Dense refs 见文首。R4 = 四任务相对 ACC 均值；mean6 = 六任务相对 ACC 均值。

### 3.1 Zip-1 G1（入口/选择器）— `bela_cvpr2027_g1_6task`

| Arm | GQA | POPE F1 | MME | MMB | SciQA | TextVQA | **R4** | **mean6** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 COVER+QATTN 172/90/32 | 57.18 / 92.37% | 82.99 / 96.73% | 1765.17 / 95.31% | 62.20 / 97.03% | 69.16 / 99.50% | 56.38 / 96.88% | 95.36 | 96.30 |
| F1 COVER+RDRIFT 172/90/32 | 57.84 / 93.44% | 83.31 / 97.10% | 1767.83 / 95.46% | 62.20 / 97.03% | 68.96 / 99.21% | 55.97 / 96.16% | 95.76 | 96.40 |
| **F2 ENTRY160** 160/96/32 | 57.88 / 93.50% | 83.74 / 97.60% | 1764.72 / 95.29% | 62.54 / 97.57% | 69.01 / 99.29% | 55.28 / 94.99% | **95.99** | 96.37 |
| F3 ENTRY128 128/64/32 | 57.89 / 93.53% | 84.28 / 98.23% | 1757.49 / 94.90% | 62.03 / 96.77% | 68.91 / 99.14% | 55.55 / 95.45% | 95.86 | 96.34 |

**G1 读数：** 同 W2048 下 F2（K0=160）R4 略优于 F1 锚点；RDRIFT（F1）优于 Q-Attn（F0）。无 TTFT，不能宣称延迟 Pareto。

### 3.2 Zip-2 T1（中后层配比）— `bela_cvpr2027_t1_6task`

| Arm | Schedule | GQA | POPE F1 | MME | MMB | SciQA | TextVQA | **R4** | **mean6** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 (=F1) | 172/90/32 | 57.84 / 93.44% | 83.31 / 97.10% | 1767.83 / 95.46% | 62.20 / 97.03% | 68.96 / 99.21% | 55.97 / 96.16% | 95.76 | 96.40 |
| **A2** | **172/120/20** | 57.62 / 93.08% | 85.44 / 99.58% | 1774.85 / 95.83% | 63.14 / 98.51% | 68.67 / 98.79% | 54.89 / 94.32% | **96.75** | **96.68** |
| A3 | 172/130/16 | 57.35 / 92.66% | 85.68 / 99.86% | 1775.86 / 95.89% | 62.80 / 97.97% | 68.67 / 98.79% | 54.88 / 94.30% | 96.59 | 96.58 |

**Δ vs A0：** A2 R4 **+0.99**；A3 R4 **+0.83**。增益主要在 POPE/MMB/MME；GQA、TextVQA 略退。

---

## 4. 明确未测 / 未交付清单（实验 Agent 待办优先级建议）

### P0 阻塞级（协议要求、尚缺）
1. **逐层真实 W / patch-ID / KV·mask·position smoke** → `SMOKE_AUDIT.jsonl`（至少 A0 与 A2）  
2. **TTFT / Fixed32 / selector_ms**（同卡交替，AUDIT off）— 无延迟则不能关 Gate  
3. **POPE 子集 confusion + Yes-rate**（`POPE_DIAG.json`）— 尤其 A2/A3 POPE 大涨需排除偏置  
4. **A0 独立重跑绑定**（可选但 DECISION 要求核对）vs 复用 F1  

### P1 按 DECISION 续跑 ACC
5. **T1-A1 / T1-A4**（配置已在 `experiments/bela_cvpr2027_rules_20261022/configs/`）  
6. **T2 B0–B2**（少入口长前期）  
7. **深层选择器负控**：同 schedule 上 RDRIFT vs Q-Attn vs seeded_random/stride（≥3 seed）  
8. **T3 C0–C3** W1024  

### P2 Zip-1 动态线（仅当固定日程证据足够且难度信号可证伪）
9. 接线 `BELA_NEXT_*` + greedy_order / residual difficulty  
10. G2 A0 profile + **shuffle/random budget 负控**（同 mean W）  
11. G3 early-drift / shuffle-drift  
12. **X0 官方 STAR-Pro** 同机对照  

### P3 评测扩展
13. 完整 **R9**（+ mmmu_val, ai2d, vizwiz_vqa_val）及/或用户指定子集  
14. 交付物打包：`RESULTS.csv`、`STOP_GO_DECISION.md`、统一 `SOURCE_DIFF.patch`

---

## 5. 代码 / 配置定位（实验 Agent 直接用）

| 用途 | 路径 |
|---|---|
| Zip-1 安装树 | `experiments/bela_cvpr2027_20261022/` |
| Zip-1 任务原文 | `.../BELA_ELVA_CVPR_Cursor_Handoff_20261022/00_直接复制给Cursor的任务.md` |
| Zip-1 接线规范 | `.../02_CURSOR_HANDOFF_CN.md` |
| COVER@K0 diff 说明 | `experiments/bela_cvpr2027_20261022/SOURCE_DIFF_STAGE1_COVER_K0.md` |
| G1 配置 | `experiments/bela_cvpr2027_20261022/configs/F{2,3}_ENTRY*_RDRIFT_W2048.json` |
| Zip-2 决策 | `experiments/bela_cvpr2027_rules_20261022/BELA_CVPR_RESEARCH_PROTOCOL_20261022/research/DECISION_2026-10-22.md` |
| Zip-2 矩阵 | `.../research/experiment_matrix.csv`（CSV 内 status 字段可能仍写 NOT_RUN，以本文/metrics 为准） |
| T1 配置 | `experiments/bela_cvpr2027_rules_20261022/configs/A*_COVER_RDRIFT_W2048.json` |
| T1 结果 | `experiments/bela_cvpr2027_rules_20261022/RESULTS_T1_6task.md` |
| G1 metrics | `logs/e014_aee/full/bela_cvpr2027_g1_6task/` |
| T1 metrics | `logs/e014_aee/full/bela_cvpr2027_t1_6task/` |
| Host allowlist | `vitcop_modeling_llama.py` final K 已含 `{12,16,20,24,32}`（A2/A3 所需） |

---

## 6. 一句话结论（给实验 Agent）

- **已完成：** Zip-1 的 G1 静态四臂（F0–F3）6-task ACC + Stage1 COVER@K0 主机接线；Zip-2 的 T1 优先级三臂（A0=复用 F1，A2/A3 新跑）6-task ACC；协议规则与离线 schedule 审计。  
- **未完成：** 几乎全部延迟/逐样本 trace/POPE 诊断交付物；T1-A1/A4、全部 T2/T3/T4、Zip-1 动态 G2–G4、STAR-Pro、R9。  
- **当前最强 ACC 点（仅相对 Dense、无 TTFT）：** **T1-A2**（172/120/20）R4=96.75；G1 侧 **F2**（160/96/32）R4=95.99。  
- **下一步建议（与 DECISION 一致）：** 先补 A2（及 A0）的 W-trace + TTFT + POPE 诊断；再决定扩 A1/A4 或进 T2 / 选择器负控；**在此之前不要接线动态 Stage1。**
