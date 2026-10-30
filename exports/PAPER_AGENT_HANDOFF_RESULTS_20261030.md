# BELA Stage1 Decision — Paper Agent Handoff (2026-10-30)

**性质**：本机实测进度与结论汇总，供论文 agent 写 Method/Experiments/消融叙述。  
**不是**：最终 camera-ready 表；未完成臂与仍在跑的任务见 §H。  
**协议源**：`CURSOR_HANDOFF.md`（Stage1 结构必要性 kit，2026-10-29）+ 用户当日早停规则。  
**代码/日志根**：
- 包：`/home/xiangyue/ViTCoP/experiments/bela_stage1_decision_package/`
- 新跑输出：`/home/xiangyue/ViTCoP/logs/e014_aee/full/bela_stage1_decision_20261029/`
- 可复用历史：`/home/xiangyue/ViTCoP/logs/e014_aee/full/bela_stage1_queryanchor_20261027/`
- 相关提交：`6510d33`（早停队列）、`bfc2577`（S5 空 content-mask 回退）、`b3fd094`/`5f8fee7`（P2/P3/E.1 接线）

**冻结生产锚点（W1024 L2）**  
Dual-depth Layer9+pen → Native COVER K64 → QA-COVER(B=8) → Stage2 **首剪 RDRIFT、后剪 Q_ATTN**，日程 `64×12 → 17×8 → 10×12`。  
W2048 E3：`172×4 → 120×8 → 24×10 → 16×10`（历史六任务可复用；B6/B10 新跑未完成）。

**评分**：`elva_scores.py` Dense_ref；本文 **R6** = 六任务相对 ACC 均值  
(GQA/61.9, POPE_F1/85.8, MME_total/1852, MMBench/64.1, SciQA-IMG/69.51, TextVQA-OCR/58.20)。  
MME_total = perception + cognition（metrics 里拆开存）。MMBench 用 `gpt_eval_score`。

**当日早停（用户强制，已写入 `launch_master_queue.py`）**
1. **GQA &lt; 55 或 POPE F1 &lt; 85** → 不再测后 4 个 ACC。  
2. 两门都过且 **接近 S1 Dual+QA（56.52 / 86.16；ΔGQA≤0.5、ΔPOPE≤0.8）** → 补满六任务。  
3. 优先臂过地板线可强制六任务；E.1 门控已与 S1 数值相同则跳过冗余后 4。

---

## A. 一句话研究状态（写给论文）

- **Stage1 Dual COVER → QA-COVER(B≈8) 对 POPE/GQA 有真实门控收益**；纯 Query Top-K（S5）POPE 仍高但 **GQA 崩盘**，不能替代 COVER+QA。  
- **Matched-random（S6）与错配问题（S7）均未过 POPE 地板**，支持「不是随便换 token / 不是任意问题都能涨」。  
- **同 W1024 L2 上，QA + 纯 Q_ATTN（P3）六任务 R6≈95.43，与 QA+RDRIFT（S1）95.34 几乎持平略高**；RDRIFT 的历史冻结证据主要来自 **W2048 V5 同预算 first_selector 对照（R4 +0.40）**，不可把 W1024 的 P3 结果写成「RDRIFT 无效」。  
- **B 扫描**：B8 附近 R6 最好；B↑ 抬 POPE/GQA 但伤 MME/SciQA/TextVQA，**不能写「B 越大越好」**。

---

## B. 核心问答（今日澄清，论文须一致）

### Q1. Stage2 多次渐进剪枝是不是只有第一次用 RDRIFT、其余用 Q_ATTN？
**是（现役 `first_selector=RDRIFT`）**。  
L2 两次 Stage2 事件约：`layer11: 64→17`（RDRIFT / E020 rank_drift），`layer19: 17→10`（Q_ATTN）。  
实现：`experiments/elva_r3/apply_llava.py` —— RDRIFT = **仅 FIRST**；later 保持 Q_ATTN。  
P3 `*_QATTN` 臂：两次都是 Q_ATTN。

### Q2. RDRIFT 和 Q_ATTN 区别？为何曾选 RDRIFT？
| | Q_ATTN | RDRIFT |
|---|---|---|
| 定义 | 当前剪点 question→vision attention mass Top-K | `0.6*rank(now)+0.4*(rank(now)-rank(old))`，仅首剪 |
| 干净对照 | 同 Dual COVER、同 **W2048 V5**，只改 first_selector | Gate1 / P0 H2 |
| 历史分数 | R4 **95.36**（GQA 57.18 / POPE 82.99 / MME 1765 / MMB 62.2） | R4 **95.76**（GQA 57.84 / POPE 83.31 / MME 1768 / MMB 62.2） |
| Δ | — | R4 **+0.40**；MMBench 不动 |

注意：更早 E020 W≈2284 线索 **不能**当 W2048 同预算证据；L2 W1024 是 **继承**该默认，不是又做一轮同日程仅 selector 全量对照才冻结。  
**今日 P3（W1024）**：QA+Q_ATTN R6 略高于 QA+RDRIFT —— 论文须分预算陈述，避免混写。

### Q3. S5 是不是 Dual + QA-COVER？
**不是。** S5 = **Query–patch 相似度 Top-K64 一次性选入口**，**无 Native COVER、无 QA 交换**，再进 Stage2。  
用途：检验「只要带 Q 打分是否够」。

### Q4. S6 是什么？为何那么多？
S6 = **Matched-Random 对照**：Native COVER 后，按 QA 的净交换数/删除身份，**随机换入** donor。  
seed `17/23/29/31/37`（≥5）是同一对照的不同随机性，不是五套算法。

### Q5. 「带 Q 的 POPE 都比较高」？
门控数据支持：**含真实 Query 条件的臂 POPE 更高**（S1/P3_QA/S5/S2 ≥85；Native/S6/S7/P3_Native ≈82–84）。  
但 **S5 GQA=53.34 &lt; 55** → 按规则停测后 4；说明 POPE 涨 ≠ 通用 ACC 涨，**不能把 S5 写成主方法**。

---

## C. 结构消融门控（W1024 L2，同 RDRIFT 除非点名）

| ID | Stage1 | GQA | POPE F1 | 六任务？ | 论文读法（初步） |
|---|---|---:|---:|---|---|
| **S0** Native COVER | COVER only | 55.57 | 82.21 | 否（POPE 地板） | 无 QA 时 POPE 弱 |
| **S1** Dual COVER→QA B8 | 主方法 | **56.52** | **86.16** | **是** R6 95.34 | 锚点 |
| **S2** salient Top-K→QA B8 | 换初始化 | 55.91 | 85.29 | 否（不够近 S1） | 过地板但弱于 COVER 初始化 |
| **S3** 一次联合贪心 K64 | 无串行 QA | （跑 GQA 中） | 84.77 | 待 GQA；POPE 已 &lt;85 → 预期停后 4 | 「一次联合」POPE 未过地板 |
| **S4** 联合 + 最多新加 8 | 信任域对照 | （跑 GQA 中） | 82.66 | 预期停 | 弱 |
| **S5** Query Top-K only | 无 COVER/QA | **53.34** | 85.74 | **否（GQA 地板）** | Q 抬 POPE、毁 GQA |
| **S6** matched-random ×5 | 随机换入 | 55.7–56.0 | 82.5–83.3 | 否 | 否决「随便扰动也行」 |
| **S7** 同图错配问题 | shuffle Q | 55.68 | 83.94 | 否 | 真实问题相关 |
| **E.1** `[4,4]` / `[3,3,2]` | 同累计 B=8 | 56.52 | 86.16 | 门控=S1，后 4 跳过 | 与 S1 门控相同 → 次数记账嫌疑（待 traces） |
| **S8** 去 novelty / coverage / 仅 gain | 子模块 | 跑中 | — | — | 未完成 |

**P3 2×2（入口 K64、W1024；Stage2 评分器不同）**

| Stage1 \ Stage2 | RDRIFT（首）+Q_ATTN | 全程 Q_ATTN |
|---|---|---|
| Native | **S0**：55.57 / 82.21（停） | **P3_NATIVE_QATTN**：55.36 / 82.34（停） |
| QA B8 | **S1**：56.52 / 86.16 / R6 **95.34** | **P3_QA_QATTN**：56.38 / 86.27 / R6 **95.43**（六任务齐） |

P3_QA 绝对分：GQA 56.38 | POPE 86.27 | MME **1724.24** | MMB **62.03** | SciQA **69.71** | TextVQA-OCR **52.84**。

---

## D. B = max_swaps 扫描（W1024 L2，Dual+QA+RDRIFT，六任务）

| B | GQA | POPE | MME | MMB | SciQA | TV-OCR | R6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 55.57 | 82.21 | — | — | — | — | 门控失败停 |
| 4 | 56.30 | 85.43 | 1697.54 | 61.51 | 69.16 | 52.72 | 94.71 |
| 6 | 56.33 | 85.88 | 1713.06 | 61.68 | 69.66 | 53.08 | 95.21 |
| **8** | **56.52** | **86.16** | **1715.68** | **61.77** | **69.61** | **53.06** | **95.34** |
| 10 | 56.36 | 86.44 | 1704.12 | 61.68 | 69.51 | 52.94 | 95.17 |
| 12 | 56.39 | 86.58 | 1691.45 | 61.51 | 69.26 | 52.71 | 94.92 |
| 14 | 56.53 | 86.81 | 1689.23 | 61.34 | 68.86 | 52.37 | 94.74 |
| 16 | 56.61 | 86.98 | 1670.75 | 61.17 | 68.42 | 52.07 | 94.39 |
| 18 | 56.77 | 87.09 | 1655.61 | 61.17 | 68.17 | 51.89 | 94.21 |
| 20 | 56.75 | 87.06 | 1658.80 | 61.25 | 68.17 | 51.47 | 94.13 |
| 22 | 56.65 | 87.14 | 1676.31 | 60.91 | 67.53 | 51.58 | 94.06 |
| 24 | 56.52 | 87.17 | 1662.92 | 60.91 | 67.92 | 51.44 | 93.97 |

**读法**：B8 为 R6 峰；更大 B 抬 POPE（及部分 GQA）但系统性伤 MME/SciQA/TextVQA。Top-B 补 MMMU/AI2D/VizWiz **尚未跑完**。

---

## E. W2048 E3（历史六任务，B8）

来源：`.../bela_stage1_queryanchor_20261027/S1_QANCHOR_E3/`

| GQA | POPE | MME | MMB | SciQA | TV-OCR | R6 |
|---:|---:|---:|---:|---:|---:|---:|
| 58.02 | 86.22 | 1781.66 | 63.40 | 68.77 | 54.76 | **97.06** |

E3×B6/B10 新九任务、E3 B8 补 3：**队列中未完成**。  
历史 W2048 V5 COVER Q_ATTN vs RDRIFT（R4 95.36 vs 95.76）见 §B Q2 —— 那是 **选 RDRIFT 的主证据**。

---

## F. 工程/协议注意（论文 agent 勿写错）

1. **POPE `question_id` 非唯一**（random/popular/adversarial）；trace/`sample_id` 必须带 category（已修：`67f9e69`/`9c4504c`）。官方 ACC 不受影响；错配图/错配问必须用唯一 sample_id。  
2. **S5/S3/S4** 曾在 GQA 末段因 content-mask 滤空 question tokens 硬失败；已与 QA `empty_question` 对齐为回退 Native keep（`bfc2577`）。S5 GQA=53.34 是完整重跑后的分。  
3. **A2 172/120/20 W2048** = USER_REPORTED，本包装未宣称已复现。  
4. **B1 W1808 / B2 W2288** = 非同预算诊断，禁止写成同成本优越。  
5. CDPruner / STAR-Pro 表中数字来自 **另一推理路径**（handoff §B），不可无标注地当同系统对照。  
6. Evidence Guard / risk probe **未批准**进生产叙事。  
7. P0 源码摘录已有：`NATIVE_COVER_EXACT.md`、`NATIVE_VS_QA_OBJECTIVES.md`、`STAGE1_SOURCE_CHAIN.md`。

---

## G. 建议论文可写 / 不可写（截至今日）

**可写（有本机门控或六任务支撑）**
- Dual COVER → 有界 QA 修正相对 Native / random / query-only 的必要性方向（S0/S5/S6/S7）。  
- B 的 U 型/权衡：R6 在 B8 近最优，更大 B 伤害保护集。  
- Stage2：W2048 上 RDRIFT first 相对纯 Q_ATTN 的小幅 R4 增益；W1024 上 QA 入口下二者接近。

**不可写 / 须降级**
- 「多次完整 COVER」—— E.1 门控与单次 B8 相同，缺差异 traces 时只能称 **有界查询修正过程**。  
- 「Query Top-K 即可」—— S5 GQA 失败。  
- 「RDRIFT 在所有预算碾压 Q_ATTN」—— 与今日 W1024 P3 不符。  
- 未测完的 S3/S4/S8、TTFT/Fixed32、bootstrap CI、外部 CoIn/SCoRe 复现 —— 勿当已完成。

---

## H. 仍未完成（交付时状态）

**在跑 / 排队（约）**：S3/gqa、S4/gqa、S8 门控；Top-B L2 补 MMMU/AI2D/VizWiz；E3 B6/B10 门控与九任务；E3 B8 补 3。  
**未开**：P4 TTFT/Fixed32/selector 耗时；配对 bootstrap；换入 token 反事实存活；真实 traces→Fig 1–6；独立开发集稀疏参数；外部强基线同机复现。

---

## I. 对照路径速查

| 内容 | 路径 |
|---|---|
| 可执行协议 | `experiments/bela_stage1_decision_package/CURSOR_HANDOFF.md` |
| 主队列（含早停） | `experiments/bela_stage1_decision_package/launch_master_queue.py` |
| Stage1 / S5 实现 | `experiments/elva_r3/llava_stage1.py` |
| Stage2 overlay | `experiments/elva_r3/apply_llava.py` |
| 对话纪要（RDRIFT 证据） | `exports/ViTCoP_AGENT_WORK_CHRONICLE_20261030.txt` §C |
| 本文件 | `experiments/bela_stage1_decision_package/PAPER_AGENT_HANDOFF_RESULTS_20261030.md` |

---

*生成时间：2026-10-30 19:40 CST。数字以对应 `metrics.json` 为准；若队列随后跑完 S3/S4/S8/Fill3，请用新 metrics 覆写 §C/§H，勿改 Dense_ref。*
