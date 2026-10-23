# 实验执行手册（按顺序，不允许跳关）

## P0 — 现役代码映射 + 真实效率

- 核对 git commit、checkpoint、vision tower、图像预处理、decoder 层数、prompt、eval/answer-parser sha、POPE 三子集和 TextVQA-OCR 版本。源代码中如有 E006/E007/E012/E014 多份分支，找出真正调用位置，而不是只看文件名。
- 每个 cut：`after_layer_index`（0-based layer 完成后），输入/输出原 Patch ID，score 生成函数与来源层，文本 Query token 范围，位置编码/mask/KV Cache/physical gather，是否可通过相同图片多次复现。
- A2 容量表：0–3层 K172；4–11层 K120；12–31层 K20。`after_layer_index` 应为 3 和 11。先确认模型真的按这一解释执行。
- 对 A0/A2/A3 的实际 `tokens_per_layer` 用 `scripts/audit.py` 校验。手工核对物理 KV 与动态 cache，不能把 32 个列表数字误当真正的加速。
- 同硬件分别测 repeated prefill TTFT、固定32 token decode、选择器独立开销、峰值显存；同步 CUDA；保存所有重复实验、warmup、显卡状态。报告 p50/p90，不用单次最佳。

## P1 — B1/B2 机制干预（不等 W，绝非论文公平排名）

| Arm | K0 / Kmid / Kdeep | W | 问题 |
|---|---|---:|---|
| A0 reported | 172/90/32 | 2048 | 原始 |
| A2 reported | 172/120/20 | 2048 | 新主基线 |
| B1 **NOT_RUN** | 172/90/20 | 1808 | 固定中段，缩小深段是否改善 POPE？ |
| B2 **NOT_RUN** | 172/120/32 | 2288 | 固定中段，增大深段能否恢复 GQA/TextVQA？ |

先只跑 GQA / POPE / TextVQA；每题完整保存 sample/image ID、原始 prediction、官方 evaluator 的 `correct`；严格配对：A0 vs B1、A2 vs B2、A0 vs A2。对 POPE 必须拆出 random/popular/adversarial 的 TP FP TN FN，P、R、F1，检查回答 Yes/No 偏置；对 GQA 统计关系/属性/计数/存在性；TextVQA 统计文字尺度、多文本框、定位类型。

**双因素 interaction**（每任务）：`Y(B2) - Y(A2) - Y(A0) + Y(B1)`。它刻画容量参数组合的交互，而非独立的语义因果证明；B1/B2 预算不同且后续 patch 集合不同。需逐样本 paired wins/losses、按图像聚类 bootstrap；避免用小幅点估计过度下结论。

## P2 — 只根据 P1 选一个分支

**若 B2 在 GQA/TextVQA 恢复明显，且 POPE 合理：** 研究“深层仍需一些证据”的机制，对 C1 提交最小 diff。C1 为 `172×4 + 108×8 + 32×8 + 20×12 = 2048`，在第 20 层后新增第三次剪枝，需要 E3 的 score、原 Patch ID、cache 正确性和新 selector 开销；与 A2 不能只靠 W 相等自称延时相同或单独证明存活时间。

**若 B2 不能恢复性能而错误集中在 ID：** 对 **A2 的真实第二次裁剪选择器**比较现役方法与 Q-Attn / Random 多种子 / RDRIFT（若现役该位置并非 RDRIFT）；全部固定候选池、K20、日程、W。先 trace，再 ACC 和 TTFT，不直接加入 Evidence Guard。

**若两种容量干预都无稳定收益：** 停止扫描 mid/deep 比例，检查 Stage1 COVER 构建及首次 E1 误删是否不可逆。C1 暂停。

## P3 — 强对手与论文

STAR-Pro 官方 LLaVA overlay 作为首要同机对照。公开日程 `128×12 / 37×8 / 18×12`，W2048，论文报告 GQA59.41、POPE87.52、TextVQA56.26。没有相同数据/模型/评分器时必须标 PAPER_REPORTED。

MiCo 也公开了官方代码，时间允许时重点学习 Stage1→Stage2 的可观测目标与成本，且其新颖性冲突必须写进 related work。模型与预算不同不能直接打 SOTA 结论。

**提交**：`BASELINE_LOCK.md`, `CUT_CALL_MAP.md`, `TRACE.jsonl`, `RESULTS.csv`, `PAIR_DIAG.json`, `POPE_DIAG.json`, `LATENCY.csv`, `SOURCE_DIFF.patch`, `STOP_GO.md`。默认禁止生产改动，若新模块只略微提高均值而使 GQA/OCR/延迟恶化，则 REJECT。
