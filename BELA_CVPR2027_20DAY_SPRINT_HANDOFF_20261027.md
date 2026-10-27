# BELA / ELVA — CVPR 2027 最后冲刺交接书（BELA 主线）

版本：2026-10-27 / `bela-sprint-v3`。**本文件立即取代此前 `CURSOR_BELA_RESEARCH_DELTA_20261026.md` 的 CDPruner 优先实验队列。**与上传的 `BELA_FINAL_HANDOFF_20261026.zip` 一起使用；继续沿用原包的身份映射、计时版本、评分定义、离线参考测试和安全 GPU 调度契约，但以这里的最新停机和优先级为准。

## 0. 研究立场（不得歪曲）

- **投稿方法是 BELA，CDPruner、STAR-Pro 等是外部论文的强基线，不是我们的新模块。**已跑完的 CD32/K64 可以比较或少量离线诊断；**停止把 CD64_L2 移植、CD 符号消融、CD 新混合结构当作当前 GPU 主要工作。**若服务器上 CD64_L2 已跑完，收结果归档即可，不重复排队。
- BELA W2048 的 E3 Dual (`172×4→120×8→24×10→16×10`) 已有 mean6 **96.85**，属于必须保住的强锚点。冻结其模型与生产默认路径，勿为了救 W1024 扰动 W2048。
- W1024 L2 (`64×12→17×8→10×12`) 的 mean6 **93.56**，GQA **55.57**、POPE **82.21** 是主要短板；当前 W1024 强对手 CD32 mean6 **94.41**、GQA **57.00**、POPE **87.67**。**我们的 W1024 主目标是进入 94.5–95 左右、缩小 GQA/POPE 缺口，并守住 MMB/Sci/TVQA。**不得按任务选择不同配置后拼一个伪主表。
- `L2_ADMIT_B2` 六任务 mean6 **93.51**，未超过 L2；`L2_L9_ONLY` 为 **91.83**。**入口新准入 A、增加 b、简单单 L9 替代停止；不自动推进 AS。**若 S-only 在剪点有真正动作和答案恢复，可单独保留为 *条件* 候选。
- 不承诺“Stage2 是根因”：目前 CD 固定32和 BELA K64 分层 L2 同时改变了入口策略、入口 K、剪点和末层 K。先用 BELA 自己的可控实验辨识问题。
- 截稿：官方 CVPR 2027 **11 月 10 日 AoE 注册/摘要**，**11 月 16 日 AoE 全文**。**今天并行开写论文，不等 ACC 都跑好才写。**

## 1. 一个足够具体的 W1024 判别问题

`W = Σ_l K_l`，L2 和固定32都恰好1024，但相对 token-lifetime 不同：

| 轨迹 | decoder 32 block 内的 K | W | 前12层 W | 后20层 W | 末段 K | 视觉 `ΣK²` |
|---|---|---:|---:|---:|---:|---:|
| `L2_NATIVE` | 64×12,17×8,10×12 | 1024 | 768 | 256 | 10 | 52664 |
| `BELA_COVER32_FLAT` | 32×32 | 1024 | 384 | 640 | 32 | 32768 |
| `BELA_S16` | 64×8,32×8,16×16 | 1024 | 640 | 384 | 16 | 45056 |

`ΣK²` 不是完整 FLOPs 或速度；真实成本包含文本、KV 和选择器。三个日程的 32 层总量与 W 已离线枚举/核实。

**直接可实现的工作假说：**BELA 双深度入口的高价值候选在 W1024 下可能过早被原 Stage2 清除，或过早把预算花在第一阶段；若成立，应以更保守且后层有下限的生命周期或小规模 *同容量* 选择规则修复，而不是增加候选准入 A。

保留有力反例：W2048 E3 前置预算也多却综合表现强；CD K32 在 POPE 上略高于 CD K64；STAR T32 的 POPE 不差。故**不是简单“保留越久越好”，必须从真实配对错误证明**。

### 关键数量关系（目标不是幻想）

以原固定参考分母计算，如果 **只将 L2 的 GQA=55.57、POPE=82.21 提升到本地 CD32 的 57.00 / 87.67，其余四项仍保持 L2**，则 mean6 约为 **95.01**。这是可解释的性能靶线，**不是已达到的实验结果**。实际还要防止 MMB/Sci/Text 的损失。

## 2. 两天内必须跑出的三条 BELA-only 先导（最多新增三个正式 arm）

所有方法都沿用 BELA 当前图像输入、现役双深度 guide、真实 projector 来源、LLaVA 权重、生产解码器与同评测集；只修改明确允许的一处。冻结图像 ID 和 question ID，先优先 POPE/GQA 加小样本 MMB/TextVQA 保护，随后仅晋级者全六任务。

其中 B1 对应旧包已登记的 `COVER32_FLAT` 实验 ID，可复用其配置/门禁（若服务器已有合法相同运行，直接复用）；B2、B3 为本交接新增 ID，需在现役 adapter 中注册并经过最小 smoke，而不是覆盖旧定义。

| arm | 唯一变化 | 预期能排除或支持什么 |
|---|---|---|
| **B1 `BELA_COVER32_FLAT`** | BELA 当前 Dual COVER Stage1 改为真实 K32，进入 LLM 后所有 32 层固定 K32，无 Stage2 删除 | **同 W 下是否仅改变 token 生命周期就能改善 POPE/GQA**；它仍是 BELA 自己的入口而非 CD |
| **B2 `BELA64_ENTRYORDER_L2`** | 同 `L2_NATIVE` 的 Dual K64 入口和完全相同的 `64→17→10` 日程；只在两个剪点用入口 COVER 贪心边际顺序优先保留当前 alive 中排名最高的17/10个 | **原生 Stage2 attention 选择规则是否比 BELA 自己的入口信息更差**。必须保存真正的 COVER greedy 顺序，不能拿按空间 ID 排序后的前17个 |
| **B3 `BELA64_S16_NATIVE`** | 保持 Dual K64 入口、现役 Stage2 score 语义；日程唯一改为 `64×8→32×8→16×16`（剪点在第8和16个 block 后） | **后层证据容量不足/早期过度前置是否为低预算瓶颈**；属于可证伪的 schedule 先导，不预先认领投稿创新 |

如果 B2 不能在现有 adapter 里合法取得 Stage1 原贪心边际/排序，先证明这一点并跳过，不能创造虚构的 entry score。B3 第8层就剪，可能损害早期视觉融合；GQA/POPE 不升或 Sci/MMB 明显受损就立即停止，不增加第二十种日程。

**已有强基线 `L2_NATIVE`、`E3_DUAL`、`CD32`、`STAR_T32` 直接复用同协议预测；不要重复烧卡重跑已经通过输出一致性检查的锚点。** 服务器上若确有完成的 CD64_L2，只把它用作附加诊断，不能挤占 B1–B3。

### 小样本先导的晋级规则

1. 优先看 GQA 与 POPE 配对效果；大致 **GQA +0.5 或 POPE +1.0 以上**才能值得扩样；这只是算力筛选阈值，不是论文显著性判据。
2. POPE 必须同时交出混淆矩阵和 yes-rate；若 F1 提升来自更偏向 Yes，而 FP 没降，不能讲减少幻觉。
3. MMB/TextVQA 约 -0.3 或更多退化要先复核；ScienceQA/MME 的安全项在最终六任务完整报告。
4. 只有全六任务的实际 `mean6 > 94.41`、稳定缩小弱项差距且效率可信，才有资格称其比当前同环境强基线竞争。**95 是追赶目标，不是四舍五入的门槛。**

## 3. 依结果只允许一次 BELA 小修复（不重新造三层框架）

- **B1 最好**：说明低预算最可能受不合适的 Stage2 生命周期影响；用真实 cut 干预确认。在投稿主线中考察 **budget-conditioned BELA schedule**（2048 固定 E3，1024 按预算固定 policy），但说明“单纯固定 K”本身非新颖；必须有双深度/删前证据的机制结论，否则不足以独立投稿。
- **B2 最好**：Stage2 score 与 Stage1 信息可能不协调。仅实现一个 **同容量、无重入** 的 scorer：用生产真正规范化的当前剪点 score 加一个**小的**、从 BELA 已有 Dual COVER 边际导出的先验；固定一个权重做先导，不大扫 λ。做来源打乱/随机保留等匹配负控。若 S-only anchor 在剪点无动作，记录并停止，不硬造效果。
- **B3 最好**：低预算的跨层暴露时长很可能是主要瓶颈。保持 BELA 原来的双深度入口与 native score，冻结 S16 进六任务、延迟和外部模型；不继续无目的扫 cut 位置。还须用一个剪点答案级干预支持“证据过早消失”。
- **三者都不行**：至迟 11/01 **停止算法新搜索**。保留 W2048 的真实优势和 W1024 的诚实负结果；将算力转到强对照、外部模型与写作。没有与现有论文足够区分的机制证据，不能把一次 schedule 调整夸成 CVPR 核心创新。

## 4. 重点诊断：POPE 为什么掉五分，GQA 是什么题在掉

先从已有 L2/CD/STAR 原始预测运行分析，**不要等 B1–B3 跑完**：

- POPE：三个官方子集、positive/negative、FP/FN/TP/TN、precision/recall/specificity、回答 Yes 比率、解析 invalid；按 image 聚类 bootstrap，每次重算 F1。
- GQA：利用当前数据集允许的类别，分 object/attribute/relation/count，记录 L2 独错、CD 独对、两者都错及相反样本；按同一 question ID 配对。
- BELA 删点 trace：保存最少量的原始 patch ID、Stage1 来源/guide 边际、剪前 alive/剪后 survivor、当前 attention/native 排名；对失败样本做 **固定同前缀且同容量** 的单剪点答案恢复实验。Dense/答案标签只能作诊断 oracle，不能喂给在线 scorer。
- 如果 native score 提升、来源打乱结果相同，那么“深度来源传递”没有被验证，不写成贡献；如果随机剪枝也一样答对，方法没有在困难题上证明 token 选择能力（参考 PAA-Bench）。

## 5. 论文主线建议（从今天开始，并行于实验）

**暂定问题，不是事实宣言：**

> Under severe visual-token budgets, pre-LLM coverage alone may not translate into evidence available for later language decoding: can BELA preserve complementary dual-depth visual evidence *when it is needed* without abandoning its high-budget efficiency?

若证据支持，故事顺序必须是：

1. 已有 BELA **W2048 强而 W1024 的 GQA/POPE 急剧衰减**，展示六任务和成对错误，不拿外部 CD 方法归为自己的创新。
2. B1/B2/B3 因果拆分身份、后续选择和生命周期，找到能明确复现的机制窗口；不把 attention 热力图当答案因果证明。
3. 基于 **BELA 两深度 guide 的互补性**、真实 decoder 可用信号，提出**一个**小修复；给出 matched-random/shuffle/score-only 的严格对照。
4. 主表分别给 W1024 与 W2048 的单套配置完整六项、真实 latency/Pareto、稳定性与第二 backbone 验证。
5. Related Work 明说 2026 HAP、HiDrop、STAR-Pro、CoRePrune、AgilePruner、Reroute/CatchUp 已做 head reliability、渐进日程、条件删点和重入，不认领先前已有的技术。

**只有日程得到一个百分点提分，不足以构成扎实的 CVPR 方法贡献。** 必须回答“为什么 BELA 的某些来源 token 在严格低 W 下容易失去答案价值，并且我们的修复确实基于这个机制”。目前没有证据保证这个叙事能成功。

### 强制写作与冻结时间线（美国太平洋时间；投稿截止 AoE）

| 日期 | 不可拖延的交付 |
|---|---|
| **10/27 当天** | 冻结 E3/L2 锚点和六任务 scorer；启动 Abstract 问题陈述、Introduction 骨架、Related Work、Evaluation Protocol；复用旧预测做 POPE confusion/GQA 配对 |
| **10/28–10/29** | B1/B2/B3 小样本 GQA/POPE＋MMB/Text 安全筛；选唯一胜者/机制方向；开始 CVPR 模板正式排版 |
| **10/30–11/01** | 最多一次有证据的 BELA 小修复＋匹配负控；**11/01 方法冻结**（若无益处不再扫参） |
| **11/02–11/05** | 赢家完整六任务，两预算主表，可靠预测哈希、当前代码独占 TTFT/Fixed32、主要机制图 |
| **11/06–11/09** | 第二 backbone/额外任务，置信区间与消融，完成主体初稿和 Limitations |
| **11/10** | OpenReview 注册/摘要、作者资料与名单确认，不能错过官方截止 |
| **11/11–11/15** | 不再修改核心方法，写满实验/附录、校对图表、匿名性、数据协议、逐表复核 |
| **11/16 AoE** | 正式提交全文；附录材料可在官方 11/23 AoE 截止前补齐，但主文必须已自洽 |

## 6. 实验产物与安全约束

输出可追溯的 `run_meta.json`（commit/checkpoint/scorer/ID hash/precision/backend）、`metrics_6task.json`、`pope_confusion.json`、`gqa_paired.jsonl`、`cut_trace.jsonl`（仅小样本）、`latency_current.json`（同修复后版本、独占计时）。所有新增 arm 默认关闭，生产路径不替换，工作 GPU0/2/4/7 前先看作业占用；绝不清理或中断正在运行任务。

现有 ZIP 的离线 27/27 tests 成功只验证定义，不代表部署 GPU 正确。必须 smoke 验证 mask、position、KV、真实每层 K 和原始 Patch ID；不把原始 `mean6` 评分除数更换成方便达标的新定义。

## 7. 本阶段仅需继续跟进的优先论文

- [HAP 2026](https://arxiv.org/pdf/2608.23921)（head quality/decoder scoring），[HiDrop ICLR 2026](https://arxiv.org/pdf/2602.23699)（分层预算和选择点）；用于诊断低 K 下 attention/reallocation 失败，不抄作新贡献。
- [STAR-Pro 2026](https://github.com/EasonAI-5589/starpro)（两阶段进化）；[CoRePrune 2026](https://arxiv.org/abs/2609.26484)（条件可删除性）；[AgilePruner ICLR 2026](https://github.com/cvsp-lab/AgilePruner)（复杂度/多样性）；[PAA-Bench 2026](https://arxiv.org/abs/2610.24740)（严格评估）；[CatchUp 2026](https://github.com/pej0918/CatchUp)（不要再用重入当首创）。
- [CVPR 2027 官方截止](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) 和 [作者规则](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)。

**给 Cursor 的最终执行话语：只推进 BELA 的 B1/B2/B3，收敛而不扫参；CDPruner 仅存为 baseline；今天写论文、11/01 冻结新方法、11/16 全文交卷。**
