# BELA / ELVA — CVPR 2027 最终冲刺 Cursor Handoff
版本：`FINAL_FREEZE_V1_2026-10-25`；工作周：2026-10-25 至 2026-10-31。
**这份文件是本轮唯一执行入口。** 旧的 136-arm 参数搜索、Evidence Guard、ICER 扩展等交接书全部降级为历史记录，不得混用或自动接入。

## 0. 总目标和限制
- 必须力争本周完成**关键实验和 8 页 CVPR 风格论文初稿**，但不得用虚假的成绩/新颖性保证 CVPR 接收。
- 四项核心绝对分：GQA / POPE F1 / MME Total / MMBench。ScienceQA-IMG、TextVQA 是不可忽视的安全监控。VizWiz 评分协议未审计前单列，不纳入优化。
- 多预算 W1024/W2048/W4096，**W2048 是确认主线，W1024 是高压缩机制检验，W4096 是最后可选泛化**。
- 真实速度优先级与 ACC 相同：必须统一实际 TTFT（注明是否包含 ViT）、固定 32 个输出 Token 时间、选择器开销、峰值显存；`sum K_l` 和 `sum K_l^2` 是代理而非整模型 FLOPs。
- 每个可保留模块必须有**独立消融、跨 Stage1/Stage2 交叉消融、与随机/廉价方案的对照**。
- 审计 Agent、用户或其他智能体给出的方案都只是待检验假设；只有真实源码和实验能升级决策。
- 代码默认 OFF，不绕过物理剪枝/KV/位置编码检查，不允许隐含未来层信息泄漏，不允许直接在最终测试集上选择参数。

## 1. 目前可依赖的实验事实（来自聊天报告；本包并未在 GPU 复现）
- A0 (172×4+90×8+32×20), W2048: GQA57.84 / POPE83.31 / MME1767.8 / MMB62.20 / SciQA68.96 / TextVQA55.97; TTFT123.7ms / Fixed32 1234.6ms。
- A2 (172×4+120×8+20×20), W2048: GQA57.62 / POPE85.44 / MME1774.9 / MMB63.14 / SciQA68.67 / TextVQA54.89; TTFT122.5ms / Fixed32 1217.7ms。
- A3 (172×4+130×8+16×20), W2048: GQA57.35 / POPE85.68 / MME1775.9 / MMB62.80 / SciQA68.67 / TextVQA54.88; TTFT120.3ms / Fixed32 1177.1ms。
- B1 (172×4+90×8+20×20), W1808: GQA 未在聊天中确认；POPE83.45 / MME1747.8 / MMB62.37 / SciQA68.86 / TextVQA54.51；延迟未确认。
- B2 (172×4+120×8+32×20), W2288: GQA58.44 / POPE85.36 / MME1804.2 / MMB62.80 / SciQA68.37 / TextVQA55.13；延迟未确认。
- W2048 新扫到 `172/125/18`: GQA57.47/POPE85.48/MME1786.7/MMB63.06。其R4与A2近乎相同，**不再大量扫 Kmid**。
- W1024 Wave-1 六任务 mean6 报告约88.8–91.9%，部分任务当时尚在运行，必须拉现役完整结果表。
- CoRePrune K32 与 QCPruner K32 是不同算法和不同Decoder W；STAR-Pro T32、T64的层平均预算才适合与本研究的 W1024/W2048 方法框架对照。不可混用 MME Total 和 Perception。

## 2. 本周唯一算法研究主线
**条件化视觉证据退休（Conditional Evidence Retirement）：在同预算下，哪些 Patch 需要在中后层多存活一段，何时可安全退出？**

先运行两个高信息量实验：
1. `E1_RET32_TO12`：A2前12层完全相同，后段 `32×8 + 12×12` 替换 A2 的 `20×20`，依然 W2048。
2. `L2_LATE_64_17_10`：W1024, `64×12 + 17×8 + 10×12`，借用 STAR-Pro **日程形状作为诊断对照**，保留 BELA Stage1/真实选择器，不能把对手日程当作原创算法。
旁边有 E2/E3、L3 的少量对照用于判明是否有连续、可解释的趋势。详见 `config/experiments.json`。

**不要直接实现新 Guard。** 只有确认在固定的 stage + K 下存在可修复的关键证据误删，才允许探索廉价、有限的候选保护诊断 `tools/retirement_probe.py`。其性能必须胜过随机交换及纯注意力，同等真实 W/延迟，否则放弃。
**若 E1/E2 在同 W2048 改善关系推理而不损害 POPE/OCR，可选择一种最简退休方式冻结架构；否则转向 W1024 延迟首次裁剪和 Stage1 入口质量。**

## 3. 工作周安排（所有日期按任务实际进度调整，但任务冻结优先）
### 10/25，P0 — 必须今天完成
- 读取现役 `git SHA`、checkpoint、evaluation SHA；补齐 B1 GQA、B1/B2 延迟、Wave-1 W1024 全部日志。
- 用 `CUT_CALL_MAP.md` 写出**第4/12层到底由谁排序**：Q-Attn、RDRIFT 或其他；附实际 traceback/路径/函数。若错位，停止此前相关算法效用结论。
- 输出 3 条真实预填充 `PATCH_TRACE.jsonl`，检查每层 K、原 Patch IDs 单调子集、位置索引、KV-cache、prefill/decode，以及阶段切换。
- 验证 A2 TTFT 123ms 是否**真的包含视觉编码器+projector+LLM prefill**。如果只是 decoder，另测真正端到端。
- 并行创建 CVPR 官方模板和论文骨架，写现有 Method / Setup / Related Work（未证实结果用 TBD）。

### 10/26，P1 — 进攻 W1024 与 W2048 的时间位置
- GPU0: W1024 `L2_LATE_64_17_10`；GPU2: `L3_MID_80_24_12`；GPU4: W2048 `E1_RET32_TO12`；GPU7: W2048 `E2_RET32_TO16` 或 STAR-Pro 官方同机基线。
- 先使用冻结的**开发筛选子集**，不允许在完整正式测试结果中迭代阈值。
- 先比较 GQA/POPE/TextVQA，确认后再加 MME/MMBench；同时记录实际 wall time 和选择器时间。
- 有条件补 E3/L4；C1 要新增一次真实剪枝事件，必须先通过 smoke。

### 10/27，P2 — 决定问题究竟在“身份”还是“时间”
- 对两预算的胜出候选做**固定实际 Stage1 Patch 候选集、固定W、固定切层**的第二次剪枝选择器消融：现役、Q-Attn、RDRIFT（若合法可用且不重复）、随机至少3种种子。
- 用 Dense/A0/A2/B2 胜负配对和 TextVQA 细分错误、POPE 的 TP/FP/TN/FN 分析是否误删完整对象/文字证据。
- 有一条核心假设成立才开 `P3`：同日出 `STOP_GO_P2.md`（严禁只有最高分而没有解释）。

### 10/28，P3 — 唯一一次最小算法改动机会
- 如果“最终 K 的身份选错”被验证，复用 Stage1 双层 COVER 特征 + 当前可用的查询相关性，探索**单次有界候选纠正**，避免 ICER 式昂贵重复重算。
- 如果“证据必须额外存活几层”被验证，冻结最简 E1/E2 或 W1024 晚剪枝机制，**不要叠加 Guard**。
- 如果两条都未证实，停止新模块；审计入口候选信息损失，保留强固定基线并写清论文边界。
- 新增功能独立开关默认 OFF。提交 3 条 smoke、Source Diff、Patch ID Trace、p50/p90 selector/TTFT 后再开始确认实验。

### 10/29，P4 — 冻结最终算法（不可再改超参数）
- 对唯一候选跑 W2048 + W1024 的 GQA/POPE/MME Total/MMBench/ScienceQA/TextVQA + AI2D；根据资源再验证 W4096。
- 如果能做，追加一个不同 backbone 的少量外部验证（至少 GQA/POPE 或等价任务）。否则如实写单 backbone 局限性。
- STAR-Pro 官方代码在相同 checkpoint、相同评测器、相同实测W/真实端到端计时下优先复现。
- 保存最终选择的**事先隔离确认集**结果，算法冻结后不得重新用它调参。

### 10/30，P5 — 完成可靠证据
- 固定算法，补2×2 Stage1×Stage2 交叉消融：双层 COVER vs 预先指定的单层；基线Stage2 vs 最终 Stage2，同 W 同层位。
- 完成 RDRIFT at each cut、固定比例 vs 退休机制、等预算、随机对照与真实 latency 的消融。
- 输出成功和失败样本可视化，正文实验表、机制曲线、消融表同步归档。
- 论文主文第一次完整通读，做 claim-to-evidence 检查。

### 10/31，P6 — 论文冻结版本
- 8页CVPR格式正文初稿完成，主要图表都有来源，所有数字由表或原始日志复算，References、Limitations、Supplementary 目录齐。
- **必须加入 CVPR2027 LLM Usage Statement**；所有作者 OpenReview profile 和 enrollment 状态检查；准备11/10 AoE注册、11/16 AoE全文，不假设截止延期。
- 交付：`FINAL_METHOD_LOCK.md`、`RESULTS_MASTER.csv`、`SOURCE_DIFF.patch`、`TRACE.jsonl`、`LATENCY_RAW.csv`、`PAPER_DRAFT`、`CHECKLIST_CVPR.md`。

## 4. 4张GPU并行和自动晋级
GPU0 优先 W1024；GPU2 优先 W2048 机制日程；GPU4 选择器/定性诊断；GPU7 官方强基线 + 统一计时。若某机不兼容具体任务，允许重排，但不能遗漏 workload 标签。
优先允许跑 P0、P1。只有 P1 提供较高收益且通过真实延迟门槛，才提升 P2/P3。`config/experiments.json` 中 GATED 组不能默认海量派发。
记录 RunID / time / SHA / GPU / dataset hash / actual W / latency 协议，每4小时生成一次进度汇总与 failed retry 记录；只依赖 Cursor 正在运行的服务器调度器，ChatGPT 本身不负责后台工作。
若两条算法路线都失败，优先保证论文可复核性，不允许捏造贡献或“领先 SOTA”数值。

## 5. 可运行文件
- `python -m unittest discover -s tests -v`
- `python -m tools.schedule config/experiments.json`
- `python -m tools.results <官方评测base.jsonl> <候选.jsonl> [--pope]`
- `config/experiments.csv` 是候选研究矩阵，不是已做GPU实验；状态字段 REPORTED/NOT_RUN/GATED。
- 真实生产集成：Cursor 先审核 `docs/ENGINEERING_CONTRACT.md`，确保真实 cut 能力后写最小补丁，不能假设仓库代码接口与历史E014完全相同。

## 6. 论文科学风险（不可忽略）
- “深层多Token伤害答案”未被证明：A2同时加中层减深层，B2增加深层还提高GQA。
- 动态K、覆盖+渐进、联合删除和文本吸收都已有文献。新方法必须是**具体、低成本、经过证实的退休决策**，而非重新命名。
- 如果仅固定日程明显提高ACC且算法缺新颖性，应诚实降低贡献强度；“必须提交”不能等同“保证主会录用”。
- 核对CVPR 2027 Author Guidelines：8页正文（参考文献和LLM声明可另页）、匿名、profile、LLM Usage Statement、OpenReview 的作者截止限制。
