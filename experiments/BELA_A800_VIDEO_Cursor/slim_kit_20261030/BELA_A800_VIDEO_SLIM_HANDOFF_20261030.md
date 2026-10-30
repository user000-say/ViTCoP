# BELA / A800 视频主表实验 — 极简执行 Handoff
**版本：2026-10-30**  
**状态：供 Cursor 直接执行的精简协议；不是已运行的 A800 日志。**  
**这份 Handoff 取代之前要求 7 臂 × 多个负控 × 多项时序诊断的重型视频计划。**

## 0. 唯一任务

视频作为 BELA 论文的**跨数据集泛化及效率验证**，不是另起一个视频算法研究项目。

只交付：
1. Video-MME（w/o subtitles）Accuracy；短/中/长分项；
2. MVBench overall Accuracy（20任务单独分数可放附录）；
3. 同卡 A800 的 one-token TTFT/模型前向时延、Fixed32（可选但推荐）；
4. 同协议 operator-counted FLOPs（vision + LLM prefill + total），以及名义视觉预算、真实 Token 数；
5. 峰值 GPU 显存（已有 profiler 可顺带，非新增消融）；
6. 以上结果的 Dense 对照与 speedup/FLOPs ratio。

**若 MLVU 或 LongVideoBench 已具备正确的缓存及 lmms-eval 配置，可以加跑1项；否则不因补数据推迟图像论文。**  
不做长篇负控、视频保护新模块、多个 F 候选、广域 B sweep、跨模型复现。

## 1. 已核实的 STAR-Pro 对齐重点

原论文：https://arxiv.org/html/2609.05916v1  
公开仓库：https://github.com/EasonAI-5589/starpro

- **320是 LLaVA-NeXT 图像的层平均视觉预算；不是 STAR-Pro 的 LLaVA-Video 视频预算。**
- STAR-Pro 视频：LLaVA-Video-7B，**64 frames**，每帧原始 169 patches，LLM 28层；采用每帧名义 T=32 / T=16 两档层平均预算。
- **T32**：每帧 Stage1 `K0=64`；经过 Decoder 第12层 `K1=11`；经过第21层 `K2=4`，余下7层保留4。视觉 Token-layer / 帧 = `64*12 + 11*9 + 4*7 = 895`，层平均 `895/28 = 31.9643`。
- **T16**：每帧 Stage1 `K0=32`；经过第12层 `K1=5`；经过第21层 `K2=3`。视觉 Token-layer / 帧 = `32*12 + 5*9 + 3*7 = 450`，层平均 `450/28 = 16.0714`。
- 图像模块的 `K64/B8/64→17→10` **不能原封不动复制到视频**。此处提供 STAR-Pro 视频日程的 BELA 候选，**属于新视频运行协议而非已有 A800 产线配置**，启动前必须记录。
- 主准确率 Video-MME = without subtitles；STAR-Pro 另外在 MLVU (m-avg)、LongVideoBench (val) 报告主表。MVBench 是我们为了快速扩展到时序能力的补充。
- STAR-Pro A800 视频速度测试：单张 A800-80GB、64帧、batch=1、10 warmups / 50 measured、`max_new_tokens=1`，CUDA events+同步；**不计 CPU 端视频解码与预处理**。
- STAR-Pro `operator-counted TFLOPs` 是 PyTorch matmul/Conv2d 的算子统计；不完整计入部分 fused attention 和 top-k/QR。因此要另外报告真实运行时间，不准把这一 FLOPs 数当模型全算子理论 FLOPs。
- 公开 starpro 仓库是图像 LLaVA overlay，**论文测试了 LLaVA-Video，但视频适配代码未包含在该 overlay 内**。不能假装现成一个 `T=32` 就可以跑视频 STAR-Pro。

## 2. P0 只做必要的实现核验（不能省）

A800 `node-250` 的 checkpoint、processor、F、每帧 patch 数、decoder 层数、模型代码、帧采样/帧时间、原有视频 `Stage1 native/qacover`、QA 交换口径须在生产 worktree 自动审计。当前上传文本**没有现役 A800 模型源码**，旧 Dense≈63.26 不能用于推定模型身份。

**如果且仅如果是上述相容的 LLaVA-Video-7B 28层/169 patch 路径**，采用 §1 的精确 64帧 T32/T16 日程。

不相容时：
- 若 64帧输入不能正常处理，必须先基于视频模型官方协议冻结一个实际 F；绝不把 8/16/32帧成绩直接写成与 STAR-Pro 的同协议比较。
- 若 decoder 不是28层，**不得盲用第12、第21层**；以实际架构设计匹配层平均预算、明确实际分段长度与公式，再冻结。
- 若现有 QA 是**整视频总 B8**而非**每帧 B8**，不能悄悄改定义；首先输出准确的 swap scope。为了对齐 STAR-Pro 的 per-frame 预算，首选 per-frame B，但前提是接口可行、真实成本可接受；改动算新视频适配协议，要记录实现。
- 若视频输入具有全局时间标记/字幕/特殊 Token，物理剪枝绝不能删掉这些非 patch token，帧索引与 position/cache 必须正确。

Stage2 当前用户报告为**未挂钩**。首先完成实际物理剪枝接线，仅提供三种模式：
- `OFF`：无 Stage2（Dense 必须 Stage1 也 OFF）；
- `Q_ATTN_ALL`：两次都按当前 question→visual attention TopK；
- `RDRIFT_FIRST`：第一次按原 RDRIFT 排名，第二次 Q_ATTN。

执行 10–20 条 correctness smoke（不同长短视频，含 cache/decode），要求：
- `OFF/OFF` 与未修改 Dense 对齐（同帧、同提示、容忍后端数值误差）；
- 两个剪点真实减少 decoder 的视觉序列长度，核实保留 ID、文本/时间 Token、位置编码、Cache；
- RDRIFT/QATTN 真正分别执行，不能仅在日志中换名称；
- 记录输入 K0、每剪点实际 K1/K2、视觉 Token-layer W、选择器耗时；
- 任何 FAIL 必须先修，不准输出完整 BELA 视频效果数据。

## 3. 极小配置筛选：只在独立 Pilot 上挑

**原则：所有候选同一模型/帧数/提示/视频输入；主要变两个轻量变量 B 与 first selector，不改评分公式；每档 T 单独选优。**  
默认先在一份**冻结的 Video-MME Pilot（例如120问，尽量复用已审计旧清单）**上运行。若旧 pilot ID、帧哈希或评分不可核验，重新冻结新 pilot。请把这些题从 Video-MME 确认集排除，避免用同一题既选参数又宣称未经接触的测试结果。

| ID | 每帧名义T | Stage1 | 每帧最多QA交换 | Stage2 | 是否执行 |
|---|---:|---|---:|---|---|
| DENSE | full | OFF | 0 | OFF | Pilot复用有效结果；Full必跑 |
| A | 32 | Native COVER | 0 | 全 Q_ATTN | Pilot必跑，防止QA反而伤视频 |
| B | 32 | Native COVER→QA | 4 | 全 Q_ATTN | Pilot必跑 |
| C | 32 | Native COVER→QA | 8 | 全 Q_ATTN | Pilot必跑 |
| D | 32 | Native COVER→QA | 8 | 首 RDRIFT、后 Q_ATTN | Pilot必跑 |
| E | 16 | Native COVER | 0 | 全 Q_ATTN | Pilot必跑 |
| F | 16 | Native COVER→QA | 4 | 全 Q_ATTN | Pilot必跑 |
| G | 16 | Native COVER→QA | 4 | 首 RDRIFT、后 Q_ATTN | **仅 D 在 T32 明显有利且低开销时** |

T32 `B8/64 = 12.5%` 最大替换比例；T16 采用 `B4/32 = 12.5%`，在容量变小后保持相近的修改权限。T32 B4 是检查较保守修正可能避免视频中 Query 过拟合。

如果现有视频 Stage1 是整视频级而非逐帧级，不可直接用这些 B 数字。首要是锁定**实际接受交换数 / 实际 K0**；仅在 framewise 版本能保证同预算时跑这里列出的 per-frame B。

**Pilot 选择规则：**
- 每档 T 选择 Video-MME Pilot Accuracy 最高且正确执行物理剪枝的候选；分数相近时，优先 Q_ATTN、较小 B、较低实测延迟；
- 若 Native 胜于 QA，就保留 Native；不得为了论文名称强行用 QA；
- 严格不拿更多 Token 的 T32 与 T16 当同成本对照；每档各选一个最优；
- Pilot 仅做配置筛选与工程稳定性检查，不作为确认性增益证据；
- 不在完整 Video-MME / MVBench 官方评测结果上反复挑 B。

## 4. 正式准确率：只跑3组即可

**Full = Dense + Pilot胜出的T32 + Pilot胜出的T16。**

必跑：
- **Video-MME without subtitles**：准确率 overall + short / medium / long；保持同一版本 `lmms-eval` / prompt / 均匀采帧策略 / 相同帧缓存；全量合法可访问样本、失败样本显式列出。
- **MVBench**：overall accuracy + 20 tasks 单项表（单项可以放补充材料）；相同模型/解析器/样本。
- **第三视频任务（建议 MLVU 或 LongVideoBench）**：仅当可直接使用已有数据与成熟评估程序时加跑；若不具备，不影响上面两项交付。

视频模型若与 STAR-Pro 不同，绝不把两篇论文表格数字直接并列为公平算法排名。STAR-Pro Video-MME w/o sub：Dense 63.6、T32 61.6、T16 60.0（仅供文献参考）。

只把 held-out 样本作为确认性评估；对于 Pilot 参与过的数据，若报告 full-overall，额外注明调参重用并单独提供 excluded-pilot 指标。

## 5. A800 latency + FLOPs（论文主表必需）

**定义两个独立评测：**
- ACC：全量 Video-MME/MVBench，正常答案生成；
- EFF：相同模型、固定64帧与视频样本面板（如50个有效视频），只跑 `DENSE + T32 winner + T16 winner`，单 A800 同 uuid 静默串行测。

**效率的必要输出：**
- GPU forward end-to-end one-token (STAR-Pro 可比口径；明示不含CPU视频解码)、`TTFT mean/median/p90`, `speedup_vs_dense`;
- LLM decoder prefill time + Stage1 total time + Stage2 score/index overhead；
- `operator-counted FLOPs`：Vision+projector / LLM prefill / overall，每组同样的 profiler 与输入；
- `visual_tokens_original`, `K0/K1/K2 actual`, `layer_average_token_budget_per_frame`, `retention_ratio`;
- `peak allocated GPU memory`（已有 profiler 时）；
- `Fixed32` 完整32-token 生成时间，推荐顺带测；若无法保证每个任务生成正好32 token，则单独标为自然结束生成时间，不能叫 Fixed32。

**计时协议**：batch1，FP16/BF16 需和正式模型冻结，预热10次、测50次，同一 A800、同一组视频、相同帧 tensor、无其他推理负载；one-token 测试 `max_new_tokens=1`，CUDA events 或明确同步的计时方式，均包含真实 Stage1/Stage2 overhead，CPU视频解码不计入可比主表。后续若测用户可感知的“含解码端到端”，另列，不能混。

**FLOPs 注意**：优先 `torch.profiler` 的 matmul/conv 统计（并与 STAR-Pro 的 operator-counted 口径说明差异）；fused attention、topk、QR 等可能不计入。严禁将 `W`、压缩率直接冒充 FLOPs 实测值。EFF profiling 最少3个有代表性视频、独立于计时，写出采样数和均值。

## 6. 排队与交付

1. 先 P0 接线及 smoke（优先 GPU0/1；只修改 A800 worktree，图像 V100 保持不变）。
2. 单独 120 问左右 Pilot 筛选 A–F，条件满足再跑 G；禁止把 A800 旧 Stage1-only 分数当完整方法分数。
3. 冻结两档赢家的 `git SHA/model SHA/F/K/B/selector/prune_layers/prompt SHA/manifest`。
4. GPU0–3 按样本分片跑 Full ACC（只3臂×2数据集；一张视频对应多个问题时保持一致的帧缓存）。
5. 准确率完成后，单 A800 quiet-run latency，另起 profiler-run FLOPs。
6. 交付一个紧凑的 `VIDEO_FINAL_PAPER_TABLE.md`，列：`Model, F, avg tokens/frame, Config, Video-MME all/S/M/L, MVBench, FLOPs(T), FLOPs ratio, TTFT(ms), speedup, Fixed32(ms), peak GB`。
7. 输出 `VIDEO_REPRO_LOCK.json`、`VIDEO_FULL_PREDICTIONS.jsonl`、`VIDEO_BUDGET_CHECK.json`、`VIDEO_GPU_LATENCY_RAW.csv`、`VIDEO_FLOPS_PROFILE.json`。有失败/缺数据必须可追溯。

## 7. 最终研究解释边界

- 若 QA 比 Native 更差，要如实报告；不为抬分改算法或者悄悄挑“容易”的视频任务。
- 若 RDRIFT 不优于 Q_ATTN，直接使用 Q_ATTN；论文可以写“视频也使用同一简单渐进剪枝”。
- 若 T32 保精度而 T16 速度最强，两档都报告，让审稿人看到 Accuracy/Latency Pareto 权衡。
- 如果 Stage2 接线失败或未发生实际物理缩短，不能将旧 Stage1-only pilot 结果伪装为完整 BELA。
- 同预算比较是同**层平均视觉 Token 预算**；FLOPs 和真实 latency 必须分别实测。
- 此处的 STAR-Pro 视频 schedule 是可公开复核的参考候选，不是已在 A800 跑通的生产配置。

**Cursor 首条执行指令：**“阅读本 slim handoff 与现役 A800 worktree/旧视频 handoff，确认模型是否与 LLaVA-Video-7B/64帧/28层兼容；执行仅必要的 Stage2 物理剪枝接线与 Dense identity smoke，记录每帧 K/B 口径。之后按 A–F（必要时 G）配对 pilot 筛选两档预算各一个赢家，冻结配置，Full 只跑 Dense+两赢家的 Video-MME / MVBench，再按 STAR-Pro A800 one-token latency 与 operator-counted FLOPs 协议测效能。不得复制旧大规模视频消融队列，不得修改 V100 图像生产实验。” 
