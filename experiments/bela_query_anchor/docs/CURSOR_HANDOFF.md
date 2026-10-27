# BELA / CVPR 2027 — Cursor 最终 Stage1 查询锚定冲刺 Handoff
**版本：2026-10-27 · 交付状态：离线代码已测试 / 尚未接入生产 LLaVA / 尚无 GPU ACC 结果**

> **实验范围冻结**：这不是 CDPruner/QCPruner 复刻，也不是第二轮 QCP 调参。保留 BELA 原生 Dual COVER + RDRIFT，唯一新增候选是在 **Stage1 初始 K64** 上，对完整原始 Patch 集合进行最多8次问题条件的**同容量替换**。任何未经验证的准确率提升都不能写进论文。

## 0. 为什么转向这一轮（决策延续性）

历史报告：L2 W1024：GQA 55.57、POPE F1 82.21、MMBench 61.34、TextVQA-OCR 53.41；E3 W2048：GQA 57.76、POPE 85.46、MME 1779.1、MMBench 63.32。与本机 CD K32 GQA57.00/POPE87.67 比较，存在真实竞争压力，但计算路径及协议必须统一。

本机失败证据：B1 COVER32 flat GQA53.09/POPE74.53；B2 ENTRY_ORDER 52.16/79.09；B3 S16 55.15/80.96；有界准入 b=2 55.64/82.15；C1 QCP 55.69/82.28。QCP 在99.6%样本有交换，但 POPE 正例召回依然约73.7%。**没有证据继续搜索 QCP 超参。** C2 random swap 尚在跑，由已有队列收尾，不重复提交。

POPE 是对象存在性 Yes/No 评测，但 **FN 多既可能来自目标证据缺失，也可能来自 Yes/No 决策偏置与解析协议**。原始 L2 的 FN1188/FP245、Yes-rate约39.5%；CD32 FN621/FP470、Yes-rate约48.3%。不能宣称 CD32 的提升全部是更好的视觉识别。

H1 Stage1 漏证据、H2 Stage2 误删、H3 跨阶段失配、H4 答案偏置目前均属于竞争解释。Stage1 是**下一次唯一算法改动优先级**，不是已证实的主要因果瓶颈。

## 1. 本次应安装的文件

```text
experiments/bela_query_anchor/query_anchor.py    # 完整、可测试的固定K选择算子
experiments/bela_query_anchor/llava_adapter.py    # 请求局部的LLaVA投影和question embedding适配
experiments/bela_query_anchor/pope_audit.py       # 逐样本混淆/错误转换审计
experiments/bela_query_anchor/make_shuffle_map.py # 冻结错配问题ID负控
tests/test_query_anchor.py                        # CPU unit + constructed stress
configs/S1_QANCHOR_L2.json                        # 本轮唯一算法arm
configs/S1_QANCHOR_CONTROLS.json                  # label-free负控
docs/CURSOR_HANDOFF.md                            # 本文件
```

本次上传的历史 ZIP `BELA_FINAL_HANDOFF_20261026.zip` 是执行交接/CPU参考包，**不是现役服务器完整源码**。用户提供的当前源码摘要只确认了 `experiments/elva_r3/llava_stage1.py` 的缓存变量、`e018_stage2_support.py` 相关片段；Cursor必须在现役仓库找到真实调用点，不能假设这个包已经自动接好了完整LLaVA。

## 2. 必须使用的数据通路（先做这个，再上GPU）

### 2.1 Stage1 仍然先算原生 Dual COVER K64

在**原生 Stage1 产生 K64 原始Patch ID之后、物理 gather和LLM prefill之前**插入一次调用。输出 `keep_ids` 必须仍是64个原始ID，不改变令牌的表征来源（送入LLM的仍是原有指定CLIP层的真实patch features）。不要把Layer9中间特征直接当作LLM实际图像输入。

### 2.2 问题信号来源：优先复用当前LLaVA已有模块

```python
from experiments.bela_query_anchor.llava_adapter import apply_llava_query_anchor
from experiments.bela_query_anchor.query_anchor import AnchorConfig

# PSEUDOCODE ONLY: 变量名根据现役仓库真实接口绑定；不是可以盲贴的diff。
result = apply_llava_query_anchor(
    mode="s1_anchor",  # 对照 "off" 或 "matched_random"
    native_keep_ids=native_keep,          # [64] ORIGINAL patch IDs
    dual_metric=selected["metric"],       # [576, D] Stage1原生Dual descriptor
    full_vision_features_for_projector=full_patch_hidden, # [576, Din],真实mm_projector输入
    question_only_ids=question_token_ids, # [T] ONLY question tokens
    mm_projector=model.mm_projector,
    llm_input_embeddings=model.get_input_embeddings(),
    projected_full_vision=already_projected_full_grid_or_None,
    config=AnchorConfig(max_swaps=8),
    diagnostics=trace_mode,
)
keep = result.keep_ids
# 后续原生物理视觉特征gather、Stage2统计、entry-K缓存均用更新后keep。
```

只允许以下两个已合法获得的模态信号：ViT已经计算的完整图像Patch状态；当前question-only token经模型原有`embed_tokens`获得的文本输入embedding。图像侧进入LLM的投影由**同一个已有mm_projector**取得。没有新ViT forward、没有完整额外LLM forward、没有标签、没有未来层。若项目已有全576投影，重用缓存避免重复计算；否则全576投影**必须独立计时**，不可谎称零开销。

特别注意：CLIP图像未投影Layer9特征与LLM文本embedding不能直接点积。即便在mm_projector映射后，cosine语义相关性也**未被保证**；论文CDPruner在部分架构上使用相似跨模态proxy并不证明本机有效。采用 `matched_random` 与打乱问题输入的负控决定这个proxy是否提供真实信息。若未能验证问题信号优于乱配问句，不保留算法。

### 2.3 Query token 的严格协议

`question_token_ids` 只能来自**当前样本真正的问题文字**。排除系统提示、聊天模板、答案、GT、assistant prefix、图像占位符ID（如-200）、padding、其他sample的token。不能简单拿整个 `input_ids` 做平均。Cursor需要记录实测token数量和首尾安全解码摘要，检查POPE的 `Is there ... ?` 中目标对象词仍在，不得用GT对象标签提取问题词。

如果当前Stage1在image-only函数里根本拿不到question，应从已有`prepare_inputs_labels_for_multimodal`的用户问题解析位置显式向下传参。禁止使用跨请求可变全局 `LAST_QUESTION`；先限定 batch=1，单图。批处理应使用per-request上下文，不允许读取上一请求的缓存。

### 2.4 原始Patch身份及Stage2合同

- 所有ID是原图网格的 `[0,576)` ID，且无重复；真实高分辨率多crop不可用固定576默默处理。
- 额外记录每个新换入的Patch在第12层和第20层剪点后是否存活；若Stage2立刻删光新增候选，不能把入口语义提升当作完整协同成功。
- 变更 `native_keep` 后，缓存的 `LAST_ENTRY_KEEP` **必须替换成新的keep**，`LAST_ENTRY_METRIC`仍是完整原生视觉descriptor，用于历史方法审计。它们的请求作用域需查实；多worker并发全局缓存不能认为安全。
- 物理序列顺序仍按原始ID空间升序，位置编码、视觉占位符及文本起止位置语义不变。
- 原生Stage2 RDRIFT调用、真实剪点 `64×12→17×8→10×12` 完全保持不变，`W=1024`；不要给新Stage1偷偷再加后层保留保护。
- 仅改变候选身份，不改变Stage2 scoring公式/未来attention/KV cache。
- `BELA_QA_ENTRY_MODE=off` 必须逐样本与旧L2输出ID、生成答案、logits及时间路径等价（计时误差除外）。

### 2.5 设计的意义及约束

算子先接受原生K64，在剩余Patch中检索问题相似度较高、且不被现有64个代表覆盖的候选。被替换者是查询分数低、视觉重复较高的候选。允许替换的条件包括：(i) 可靠非平坦问题信号；(ii) 查询得分明确更高；(iii)视觉novelty足够；(iv)全视觉覆盖不能明显下降。一次最多8次交换；否则保持原集。

这是**约束型问题候选调整**，不是 POPE 的“存在概率检测器”。负例不存在对应目标Patch，不能把最大相似度当作对象存在性真值，不能因为POPE是二分类就用标签或阈值在测试集直接优化答案输出。

本实现 `min_raw_std`、`min_query_gain`、`min_novelty` 等阈值仅为首轮**预注册的固定默认值**；不得依据测试集成绩扫描。所有阈值须在正式实验记录中报告。算法很接近已有QCPruner/CDPruner的 Query + Coverage/Diversity，**创新性不能从这个公式单独认领**。投稿希望看到BELA Dual入口与RDRIFT在严格W和真实TTFT下的新增效率/质量证据。

## 3. GPU开始前验收（P0）

1. 本包 `python -m unittest discover -s tests -v` 全部通过；原生base/off等价、单图batch=1、重复ID、负图像占位符、空问题、NaN、稀疏候选均已覆盖。
2. 现役路径smoke：每样本 `native K=64 → final K=64 → first cut17 → second cut10`；记录至少3个图像、不同问题、同图异问题，验证候选能够随question变化；如果100%样本完全不变先检查向量对齐。
3. 原生底层图像 `full_patch_hidden` 与Dual `selected["metric"]` 是两套用途不同的特征。核查二者原始Patch索引一致，`mm_projector`接收与原LLaVA同类型的真实输入。
4. `S1_QUERY_SHUFFLE` 用另一问题的question embedding替换，不改图；可用`python -m experiments.bela_query_anchor.make_shuffle_map --manifest /path/to/samples.jsonl --output /path/to/shuffle.json --seed 17`冻结ID映射，源问题不得来自同图；必须检测问题与图像不一致时结果是否失去优势。不要用实际答案指导该负控。
5. 原生L2、S1、随机/打乱控制必须同一模型、同一图像与问题ID、同一评测配置；当前独立部署路径未统一前不得把跨方法系统耗时称作因果消融。
6. 记录当前算法选择器的耗时、额外全576投影、`torch.cuda`同步次数及峰值显存；CPU smoke不等于真实加速。

## 4. 非GPU先决：POPE数据口径审计（最高优先级）

2026年10月有公开论文指出，某些LLaVA/LLaMA评测脚本读取 `yes/no` token ID 时会遗漏空格前缀token，单独造成数个百分点的F1差异。见 https://arxiv.org/abs/2610.25964 与官方 POPE https://github.com/RUCAIBox/POPE 。

**不是说你们一定存在此问题。** 只需核对已保存的原生L2、CD32、C1及待完成C2：
- 实际 `prompt`、tokenizer、原始首token或原始生成字符串；
- 原始Yes/No解析器、invalid约定，是否比较了裸`yes/no`或含空格变体；
- macro-F1/pooled-F1；三类negative split；问题是否一正一负。
- 输出 `TP/TN/FP/FN/yes_rate/precision/recall/F1`，保存sample级配对变化。
- 统一不改原有官方打分口径用于主表；若发现历史协议不一致，先统一后**重新审计所有本机主结果**，不能只纠正BELA。
- `pope_audit.py`只统计锁定评测器的已解析标签，不会擅自把unknown归到No，不是新的官方评测器。
- 可额外在验证集离线研究Yes/No logit threshold，以验证H4；这不是算法主结果，不能用测试答案调校。

POPE存在过多FN并不等于一定有目标patch被Stage1丢失；高质量Stage1应在提升正例Recall的同时报告对应FP及GQA的变化。尤其是CD K32相对L2，少567 FN、但多225 FP，说明yes倾向明显不同。

## 5. 唯一正式算法实验 + 必要负控

| arm | Stage1 | Stage2 | 预算 | 作用 |
|---|---|---|---|---|
| L2_NATIVE | 原生Dual COVER K64 | 原生RDRIFT | W1024 | 已知强制锚 |
| S1_QANCHOR_L2 | Dual COVER K64 + 最多8 Query锚定替换 | 原生RDRIFT | W1024 | 本轮唯一新算法 |
| S1_QUERY_SHUFFLE | 保持算法、仅用错配问题embedding | 原生RDRIFT | W1024 | 是否真使用问题相关证据 |
| S1_MATCHED_RANDOM | 与S1相同具体删除ID和次数，新增ID随机 | 原生RDRIFT | W1024 | 排除交换量/入口扰动本身 |

首轮 4任务：GQA、POPE、MMBench、TextVQA-OCR；可以先GQA+POPE门禁，再跑保护任务。优先从已有pilot manifest拿完全相同图像/问题ID。对随机负控首先seed17；晋级后用29/43复测随机稳定性。打乱问题严格来自独立随机同批图像/问题且不能让同图不同问串到同一图；冻结映射seed。

**准确率筛选标准：**
- GQA至少 **56.37** 或 POPE至少 **84.21**（为本轮既定研发门槛，不是预测值）；另一项不能明显下降。
- MMBench≥60.84，TextVQA-OCR≥52.91。
- 必须强于shuffle与matched random；如问题打乱仍得到同等收益，不能宣称query-aware机制。
- POPE报FN、FP、yes-rate；只抬Yes但FP同步失控，不认作恢复视觉证据。
- 配对bootstrap按image_id聚类；报告不确定性，不以单次0.1分当作稳定增益。
- 真实TTFT、Fixed32、选择器耗时、peak VRAM；建议优先要求额外TTFT在5%以内，若超过需要展示实际Pareto价值。硬件/attention backend/精度/protocol严格一致。

**不通过**：停止S1，不加CLIP文本塔补救、不扫8/16/32交换数、不叠加QCP、不再试复杂度动态K。冻结现有E3 W2048与L2，并把负结果如实用于机制边界。**通过**：同一 S1 程序固定超参跑W2048 E3（入口K172、原生E3日程），再六任务完整确认，不能以L2收益推出E3有效。

## 6. 具体运行与日志交付

四张卡 0/2/4/7 已空闲，但不要把CPU无误等同production正确。先每卡独立smoke、确认`CUDA_VISIBLE_DEVICES`只暴露目标卡，batch=1。四任务并行，不覆盖旧日志，不重启已经完成的C1/C2。

新目录例如：
`logs/e014_aee/full/bela_stage1_queryanchor_20261027/`

每个arm至少导出：
- `run_meta.json`: git commit、model/checkpoint hash、tokenizer hash、prompt/scorer hash、precision/backend、环境、seed、arm、W和实际逐层K、source_features及额外projector开销；
- `predictions.jsonl`: sample_id、image_id、question、原始decoded输出、pred_label、label、official_score、valid、subsplit；
- `entry_trace.jsonl`: native64IDs、final64IDs、swap数量、被删/新增Patch、query proxy分布、coverage变化、同图跨问题稳定性；
- `metrics.json`: 官方任务值及POPE分组、计时分布和VRAM；
- `error_analysis.md`: QCP失败原因、POPE FNs恢复与FP代价、query-shuffle、paired GQA；
- `DECISION.md`: 是否晋级，若失败停止理由。

**离线脚本**：
```bash
python -m unittest discover -s tests -v
python -m experiments.bela_query_anchor.pope_audit \
  --baseline /path/to/L2/predictions.jsonl \
  --candidate /path/to/S1/predictions.jsonl \
  --output /path/to/paired_pope.json
```

## 7. 投稿安排与方法新颖性底线

- 10月27—28日：POPE协议审计 + 本轮代码接入/验证；并行写完Introduction背景和Related Work。
- 10月29日：GQA/POPE pilot 结果；必须作出保留或否决。
- 10月30—31日：通过才跑四任务保护与六任务/full，必要时W2048对照。
- **11月1日算法冻结**。11月2—6日完整计时、消融和论文主表；之后优先提交论文。
- CVPR2027 11月10日AoE注册与摘要截止；11月16日AoE正文截止（见官方 https://cvpr.thecvf.com/Conferences/2027/CallForPapers ）。

**不能在论文中声称**：这个QAnchor本身是“首次问题相关覆盖”“首次动态视觉Token selection”“已恢复所有POPE缺失对象”“从代码测试证明性能提升”。QCPruner、CDPruner、STAR-Pro、AgilePruner、HAP已有相关研究；本方案若成立，贡献必须来自 **BELA多深度入口与原生渐进交互的受控协同增益、负例/正例边界、严格预算及真实效率**。

## 8. 给Cursor的终端执行指令

> 请先完整阅读此文与现役生产路径；保留原方法开关，**只**把`query_anchor.py`接入Dual COVER的K64结果之后、image gather之前；显式传入当前question-only ID，复用已经存在的mm_projector和embedding层。离线15项unit通过不代表线上正确。优先统一POPE官方Readout并输出旧预测混淆矩阵，随后按四个arm执行严格同样的GQA/POPE先导、负控和计时。不能用答案指导Patch选择，不能更改RDRIFT与1024/2048预算。29日出决定，无正证据即撤回。请提交上述trace、配置、git diff、测试与DECISION.md。
