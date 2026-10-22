# Cursor 执行交接书 — BELA/ELVA 输入自适应证据预算与交互生命周期
**版本：2026-10-22 / Experimental Prototype / NOT SERVER-INTEGRATED**

## 先读这句话

本交接包真正包含经过 CPU 测试的预算核算、视觉覆盖难度探针、无标签阈值政策、动态 Stage2 日程控制、原 Patch ID 契约、POPE 错误诊断、逐层 Token trace 验证和阴性对照生成工具。**它不是一个可对现役 BELA 主干直接 git apply 的完成补丁**。本次上传唯一可核对的模型代码是旧的 E014 `clip_encoder.py`、`vitcop_modeling_llama.py`；后续 BELA 双层 COVER、RDRIFT 当前生产源码及全部数据在附件中缺失。Cursor 必须先审计服务器上的真实代码，不可覆盖现役模块。

## 一、严格不可改变的基线

- LLaVA-1.5-7B checkpoint、CLIP preprocessing、FP16/backend、tokenizer、prompt、scorer、generation 参数全部以当前**可复核的生产配置**为准，不从旧聊天推测。
- Freeze 基线 `dual Layer9+penultimate + COVER -> Stage2 RDRIFT -> post4/post12 (172,90,32)`；检查 W=2048（32 decoder blocks）。
- Offline 阶段任何信号只读；不得一边做 POPE 诊断一边调参数；新分支 OFF-by-default，`BELA_NEXT_ENABLE=False` 时必须与历史模型身份、logits、生成结果回归一致。
- 禁止直接使用附件 E014 的 `e014_choose()` 代替当前 dual COVER。它的 `STAGE1_K=172` 写死，缺当前的 multi-stage RDRIFT；其 E013 `FIXED_COVER` 还依赖未上传 helper。
- 没有原始 checkpoint、benchmark 和 scorer 就只跑本包测试，严禁伪造“全量 GPU PASS”。

## 二、交付代码清单

| 文件 | 真实作用 | 边界 |
|---|---|---|
| `bela_next/budget.py` | post-block W 精确核算、3档 Profile、剪枝身份不复活约束 | 不计真实算子 FLOPs |
| `bela_next/signals.py` | NumPy 参考：真实 COVER 贪心前缀残余覆盖、同 ID attention turnover | 不是因果信息量 |
| `bela_next/stage1_torch_adapter.py` | Torch 版覆盖信号与实际 COVER 选择器安全调用契约 | 不含真实 COVER 或投影代码；GPU 开销待优化 |
| `bela_next/policy.py` | 无标签 DEV 四分位门限、档位分配、同 W Stage2 cut12/cut20 | 阈值未 GPU 验证 |
| `bela_next/stage2_adapter.py` | 复用现役 RDRIFT 得分的选点/原 ID 单调剪枝契约 | 不做真实 Decoder KV gather |
| `scripts/plan_offline.py` | 阈值冻结 / heldout 计划生成 | 不得使用带答案的 TEST 拟合 |
| `scripts/make_matched_controls.py` | 随机打乱图像预算标签，严格保留 EASY/NORMAL/HARD 数量 | 须用唯一 IMAGE ID 做输入 |
| `scripts/analyze_pope.py` | YES/NO、TP/FP/TN/FN、分子集、配对翻转 | 诊断版，最终以官方 scorer 为准 |
| `scripts/audit_trace.py` | 实际每 block Token 数和原 patch ID 事件核验 | 不能替代 KV/position 检查 |
| `scripts/server_preflight.sh` | 只读服务器 Git/import/SHA 审计 | 不改服务器文件 |
| `configs/experiment_matrix.csv` | 预注册候选实验臂 | 不代表任何臂已运行 |
| `configs/run_manifest_template.json` | 强制冻结实验元数据 | 缺字段视为未证实结果 |

## 三、P0 — 仅检查现役服务器源码（不允许修改）

在真正的生产模型仓库运行：

```bash
bash /path/to/THIS_HANDOFF/scripts/server_preflight.sh
git status --short
git rev-parse HEAD
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

生成 `BASELINE_LOCK.md`，逐条记录：
1. **实际导入的** `clip_encoder.py` / `vitcop_modeling_llama.py` 等路径和 SHA256。
2. 当前 COVER 的计算依据：第9层/倒数第2层特征如何归一化、是否真正贪心、能否输出**选择顺序**；哪些结果已经空间排序。
3. 当前 RDRIFT 得分在哪里计算，是否保留同原始 patch ID 的历史值，是否存在 `now-old` 不同分母/Question Mask 的偏差。
4. Stage1 视觉 Token 排序、LLM `image_start`, prompt 文本 span；two-stage 裁剪发生在 block 后还是前。
5. 实际 W=2048 是否由 Decoder 真实每层视觉 token 数逐层观测到；分支名不能当作真实证据。
6. Pre-fill/post-prefill/decode 的 `attention_mask`、`position_ids`、RoPE、`DynamicCache`、KV 物理压缩及 beam/batch 行为。
7. 原始预测、官方 scorer、执行脚本、GPU、计时边界完整路径。

`E014_AEE/llava/model/multimodal_encoder/clip_encoder.py` 可作为历史参考：候选替换代码在其 `visual_tokens_spatial_similarity` 调用附近，原始 IDs 与 `last_dynamic_info` 的采集也在那里；`vitcop_modeling_llama.py` 的 E006 QED 控制和 `post-layer` 物理删除位置可作为参考。**这不是现役 BELA 的已确认行号**。

## 四、P1 — 先跑当前交付包的离线验证

> 从本包根目录执行；工作目录及 import 路径必须正确。

```bash
python -m compileall -q bela_next scripts tests
python -m unittest discover -s tests -v
python -m scripts.plan_offline calibrate \
  --input examples/dev_signals.jsonl \
  --policy examples/frozen_policy.json \
  --model-revision SYNTHETIC_ONLY
python -m scripts.plan_offline plan \
  --input examples/heldout_signals.jsonl \
  --policy examples/frozen_policy.json \
  --output examples/heldout_plan_fixed.jsonl --mode fixed
python -m scripts.plan_offline plan \
  --input examples/heldout_signals.jsonl \
  --policy examples/frozen_policy.json \
  --output examples/heldout_plan_dynamic.jsonl --mode adaptive
python -m scripts.make_matched_controls \
  --input examples/heldout_signals.jsonl \
  --policy examples/frozen_policy.json \
  --output examples/heldout_plan_shuffled.jsonl
python -m scripts.analyze_pope \
  --dense examples/pope_dense.jsonl --bela examples/pope_bela.jsonl
python -m examples.make_trace_example
python -m scripts.audit_trace --input examples/sample_trace.jsonl
```

`examples` 是**虚构的纯工程输入**，只验证算法边界。绝对不要把其结果作为论文指标。

验收：所有测试通过；每个 Profile 的 post-block 计数可逐层反算；cut12 与 cut20 同档**精确相同 W**；无重复/复活 patch ID；同一图像固定 Stage1 K0，Token 删减只发生在预注册事件边界。

## 五、P2 — 先添加只读 Trace，不改算法

### 视觉入口新增的必要 trace

在真实 COVER 选择器内部记录（GPU/accuracy 可开启；TIMING 关闭）：
- `request_id/image_id/CLIP_image_hash/patch_count=576`
- `layer9_patch_features` 和 `penultimate_patch_features` 的**真实**选点依据；浮点特征可只用于抽样 trace 文件，避免巨量 JSON。
- `cover_addition_order_original_patch_ids`：真正的每步加入顺序；如果算法不是贪心，不得虚构顺序。
- `selected_patch_ids_original_order`：最终送 LLM 的集合（空间排序）；
- `feature_coverage_residual_at_k16_32_64_96` 与计算 wall/CUDA time（一定是相同特征定义/权重）。

如果当前实现不是贪心 COVER，先在**离线副本**运行原算法分别输出 K16/K32/K64/K96 的原始集合，评估每个集合对**全576原 patch**的覆盖残差；不要用空间排序去近似贪心顺序。不允许无对照情况下把 COVER 整体换为新 Facility Location。

### Decoder 两个边界新增的必要 trace

- 每个 block **运行之前**的真实存活视觉数量（精确 W 审计）。
- `post_block`, `before_orig_patch_ids`, `after_orig_patch_ids`, `RDRIFT_orig_patch_scores`。
- 有效 question 文本索引集合及其长度、attention backend；同 ID 的历史分数应明确缺失或真实可用。
- `last_dynamic_info`, 模型日志中的 patch positions、`attention_mask/position_ids` 形状、decode1/2 cache 的 length/positions。
- 同一 image 可能有多个 question；Stage1 只按 IMAGE 信号，Stage2 可以按问句 signal。样本拆分必须按 image_id 分组，避免图像泄漏。

将模型 trace 转为 `scripts/audit_trace.py` 指定的 JSONL，不要用 plan 中的理论长度冒充实际长度。GPU 不通过 P0、P2：停机查清，不进入全量精度。

## 六、P3 — 可选接口接入（必须新分支、默认关闭）

建议新增新的配置项（名称为建议，别与当前 FLAGS 冲突）：

```text
BELA_NEXT_ENABLE=False
BELA_NEXT_STAGE1_MODE=FROZEN|COVER_RESIDUAL_BUDGET
BELA_NEXT_STAGE2_MODE=FROZEN_RDRIFT|BINARY_LIFETIME
BELA_NEXT_POLICY_PATH=/ABSOLUTE/PATH/TO/FROZEN_POLICY.json
BELA_NEXT_AUDIT=False
```

### Stage1 实现要求
1. 将已有 Layer9+Penultimate 的**真实 COVER** 调用改成可接受 `budget=K0`，不改其特征计算、选择准则及项目基线。
2. 在固定选择之外提供 `greedy_order` 或不同 K 的实际保留集合，用 `torch_coverage_difficulty`/其经 profiler 优化的版本获得单个 `difficulty`。
3. 加载冻结 DEV `QuantilePolicy`；得到 EASY96 / NORMAL160 / HARD172 的目标 K0。
4. 只在投影/拼接 LLM 之前收集对应的真实 PATCH 向量，保持原始 Patch ID 及空间顺序。
5. **不可**直接使用旧 E014 `STAGE1_K=172` helper 混成动态预算；也不能简单更改常量，却遗漏后续 tensor shape、metadata 或位置编码同步修改。
6. 不能把上一次请求的 policy 状态缓存到下一张图像。

### Stage2 实现要求
1. **第一版**用现役原封不动的 RDRIFT 按给定 Kmid/Kfinal 进行两次物理剪枝，cut1固定4、cut2固定12。请先让动态 K0 和静态 Stage2 走通。
2. 二进制生命周期实验只改变 cut2=12 或20，并调整匹配的 Kmid；**每张图像的 W 与原计划完全相同**。
3. Controller 只能使用 cut1 之前已经实际计算、且同 ID/有效问题 token 范围一致的注意力信号。若当前代码没有两次真正可对齐的历史 attention，标记缺失并退回固定路径；不允许为了动态而额外输出整层矩阵。
4. 所有剪枝采用 post-block 语义；Patch ID 单调嵌套，不允许恢复已删 Token；文本 Token 位置、cache、mask、RoPE 保持当前可用版本的正确流程。
5. 在任何 full benchmark 前先做 prefill + decode1 + decode2 + batched/多问题 smoke；长度完全匹配、logits 严格可解释地一致，并验证选择器仅在预设事件被调用。

### 预算路径
```
EASY:   96 × 4 +  64 × 8 + 32 × 20 = 1536  (cut2=12)
        96 × 4 +  48 ×16 + 32 × 12 = 1536  (cut2=20)

NORMAL:160 × 4 +  96 × 8 + 32 × 20 = 2048
       160 × 4 +  64 ×16 + 32 × 12 = 2048

HARD:  172 × 4 + 154 × 8 + 32 × 20 = 2560
       172 × 4 +  93 ×16 + 32 × 12 = 2560
```

注意 `W=64×32=2048` 是视觉 Token-layer 工作量概念；实际测量还须计算 CLIP、投影、选点、LLM FFN/Attention、cache；动态方法样本之间预算不同，要报告整个 heldout 的**实际平均 W**。

## 七、P4 — 实验执行与评价报告

严格按 `configs/experiment_matrix.csv` 中 P0/G1/G2/G3 分批执行。

**最小可信顺序**：
1. 固定 BELA Anchor vs 同协议现役 Dense 与官方 STAR-Pro T64，GQA+POPE 完整评分。
2. 当前 Stage1 而不改 RDRIFT 评分的 160/128 入口预算重新分配对照（同 W2048）。
3. Stage1 COVER residual 预算图像级打乱对照，完成同实际平均 W 的完整 GQA+POPE。
4. 若 G2 过关，新增 Stage2 cut12/cut20 的在线对照，与匹配 cut 比例的打乱对照比较。
5. 只选前两名配置做完整九任务、W1024 压缩档、必要跨模型和 GPU TTFT/Fixed32。
6. 所有结果必须保留逐样本预测、实际 schedule、每样本真实 W、真实 selector 耗时、original Patch IDs、硬件和 scorer 哈希。不存在数据即标 `NOT_MEASURED`。

**POPE 的决定性诊断**：
- 原始 random/popular/adversarial 三类 TP/FP/TN/FN、Yes-rate、invalid/parsing 比例。
- 主分数以项目官方 scorer 的 POPE F1 为准；`scripts/analyze_pope.py` 只做严格 Yes/No 诊断，不替代官方评测。
- 比较 Dense 正确/BELA 错误、RDRIFT 与其他 Stage2 的同 ID 保留变化。负样本无“目标不存在的对象框”，不能用单个 bbox 是否保留解释全部 POPE。
- 重点检查预算适应是否减少错误**而不是只改变 Yes 答案偏置**。

**效率验收**：
- 同一 GPU、同一缓存和预热条件交替执行，计真实 model-level TTFT 和 Fixed32，计 Vision/COVER/Selector/Decoder 分段耗时；
- OFF Identity（禁用新开关）与历史基线的 Layer ID、final Patch ID、logits、生成字符串一致；
- AUDIT=True 只用于 debug，测速关掉；
- PyTorch 中 `.item()`, `.cpu()`, Python N×K 循环和重新计算 Attention 都可能改变 TTFT，不能称为“无额外成本”。

## 八、Gate 失败时的动作

| 失败模式 | 必须采取的动作 |
|---|---|
| 原 COVER 没有真实 addition order | 不得拿 sorted IDs 冒充；先做离线多 K 覆盖诊断；Stage1 动态暂停 |
| old baseline 不可复现 | 冻结当前环境，不开启新实验；查 prompt/scorer/cache/源码 |
| 无早期同 ID attention | Stage2 adaptive 不启动，保留 frozen RDRIFT |
| 平均 W 超预算 | 正式报告实际 W；调 DEV 政策并在新 heldout 验证；不能直接改 TEST 阈值 |
| 随机预算不输 Image-aware | 放弃“复杂度可预测增量效用”主贡献 |
| 控制器 TTFT 吃掉精度收益 | 优先选择更简单的固定配置，不能只报告 token-layer |
| POPE F1 上升但 Yes-rate 改变较大 | 先排查偏置与真实漏检；暂不宣称视觉证据保留改善 |
| GPU decode KV/mask 测试失败 | 立即回滚新分支，不继续跑全量 |

## 九、最后请 Cursor 交付而不是只回答“已完成”

输出以下**真实运行产生**的文件：
1. `BASELINE_LOCK.md`：源码路径/SHA、旧新开关、数据与评分器冻结说明。
2. `SOURCE_DIFF.patch`：生产新分支的真实源代码差异，不能把本交付包称作完整生产 patch。
3. `SMOKE_AUDIT.jsonl`：包含原始 patch ID、逐层真实 W、S/D、mask/KV/position 状态的 GPU trace。
4. `RESULTS.csv`：每臂每任务原始分数、R4/R9、逐样本 W、TTFT、固定长度生成耗时、COVER overhead、Selector overhead。
5. `POPE_DIAG.json`：三类 POPE 的 confusion/Yes-rate、与 Dense 配对错误变化。
6. `STOP_GO_DECISION.md`：对复杂度 Stage1、二值 Stage2 各自的“继续/放弃”，附实测证据。

**不允许汇报“全部成功”，除非上述材料真实存在。**

## 十、论文相关工作和官方期限

见 `01_研究负责人独立判断.md` 的原始论文链接。CVPR 2027 注册 2026-11-10 AoE、全文 11-16 AoE、补充材料 11-23 AoE；登记和论文写作必须与实验并行，而不是等动态预算全部跑完才开始。
