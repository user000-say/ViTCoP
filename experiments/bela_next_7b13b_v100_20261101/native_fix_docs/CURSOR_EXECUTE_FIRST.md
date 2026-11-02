# BELA-NeXT-7B 真正修复：Cursor 直接执行指令（2026-11-02）

## 立刻停止错误做法

当前 A0/A1 确实发生了剪枝，但你上报的 `simple_crop_concat`、AnyRes 强制 `[[672,672]]`、丢失 QA B8、剪后 RoPE 重排、修改生成与 SciQA exact_match，都是必须分别核验的工程差异。**禁止继续把当前结果作为 BELA 最佳版的性能结论。** 已运行任务可保留原始预测用于比较，但应标记为 `LEGACY_SIMPLE_PACK_B0`，不可偷偷覆盖成绩。

当前提供的附件**没有实际 V100 工作树、没有 `apply_llava_next.py`/`dual_cover.py`、也没有原 1.5-7B `qa_core` 源文件**。本包因此提供真实可执行参考实现+可自动运行的合成测试，**并没有谎称已经修改了你的远程仓库**。你需要在现有 repo 实现两个 original-algorithm wrapper 后再跑真实 checkpoint smoke。

## 必须保留的论文方法

- Stage1: **历史有效版 Dual-depth Native COVER** (mid ViT layer 9 + penultimate, `slots=32`)，不能用随意余弦 TopK 替代。
- 有界 QA-COVER: `max_swaps=8`，`donor=16`，`drop=12`，`seed=17`；必须复用现存 `qa_core`/coverage-delta donor-swap 算法，且真实 question embedding/ID 来源需经核验。每张图 **全局 B8**，非每个 Crop B8。
- Stage2: Q_ATTN_ALL，第12/20层后物理剪枝；对 7B 上 94.4% 五个 A0~A4 使用严格 Patch Token-layer `W=5120`，不替换成 STAR-Pro QR。
- 原生 LLaVA-NeXT `pack_image_features`: `spatial_unpad` + learned `image_newline`；没有任何 `simple_crop_concat`。
- 不强制 `image_grid_pinpoints=[[672,672]]`；processor 和 checkpoint 配置要和 2026-10-28 Dense 一致。无论有多少 Crop/实际 unpad Patch，必须打印实际原生 Patch 数和 newline 数。**W=5120 是视觉 Patch Token-layer，不等于全序列 KV / 端到端 FLOPs，也不能对任何图都说初始视觉 Token 恰好 2880。**

## 代码接法：当前 repo 精确接口

1. 查明 `transformers.__version__`、当前 `modeling_llava_next.py`、真实 `model.get_image_features` / `model.pack_image_features`、`model.language_model.model.layers`。如果其 API 不符合 `bela_next_fix/runner.py` / `progressive_hooks.py` 的 **HF 4.48-style API**，**先适配当前版本，绝不捕获异常回退到 Dense**。
2. 把本包 `bela_next_fix` 目录复制到你的项目中（或通过 `PYTHONPATH` 引用）。导入 `BelaConfig`, `ARMS`, `greedy_native_bela`，替换 BELA **专用**评测分支的调用入口，保持 Dense 原入口原封不动。当前参考运行器仅支持 **单卡完整权重、batch=1、FP16、eager attention、greedy generate、DynamicCache**，其他路径请先扩展+测试。
3. **实现原算法包装器** `your_project_bela_compat.py`：

```python
# STRICT: below are interface templates, NOT completed legacy functions.
# Use the exact ORIGINAL 1.5-7B code; do not add a new approximate COVER.

def original_dual(*, mid, pen, keep, slots, seed, context):
    # call the historical native Dual-depth COVER module
    # output: 1-D long tensor of length keep, in native *packed patch* space
    raise NotImplementedError('Wire the original Dual COVER here')


def original_qa(*, mid, pen, question, question_ids,
                initial_keep, max_swaps, donor, drop, seed, context):
    # call the historical QA-COVER with true coverage-delta donor-swap
    # return EXACT same count, <= max_swaps changed patch identities
    # ensure text features are in the same representation as legacy QA
    raise NotImplementedError('Wire the original qa_core + bounded swaps here')
```

**关键**：这不是让你把历史单图 576 Patch 的 ID 直接复用到 NeXT 原始五裁剪顺序！输入 `mid`/`pen` 已经通过 `model.pack_image_features` 的原生坐标顺序重排并去掉 `image_newline`。选择器操作的是当前**unpad 后、已投影或原生 ViT空间的一维 Patch 映射**；如果历史算法只支持 24×24 网格，要在**多裁剪分组坐标**中保持几何正确（每组做 coverage 再合并），而不是错误地把所有 Patch 当作一张任意长条图。必须提供相应分组策略和原 Patch ID 映射。

4. QA 的 question representation 不能把 LLM 4096-D 与 ViT 1024-D 直接点积。`run_original_bela` 传入 `question` 为真实 LLM text embeddings，`question_ids` 为文本 ID，`context['model']` 包含完整模型供 legacy wrapper 使用；**wrapper 应采用历史 QA 路径曾使用的文本/视觉对齐方式**。如果不能确定真实一致的跨模态相似度空间，报错而不是装作 QA 已接好。正式跑分应把提问的实际 Token span 传入 `question_token_positions`，不应长期使用默认所有非图像 Token。

5. 最终原生生成目标：除压缩选集与物理 KV 缩短以外，prompt、chat template、EOS、`max_new_tokens`、beam/greedy、stop 标准、答案 parser 均与 2026-10-28 Dense 评测对齐。参考 runner 目前只实现 Greedy + EOS（**不实现 stop strings、beam、采样**）；若 Dense 原始 protocol 用其他规则，先在本地增补逐 Token 等价，再运行正式六项。

## V100 执行顺序（绝对不碰 GPU 0/1/2/3）

```bash
# 在 V100 机器上、进入当前仓库，选 4 号 GPU 作为进程 cuda:0。
CUDA_VISIBLE_DEVICES=4 python -m unittest discover -s tests -v

# 真实图片, 本机实际 checkpoint/cache, 实现 ORIGINAL wrappers 后：
CUDA_VISIBLE_DEVICES=4 python smoke_real_next.py \
  --checkpoint llava-hf/llava-v1.6-vicuna-7b-hf \
  --image /path/to/one_REAL_dev_image.jpg \
  --dual your_project_bela_compat:original_dual \
  --qa your_project_bela_compat:original_qa \
  --max-new 8 --arms A0 A1 A2 A3 A4 \
  --json-out bela_next_smoke_native.json
```

**先检查单卡 V100 是否能容纳 FP16 NeXT-7B + vision tower + greedy eager-attention + DynamicCache。** V100 16GB 版本很可能不够；不能因为有4张空卡就假定每卡可以单独装7B。需要多卡模型并行时当前单卡参考 runtime **不适用**，须按你的进程和模型切分方式修改并重新验证 KV 与 Hook。

## 六道不可绕过的真实性门槛

**G0：同权重、同 processor Dense 等价。** 关闭本包所有 pruning 时，其 token IDs 必须与原 `model.generate` 逐 Token 相同（默认 `max_new_tokens=8` smoke）；对 GQA/POPE/MME/SciQA 样本各抽样，诊断不一致则禁止任何测试。

**G1：原生图片布局。** 记录真实 `image_sizes`、grid shape、pack lens、patch ID map、行分隔数量；比对直接原生 `pack_image_features` 的逐 Token embedding（不允许仅比形状）。不能出现 `simple_crop_concat` 或强行重设 grid。

**G2：历史算法真实调用。** 日志必须记录 `legacy_dual_called=True`, `legacy_qa_called=True`, `swaps<=8`、donor/drop/seed 与对应 Patch ID 差异。可添加一个故意使 QA 返回超 B8 的单元测试，必须触发失败。若没有换入位（`swaps=0`），仍需验证函数确实执行，且在多题上出现问题条件变化。若旧版算法接口无法使用真实文本语义，不得宣称历史最佳已复现。

**G3：真实三阶段可验证。** A0 `320→91→46`；A1 `320→100→40`；A2 `320→82→52`；A3 `288→136→48`；A4 `256→160→64`，必须在解码层输入和物理 KV Cache 上被真实观察。结构 newline 和文本单独计数；对 sequence 与 KV 全层长度断言一致。

**G4：RoPE 及生成。** 视觉选出后保留原始 Dense 序列位置坐标，Layer12/20 后 slice 正确 `position_ids/position_embeddings/attention_mask`，物理 `cache_position` 是压缩缓存位置，decode 时 `position_ids` 继续原 Dense 位置轨迹；用首字母、OCR、多 token 文本检查不会出现错位。禁止 `0..L-1` 直接重置语义 RoPE。`eager` Attention 探针与独立参考数值核对，检查实际 Q_ATTN 而非仅产生不同 hash。

**G5：生成及评分公平。** 不要为了避免 OOM 自行把仅 BELA 的 `max_new_tokens` 改为更小、截断到 newline 或修改 prompt；SciQA 原始字符串与规范化字母准确率**分别报告**，在 Dense 和 BELA 上用完全同一个评分器统计。GQA/POPE/MME/MMBench/TextVQA 同样用已验证协议。任何失败 `FAILED`，不得报告成绩为0或空任务完成；失败样本/任务的重跑必须记录去重。

## 真实实验优先级

- 第一优先：原版最佳 BELA (`Dual+QA B8+Q_ATTN`) 在 **A0** 同预算下，对当前 `simple_crop_concat+B0` A0 做同批 Dev 配对，首先分离工程保真与 QA 的收益。严格来说同时修复多项不能单因素归因；增补保真 B0/保真 B8 两臂会更清楚。
- 第二优先：保真+QA B8 的 A0~A4 在小 GQA/POPE/MME/SciQA/TextVQA Dev 集验证，关注 TextVQA OCR、真实生成字符和 SciQA 原始/规范化两列，冻结最优。
- 第三优先：全量六任务，并记录 TTFT、prefill、decode、peak VRAM，之后再做更复杂的 QA B16 / Crop 分配。不要直接承诺达到外部 STAR-Pro 95–97%：算法不同，Dense 与评测口径可能不同。

### 必须返回给用户的证据

- `transformers` 版本、`git rev-parse HEAD`、源码 diff 和真实测试入口；
- `G0` 原生 Dense 8 token 对齐（若不一致，失败停止）；
- 所有 A0~A4 原始 Patch+newline 统计/QA Patch ID/两次层内物理剪枝/每层 KV 长度；
- 审计日志和 `failed_tasks.jsonl`，SciQA 原始值与字母解析结果；
- 选定 Dev 配对收益，完整六任务成绩和可复现运行命令。

不要以 `CUDA_VISIBLE_DEVICES=4,5,6,7` 启动四个线程并假定 CUDA 设备编号4–7依旧有效；该环境变量内本进程的合法设备索引是0–3。通常每个单 GPU 进程使用 `CUDA_VISIBLE_DEVICES=<物理号>` 并在进程内使用 `cuda:0`。
