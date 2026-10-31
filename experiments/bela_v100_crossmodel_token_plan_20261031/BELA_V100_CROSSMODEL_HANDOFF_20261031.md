# BELA / CVPR 2027 — V100 多模型视觉 Token 预算与最小实验 Handoff
**版本：2026-10-31 | 研究负责人审核版 | 离线预算已经验证，服务器尚未实跑本批新配置**

> 核心任务：完成 LLaVA-1.5-7B 的 77.8%（层平均）10任务 ACC；让已在 V100 准备/部署的 LLaVA-NeXT-7B、NeXT-13B、Qwen2.5-VL-7B（可选3B）、Qwen3-VL-8B、Qwen3.5-9B 各得到可审计的模型专属 Token 预算与跨架构验证。仅匹配预算，不新造模块；保留 V100 已冻结图像 QA B8 + first RDRIFT/later Q_ATTN 与纯 Q_ATTN 两种候选。

## 0. 不得混淆的两个“77.8%”

**层平均视觉 Token 预算：**

```
segments = (l1, l2-l1, L-l2)
W = l1*K0 + (l2-l1)*K1 + (L-l2)*K2
T = W/L
compression = 1 - T/N
```

这里 `L` 为 Decoder 层数，`N` 是**该样本视觉预处理/合并后、送入 LLM 的真实视觉 Patch Token 数**（不包括文本、特殊图像边界/帧分隔符）。`K0/K1/K2` 是三个阶段存活的视觉 Token 数；若与 STAR-Pro 对照，剪枝层为 **1-based 的“第 l 层执行完”**。

- LLaVA-1.5-7B：`N576, L32, T128, W4096` → 精确 77.7778%。入口 `K0=256` **并非保留128入口**。
- LLaVA-NeXT：论文的 5-crop 参考 `N2880`，精确 77.7778% 为 `T640`；`T320` 是88.9%，`T160` 是94.4%。**NeXT 的320不是77.8%。**
- Qwen2.5/Qwen3/Qwen3.5：动态原始 `N` 随图片及 processor 改变。1296或1024只是特定论文参考值，**不得在每张图上无条件写死**。若实际 N 不同，用本包预算脚本按实际 N 调整，记录真实 achieved W 与 ratio。
- Qwen3.5 Stage1-only `K228 of N1024` 约77.73%，**不是与渐进式层平均T228具有相同的计算图**；两者不能混称公平同算法对照。

**匹配预算不等于匹配真 FLOPs 或延迟。** 视觉塔在这些方案中一般仍先完整编码，LLM 隐层宽度不同导致 Token-layer 数本身不能跨模型比较运行时间。

## 1. 已核对 STAR-Pro 公开模型日程；以下优先采用

| 模型 | 参考 N / L | 主要77.8%同预算日程（第 l1/l2 层后剪） | 同 W 备选 | 论文已有更紧档 |
|---|---|---|---|---|
| LLaVA-1.5-7B | 576 / 32 | `256→74→36`, cuts 12/20, W4096/T128 | `256→80→32`, `256→68→40` | T64 `128→37→18` |
| LLaVA-NeXT-7B | 2880 / 32 | `1280→367→182`, cuts 12/20, W20480/T640 | `1280→400→160` | T320 `640→182→92` |
| LLaVA-NeXT-13B | 2880 / 40 | `1280→320→160`, cuts 15/30, W25600/T640 | `1280→360→100` | T320 `640→160→80` |
| Qwen2.5-VL-7B | sample N; 1296 example / 28 | **自拟** `576→198→72`, cuts 10/18, W8064/T288 | `576→188→80` | **自拟** T256 `512→176→64` |
| Qwen2.5-VL-3B (if loaded) | sample N; 1296 example / 36 | **自拟** `576→144→72`, cuts 14/24, W10368/T288 | none | T256 may use 512→128→64 |
| Qwen3-VL-8B | sample N; 1296 example / 36 | **自拟** `576→144→72`, cuts 14/24, W10368/T288 | `576→168→52` | **STAR-Pro primary** T256 `512→128→64`, T128 `256→64→32` |
| Qwen3.5-9B | sample N; 1024 example / 32 hybrid | **Stage1-only** K228 (≈77.73%); **not a deployed progressive schedule** | Stage1 K256 / K128 | `456→120→48` cuts 12/24, T228 is **HYPOTHESIS ONLY** (requires Gated DeltaNet state proof) |

The primary STAR-Pro exact rows are from original Table 5 (https://arxiv.org/html/2609.05916v1). Qwen2.5 stage schedules and Qwen3-VL T288 are **BELA engineering proposals**, not paper-published defaults. STAR-Pro's Qwen3-VL published 256/128 budgets are preferable for direct literature alignment. Qwen3.5 reference final K tiers come from CoRePrune (https://arxiv.org/html/2609.26484), **not a validated BELA mixed-attention pruning schedule**.

### Per-image variable-N adaptation
For Qwen and nonstandard AnyRes NeXT, the utility `BELA_V100_BUDGET_ALLOCATOR_20261031.py --model NAME --n ACTUAL_VISUAL_N`:
1. Sets target T≈round(2N/9) for 77.8%; candidate K0≈2T; chooses integer K1/K2 preserving monotonic pruning and minimizing weighted budget error.
2. Emits realized W/T/ratio, **not optimal ACC or model-level correctness**.
3. Actual model runtime must additionally preserve crop/thumbnails, image-grid IDs, position indices and DeepStack/linear attention state; utility does not enforce those.
4. Do **not** send all samples through an unreported global `max_pixels` that changes input resolution from Dense to pruned; both arms must share the same decoded images and processor settings.
5. For STAR-Pro exact comparisons on NeXT 5-crop N2880, keep the canonical crop layout. Different N per sample means report exact per-sample reduction instead of pretending the whole set ran fixed2880.

## 2. 7B 77.8% main experiment — completion before cross-model queue

### Hard preflight blocker
Earlier V100 history reported `W4096 / K256` blocked by `e013` COVER entry allowlist `{96,128,160,172}`. **Audit current production first**; if still blocked, extend allowable K in Native COVER and QA for true K256 without changing their selection objective or silently clamping to172. Verify selected original Patch IDs, coverage, duplicates, layer survival, KV cache and true `W=4096`.

### Minimal schedule selection
- Maintain **exactly the same W=4096 / decoder cut positions 12 & 20**.
- Run the three schedules `A=256/74/36`, `B=256/80/32`, `C=256/68/40` with first RDRIFT + later Q_ATTN and `QA B8` on **complete GQA+POPE+MME**, not sample subsets. These datasets are selection/evaluation reused, so report exploratory role honestly.
- Pick the balanced winner by per-task accuracy/F1 and compute, not GQA-only. At that single winning schedule compare **two final selectors**:
  `S1 first RDRIFT + Q_ATTN later` versus `P3 Q_ATTN at both events`; common Stage1 B8, same K and cut layers.
- After schedule winner is known, optionally test `B16` **only at the winning schedule**, fixed Stage2 (first RDRIFT); B8/256=3.125% max swaps, B16/256=6.25%, versus historical B8/64=12.5%. Don't declare historical B8 optimal at K256.
- For final selected two methods, run 10 full benchmarks: `SEED-Bench, GQA, ScienceQA-IMG, TextVQA, POPE F1, MME, MMBench-EN, MMBench-CN, AI2D, MMMU`, using exact frozen evaluator and Dense reference. Standard STAR-Pro paper reports a 10-column mean with MME/20; our own R10 must explicitly state formula, preferably additionally give **paired normalized-to-our-Dense mean**, not mix it with historical R6.
- Log TTFT / Fixed32 on same physical V100 and matched fair-256; also log selection overhead and peak GB. FLOPs optional but valuable if present in the same profiler. Never compare raw TTFT across models/hardware as algorithm speedup.

## 3. Cross-model minimal matrix — don't launch hundreds of arms

**For NeXT-7B/13B and Qwen2.5-VL-7B/Qwen3-VL-8B**:
- `DENSE` no pruning, `NATIVE` genuine architecture-faithful Dual COVER plus progressive Stage2, `QA` same Native plus bounded QA correction; all three must share identical N mapping, per-image W, prompt and accuracy scorer.
- Start at each model's main preset from §1; optionally add paper's next stronger tier **only after main primary group passes**.
- Keep model-specific Qwen/NeXT group mappings correct: NeXT global thumbnail/local crop patch IDs; Qwen2.5 `grid_thw`+M-RoPE; Qwen3-VL DeepStack and `mrope_interleaved` plus per-vision merged token identities. Do not prune text or special structural tokens.
- Suggested base QA B at K0=256 is 8 (alt16); K0≈512–576 16 (alt32); K0≈1280 16 (alt32). This is a **controlled search**, not claiming constant modification fraction. First get baseline QA B then add alt only when score decline/gain unclear and budget allows.
- For NeXT keep the historical same six core datasets as agreed (`GQA, POPE, MME, MMBench-EN, ScienceQA-IMG, TextVQA`), optionally add the four STAR-Pro benchmark columns if time.
- Qwen original user-requested ten tasks: `MME, MMBench-EN, POPE, TextVQA, GQA, ScienceQA-IMG, AI2D, ChartQA, OCRBench, MMStar`. Some may require model-specific generation/evaluation adaptation; `mmbench-cn` can be an extra but is not a hidden replacement for MMStar.

**Qwen3.5-9B strict gate**:
- Official model uses 32 hybrid layers, 24 linear Gated DeltaNet + 8 full attention. A conventional LLaVA-style KV pruning implementation is invalid without special handling.
- For deadline, do only `Dense / Stage1 Native K≈round(2N/9) / Stage1 Native+QA` plus K256 and K128 if desired. These are single-stage static visual-subset experiments, not full BELA Stage2; label separately in paper.
- The proposed `456→120→48` at cuts 12/24 would match T228 for N1024 if fully validated, but DO NOT schedule it automatically. Need full-attention-layer alignment, DeltaNet state consistency, M-RoPE/DeepStack, cache and text-invariant checks. If any fail, omit Qwen3.5 staged result rather than fabricate.
- V100 native BF16 and fast Gated DeltaNet kernels cannot be assumed; smoke FP16 and backend/memory, record dtype and fallback. 13B NeXT needs 32GB or planned tensor parallel, avoid mixed FP16/4bit direct efficiency comparison.

## 4. Priority queue for time-limited V100

1. **LLaVA-1.5-7B / W4096 / 77.8%**: 3 equal W schedule checks → best two Stage2 selectors → 10 full tasks & fair-256 EFF.
2. **LLaVA-NeXT-7B / T640**: validate 5-crop N and K1280, Dense/Native/QA, six full tasks.
3. **Qwen3-VL-8B / T256 (STAR-Pro-aligned)**: Dense/Native/QA, Qwen ten tasks; only then optional strict T288.
4. **Qwen2.5-VL-7B / dynamic N / T≈2N/9**, Dense/Native/QA, Qwen ten tasks; optional fixed T256 for literature comparison.
5. **LLaVA-NeXT-13B / T640**: memory smoke on 32GB-capable GPU, six tasks, record any model parallelism.
6. **Qwen3.5-9B / Stage1-only**: architecture and kernel compatibility → Dense/Native/QA with K228 reference; optionally K256/128.
7. **Qwen2.5-VL-3B**: only if already loaded, quick migration smoke or extra appendix validation.

Before dispatch, refresh `nvidia-smi` and GPU IDs; older report of GPUs 0/2/4/7 being free is **not a live allocation**. Existing V100 master/Fixed32 watcher status must be checked. One GPU/arm lock; keep full model loading, caches, prompt and scorer stable. Do not silently skip failed samples or mark early-stop arms as comparable full ten-task aggregates.

## 5. Non-negotiable source/code audit

- Read actual local `config.json` & processor: model revision, decoder layers, `vision_config`, `grid_thw`, crop count, merged patch identities, DeepStack indices, dtype, backend, quantization.
- Reproduce exact Dense OFF logits/answers. Enabling/disabling Stage1 and Stage2 must genuinely alter the target visual positions without deleting other tokens.
- Read new `K256/K1280` guard and memory profile; never reuse LLaVA-1.5 fixed 576 patch ID range on AnyRes/Qwen.
- Use (1-based) after-decoder-layer cut semantics; `K0×l1 + K1×(l2-l1) + K2×(L-l2)` is **the token-layer accounting**, not total FLOPs.
- On Qwen2.5, official configuration has 28 layers for 7B and 36 for 3B. Qwen3-VL-8B has 36 + DeepStack; Qwen3.5-9B has 32 hybrid layers.
- Output per sample: `observed_N, kept_K0/K1/K2, realized_W, realized_rate, raw_patch_ids, group_id, query_actual_swaps, predictions, evaluator_version, model_sha`.
- For large-k QA, record selector runtime and delta vs Native; avoid serial full-K Python loops. The earlier optimized A800 batched video QA is a useful *engineering pattern*, but never assume it is already integrated or fast on V100 Qwen/NeXT.
- Same model & same dataset full accuracy and latency should be paired. Cross model dense benchmarks are incomparable without per-model own reference.

## 6. Deliverables

- `V100_CURRENT_MODEL_AUDIT.json`: checkpoint, real layers, N histograms, dtype, GPU capacity.
- `V100_BELA_77P8_W4096_ACC.md`: 3 schedule checks, 2 finalist 10-benchmark rows, B8/B16 if run.
- `V100_CROSSMODEL_PRIMARY_ACC.md`: NeXT/Qwen Dense/Native/QA results, n, evaluator, rate.
- `V100_TOKEN_BUDGET_REALIZED.csv`: all sample-level W and effective ratios (record median and p10/p90 if variable N).
- `V100_FAIR256_TTFT_FIXED32.md`: same-model, same-card timing and baseline speedup.
- `V100_PAPER_READY_CLAIM_AUDIT.md`: keep paper and local results separate, compare exact compression convention.

## 7. Cursor immediate command text

> Audit the actual V100 checkpoints, model configs, GPU memory, data harness, and native-BELA source first. Confirm whether the historical Stage1 K allowlist still blocks K256. Freeze the LLaVA-1.5-7B 77.8%-layer-average W4096 schedules exactly: 256→74→36, 256→80→32, 256→68→40, cuts after decoder layers12 and20. Keep the native Dual COVER objective and QA exchange semantics; do not silently replace with a TopK proxy. Check three schedules with QA B8+first RDRIFT on complete GQA/POPE/MME, choose one balanced schedule, then evaluate both first-RDRIFT/later-QATTN and all-QATTN on all ten specified benchmarks at the same W. Next apply the provided CSV presets to NeXT7B/13B, Qwen2.5VL7B and Qwen3VL8B; use actual N to adapt any dynamic-token inputs. NeXT77.8 uses T640, Qwen3VL standard paper budget is T256; Qwen3.5 uses Stage1-only until hybrid-state pruning is proven. Always run Dense/Native/QA with matching frame/crop/image input, N/K/W logs, and no silent sample skipping. Generate all reports and keep V100 published claims tied to measured local results.

## 8. Primary research and config sources

- STAR-Pro original model-specific Table5/Appendix: https://arxiv.org/html/2609.05916v1
- Qwen2.5-VL official dynamic-resolution processor instructions: https://github.com/QwenLM-corp/Qwen2.5-VL
- Qwen2.5-VL-7B config (28 layers): https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json
- Qwen2.5-VL-3B config (36 layers): https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json
- Qwen3-VL-8B config (36 layers and DeepStack): https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json
- Qwen3.5-9B config (32 hybrid layers): https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json
- CoRePrune Qwen3.5 candidate/final tiers: https://arxiv.org/html/2609.26484
- QCPruner on Qwen2.5-VL: https://arxiv.org/abs/2609.19990

**Status:** The included CSV and allocator were created and validated in a local CPU sandbox. They do not demonstrate the model-internal patch layout, KV/DeltaNet state correctness, GPU runtime or Accuracy on the V100 cluster.
