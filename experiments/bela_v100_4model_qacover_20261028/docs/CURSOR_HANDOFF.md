# BELA / CVPR 2027 — Cursor Cross-Backbone V100 Handoff
**Date:** 2026-10-28  
**Status:** 23 existing QA-COVER unit tests + 16 new cross-backbone preflight tests pass on CPU. **NO V100 jobs and NO four-model ACC measurements have been run by this package.**  
**Authoritative live project:** the user's GPU repository and published model checkpoints — NOT this standalone offline package.

## 0. Decision and claims
- **Frozen contribution candidate:** native Dual COVER → bounded Question-Anchored Stage1 replacement (QA-COVER) → native RDRIFT where architecturally supported.
- **Do NOT** introduce a new method, weight sweep, token-router, CLIP text tower, late-token recovery, or second-stage patch before the 2026-11-16 AoE CVPR deadline.
- The existing LLaVA-1.5-7B W1024 QA-COVER, reported by user: GQA56.52, POPE F1 86.16, MME1715.68, MMBench61.77, ScienceQA-IMG69.61, TextVQA-OCR53.06; mean6 95.34 with project-specific Dense references. These numbers **are user-reported, not remeasured here**, and may not be inserted as results for any of the four new checkpoints.
- The output of this package is a research **source and config audit + selector contract**, not a model-specific Hugging Face code patch. The `tools/entry_bridge.py` does **not** modify visual placeholders, post-merger order, 3-D position IDs, DeltaNet state or KV caches. Do not run benchmark inference with it alone.
- Four requested models: `llava-hf/llava-v1.6-vicuna-7b-hf`, `llava-hf/llava-v1.6-vicuna-13b-hf`, `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3.5-9B`.
- LLaVA-NeXT six tasks: GQA, POPE, MME, MMBench-EN, ScienceQA-IMG, TextVQA-OCR. Qwen ten tasks: MME, MMBench-EN, POPE, TextVQA-OCR, GQA, ScienceQA-IMG, AI2D, ChartQA, OCRBench, MMStar. Same image/question IDs for Dense/Native/QA and **each model gets its own Dense reference**.

## 1. Research and implementation provenance
1. **STAR-Pro (2026), arXiv:2609.05916**: before-LLM residual QR for vision coverage, different visual crop/frame handling, text→vision attention at stage2, native per-model layer schedule. Qwen3-VL has DeepStack and uses T=128/256; LLaVA-NeXT 7B/13B has 2,880 standardized visual tokens, T=160/320 etc. STAR-Pro architecture-specific schedules are reproduced exactly in `configs/model_budget_specs.json`. Using its **public budget protocol** is a baseline/control decision, not BELA novelty. https://arxiv.org/html/2609.05916
2. **CoRePrune (2026), arXiv:2609.26484**: conditional removability differs by decoder depth and delete set; on Qwen3.5 it reports final-K 64/128/256 (NOT layer-average W), and attention scores only at native full-attention layers in several diagnostics. It also observes an optimal intermediate candidate cap, not always max. https://arxiv.org/html/2609.26484
3. **AgilePruner (ICLR 2026)**: scene complexity, saliency and diversity dependence; cannot infer dynamic K is a novel BELA module. https://github.com/cvsp-lab/AgilePruner
4. **QCPruner (2026), arXiv:2609.19990 / CoverPruner (EMNLP 2026)**: question-conditioned coverage is prior art; it is NOT a first-of-its-kind discovery by QA-COVER. https://arxiv.org/abs/2609.19990
5. **ET-Prune (Aug 2026), arXiv:2608.01979**: query-aware, text-region safeguard and adaptive budget in OCR-heavy benchmarks; supports testing ChartQA/OCRBench, not another BELA add-on now. https://arxiv.org/abs/2608.01979
6. **AVTP (EMNLP 2026)**: dynamic ratios across multi-image inputs on Qwen3VL and other backbones. A generic adaptive-token-count controller is not a safe new novelty claim. https://aclanthology.org/2026.findings-emnlp.122/
7. **Official Qwen configs:** Qwen3-VL-8B has 36 decoder layers, 27 ViT layers, spatial merger=2, DeepStack visual indexes [8,16,24]; Qwen3.5-9B has 32 decoder layers with 24 linear-attention and 8 full-attention layers, vision depth27 and no DeepStack visual injection in its config.
   - https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json
   - https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json
8. CVPR official: registration 2026-11-10 AoE, full paper 2026-11-16 AoE; prioritize a reliable manuscript over unsound ports. https://cvpr.thecvf.com/Conferences/2027/Dates

## 2. Benchmark result provenance / do not hallucinate ACC
- `configs/results_pending.csv`: **192 unmeasured rows** = 2 budgets × 3 arms × (2 NeXT ×6 tasks + 2 Qwen ×10 tasks). All `score` cells are EMPTY and `status=PENDING`.
- `configs/paper_dense_reference_NOT_MEASURED.csv`: 24 **selected published Dense** entries from STAR-Pro (two NeXT models and Qwen3-VL) and CoRePrune (Qwen3.5). Many cells intentionally blank because not reported. Published TextVQA may not match the user's `TextVQA-OCR` parsing. NEVER copy these into runtime results.
- `configs/llava15_7b_USER_REPORTED_ANCHOR.csv`: historical six-task score pairs, not part of new test.
- Metrics: GQA exact match %, POPE F1 % + precision/recall/yes-rate/confusion, MME *points* (perception+cognition), MMB-EN %, ScienceQA-IMG %, TextVQA official VQA accuracy %, AI2D %, ChartQA **relaxed** accuracy %, OCRBench score **0–1000**, MMStar %. Do not average raw MME/OCRBench with percentages.
- Each checkpoint/processor MUST be evaluated Dense under the exact same image/resolution/scorer protocol before any relative normalization. Keep benchmarks/dataset hashes constant across the three arms.
- Use `python -m tools.record_scores ...` only **after** official scorer outputs exist; it requires checkpoint/git/processor/dataset/scorer/run-path provenance.

## 3. The correct token units and target budgets
**Definition:** N = actual number of decoder-visible **image** tokens in *this sample*, after any relevant merger, crop packing and removal of separator/newline non-image tokens. L = text decoder layers. W=sum over layers of alive image tokens. Nominal layer-average compression=1-W/(N L). For cross-family comparisons, also provide operator-counted FLOPs, TTFT, Fixed32 and memory.

Reference **model-specific** schedules from STAR-Pro (BELA uses the same schedule as a CONTROL, but not its QR scorer):
| Model | N_ref | L | Main | Strong compression |
|---|---:|---:|---|---|
| LLaVA-NeXT-7B | 2880 | 32 | 640 ×12 →182 ×8 →92 ×12, W10240, avg320 (88.89% removed) | 320 ×12 →91 ×8 →46 ×12, W5120, avg160 (94.44%) |
| LLaVA-NeXT-13B | 2880 | 40 | 640 ×15 →160 ×15 →80 ×10, W12800, avg320 (88.89%) | 320 ×15 →80 ×15 →40 ×10, W6400, avg160 (94.44%) |
| Qwen3-VL-8B | 1296 | 36 | 512 ×14 →128 ×10 →64 ×12, W9216, avg256 (80.25%) | 256 ×14 →64 ×10 →32 ×12, W4608, avg128 (90.12%) |
| Qwen3.5-9B | 1024 | 32 | **pre-LLM-only** fixed128, W4096 (87.5%) | **pre-LLM-only** fixed64, W2048 (93.75%) |

Notes:
- Qwen3-VL optional deeper stress 94.44% is `144×14→36×10→18×12`, W2592. It is **mathematically constructed by BELA**, not a published STAR-Pro configuration. Do not automatically queue it.
- Qwen3.5 `K_L` in CoRePrune is the **final retained-token target**, not the same as BELA's stage1-only K or layer-average T. Our two Qwen3.5 settings are intentionally Stage1-only diagnostic controls, so they do NOT validate full BELA's RDRIFT.
- For actual N ≠ N_ref, use the JSON-defined reference counts scaled as `round(N/N_ref × K)`, then recompute *actual* W/rho; don't pretend W is unchanged by dynamic resolution.
- Stage1 K, all cut counts and original IDs must be logged **per sample**. `tools/budget_planner.py` checks schedules. No automatic reallocation across datasets based on their labels.
- Group-aware crop allocation for NeXT: keep thumbnail/base and each crop as separately addressable groups, with deterministic quota, excluding newline embeddings from the visual-K budget.

## 4. Production integration: real model-specific work remaining
### 4A. LLaVA-NeXT-7B/13B
- Use actual `LlavaNextForConditionalGeneration` or the canonical LLaVA-NeXT code at the chosen model revision; do not assume all code is byte-compatible with LLaVA-1.5.
- Capture aligned ViT layer-9 and penultimate **original crop-local visual patch representations** and map them into the real **packed** sequence after crop/unpad processing. Post-pack newline/separator embeddings are not visual patch candidates.
- Construct `group_ids[N]` (base thumbnail + local crops), align actual visual features / LLM projected embeddings with each original packed token ID; apply native group-level Dual COVER and `tools/entry_bridge.py` query admission, keeping the same K per group.
- At the original text decoder, adapt RDRIFT cuts to 12/20 (7B) or 15/30 (13B), keeping selected visual IDs, text slots, position IDs and KV cache mutually consistent.
- Mandatory off-path equivalence: model with `BELA_MODE=off` must produce exactly the same answer/token IDs as original Hugging Face path on >10 diverse images, including non-square and multi-crop.
- **No automatic 13B FP16 claim on V100-32GB**; if OOM, stop or use memory-parallel infrastructure/quantized-only labeled separately.

### 4B. Qwen3-VL-8B-Instruct
- Use **post-merger** image-token units and actual N, not pre-merger 16×16 pixel ViT patches. Preserve original (t,h,w) 3D M-RoPE coordinates.
- Qwen3-VL's `deepstack_visual_indexes=[8,16,24]` produces extra feature streams injected into the first decoder layers. Stage1 selection must gather **the same original IDs from every aligned DeepStack stream**, update `visual_pos_masks`, and preserve positional context. Token-count and model-output equivalence are required BEFORE benchmarking.
- **Do not simply average or reshape pre-merger ViT hidden features into merged Token IDs**. Derive a verified pre-to-post mapping and meaningful same-length middle/deep dual descriptors; document shapes and identity checksum. If impossible within time, label any final-only variant explicitly and DO NOT claim Dual COVER was transferred.
- Current QA-COVER uses cosine between projected visual features and question-only LLM input embeddings; this is a proxy rather than a proven alignment in a new model. Include query-shuffle / matched-random on POPE and GQA pilot.
- Stage2 cuts at 14/24 and keeps the paper-matched average budget. Use original M-RoPE coordinates; preserve `rope_deltas` and correct cache state during prefill/generation.
- Fail closed on mismatched DeepStack [N,D] or image-placeholder counts. The provided bridge only validates / selects tensor rows; it DOES NOT install the appropriate Qwen hooks.

### 4C. Qwen3.5-9B — **deliberately NOT a standard Transformer**
- `Qwen/Qwen3.5-9B` official `layer_types` contains 24 `linear_attention` / Gated DeltaNet and eight `full_attention` layers (1-indexed full layers 4,8,...,32).
- The original RDRIFT cannot read regular text→visual attention weights at every layer, and deleting tokens mid-sequence changes the recurrence/state for the Gated DeltaNet. This is NOT just a M-RoPE or ordinary KV cache adjustment.
- Test Dense and **Stage1-only QA-COVER** K128/K64 if the V100 backend and FP16 run work. Compare with exact same-model `NATIVE_STAGE1_ONLY`; do not call it "full BELA" or mix its W with CoRePrune's final-K results.
- Only after model-state equivalence is proven at cut layers, under a real `full_attention` scoring path with no future-state access, may anyone attempt a separate full two-stage port. Under the CVPR deadline, STOP at Stage1-only if this requires new recurrent-state surgery.
- Official Qwen3.5 uses BF16 weights; V100 is SM70 without native BF16, and the DeltaNet fast kernels may not have V100 support. Force an explicit FP16 smoke or use a supported, separately labeled low-bit backend; log any fallback. Do NOT assume it will fit or be fast.

## 5. V100 hard gates (in this order)
**G0 — resource/preflight, zero model inference**
```
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
python -m tools.budget_planner --model llava-next-7b --tier stress --observed-n 2880 --groups 576,576,576,576,576
python -m tools.budget_planner --model qwen3-vl-8b --tier stress --observed-n 1296
python -m tools.model_config_probe --config-json /real/cached/checkpoint/config.json
python -m unittest discover -s tests -v
(cd qa_cover_original && python -m unittest discover -s tests -v)
```
These do not load GPUs. Probe config *from the exact cached model revision*, not a model card copy.

**G1 — single-sample no-prune equivalence and prefill proof**
- First run **Dense** using native model path, then `off` wrapped BELA and assert identical generated answers and token IDs under deterministic decoding.
- Check actual post-merger image N, image placeholder count, crop group sizes, raw patch IDs, input embedded text/image boundaries, M-RoPE, DeepStack streams and required cache state; one full-sequence pass with original inputs required.
- ViT intermediate/penultimate descriptors must be uniquely aligned to *the same* post-merge image ID.
- Compare generation with `enabled=False` to original, including one non-square image for NeXT and at least two different resolutions for Qwen.

**G2 — selection correctness / performance pilot**
- Model-specific BELA_NATIVE and QA-COVER: same N per image, same actual layer schedule, same vision backbone output values, same checkpoint/prompt/sampling/scorer.
- Run 64–128 paired samples across POPE, GQA, TextVQA and OCR-related tasks **only after G1**; verify at least some sample answers change and QACOVER is not silently a no-op.
- On each new architecture, check swapped IDs, actual survival, query-shuffle or matched random for POPE/GQA. If real question is not better than random/mismatched control, don't write query-grounded mechanism claims.
- STOP immediately on shape mismatch, position/crop error, invalid cache, OOM, output invalidity or significant TTFT penalty; do not hide problems under a higher W.

**G3 — requested full evaluation**
- NeXT7 then NeXT13: six LLaVA tasks, two budgets (the `results_pending.csv` rows).
- Qwen3-VL: ten Qwen tasks, two official-layer-average budgets T256 and T128. Only after these and time allow the mathematically-derived T72 stress.
- Qwen3.5: ten tasks only if native FP16/stage1 pipeline stable; otherwise document exact blocker and stop without fabricated metrics. Its Stage1-only results are clearly labeled, separate from full-BELA main results.
- Exact deterministic question IDs, dataset version, official scoring, POPE confusion and Yes-rate, GQA per-type analysis. Do not tune query admission on public test labels.
- Add same-backbone DENSE / native / QACOVER, not a hybrid score stitched from different papers, and if possible reproduce STAR-Pro at matching N/W for NeXT/Qwen3VL.

**G4 — real efficiency**
- Same GPU (no cross-GPU absolute speed claim), batch1, FP16/eager or SDPA compatible on V100; no BF16/FlashAttention2 assumption.
- TTFT p50/p90; 32-token fixed decode; warm10/meas50 when feasible; time separately: image preprocessing, full ViT, post-merger features, projector for all N (if required), QA-COVER, decoder prefill/decode and peak VRAM.
- W equality does not imply same actual FLOPs/TTFT; model family and fused operator differences require independent reporting.

## 6. SCOREBOX / freeze conditions
- `configs/results_pending.csv` starts blank. Do not copy `configs/paper_dense_reference_NOT_MEASURED.csv` into those cells.
- At minimum require QA-COVER > native on POPE/GQA or equally challenging model-relevant tasks, no major TextVQA/OCRBench/ChartQA loss, and no unacceptably high selector overhead. Report any negative result.
- The paper should make calibrated claims: NeXT is a meaningful extension of the same LLaVA model family; Qwen3-VL is cross-family proof only if DeepStack and multi-position semantics are verified; Qwen3.5 Stage1-only supports entry admission but not progressive Stage2.
- New model runs must NEVER block already-reported LLaVA-1.5-7B W1024/W2048 baseline verification, 2026-11-10 enrollment and 2026-11-16 full CVPR manuscript.
- CPU source validation here establishes *shape/id/budget logic only*. It does **not** establish model inference correctness or GPU Accuracy gains.

## 7. Cursor must report back (copy exact file paths)
1. `LIVE_MODEL_CALLMAP.md` for each actual class and input/output / DeepStack / merged-position tensor shapes.
2. `PATCH_DIFF_<model>.patch` merged into live repository, disabled path exactly original.
3. `GPU_FREE_MEMORY.json` and `FP16_BACKEND_SMOKE.json`.
4. `MODEL_REGRESSION_<model>.json` with original/disabled token-level agreement.
5. `TOKEN_TRACE_<model>.jsonl`: original image ID, N, group, entry IDs, per-cut IDs, positional checksum, W, cache and DeepStack check status.
6. `PREDICTIONS_<model>_<dataset>_<arm>.jsonl` and official scores joined by unique sample index, never bare POPE question_id.
7. `RESULTS_WITH_PROVENANCE.csv` + per-model Dense denominators + paired delta tables.
8. `LATENCY_<model>.json` with full algorithm overhead and peak VRAM.
9. `MODEL_PORT_DECISION.md`: supported/blocked reasons and whether it proves full BELA or Stage1-only.
