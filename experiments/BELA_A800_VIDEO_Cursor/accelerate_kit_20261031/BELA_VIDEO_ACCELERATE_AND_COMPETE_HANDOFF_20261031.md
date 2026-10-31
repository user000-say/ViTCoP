# BELA Video Engineering Acceleration + Competitive-Accuracy Handoff
**Date:** 2026-10-31  
**Status:** REVIEW / remote GPU not accessed; production code not available in this session  
**Scope:** A800 LLaVA-Video-7B 64-frame path. Preserve the frozen V100 image method; no new video research module unless conditional gates justify it.

## Executive decision

**Do NOT stop at the fast `NATIVE_L2_TOPK` proxy, and do NOT assume code optimization can improve the existing 58.30% accuracy.** Implement two separately measurable workstreams:

- **Efficiency**: exact-output optimization of per-frame QA-COVER (eliminate 64-frame serial Python/CUDA sync, reuse similarity/Query, batch swap candidates). This phase must reproduce existing QA keep IDs and predictions.
- **Fidelity/accuracy**: faithfully port actual image Dual-depth Native COVER and correct GPU-side current-layer text-query/visual-key attention; only then compare `B=0/2/4/8` at T32 on a disjoint development subset.
- T32 is primary; T16 is optional after T32 is verified. No new temporal coverage module, no sweeping many schedules or architectures.

## 1. Existing on-machine evidence — use as immutable reference

- GPU: node-250, A800; verify actual UUID, checkpoint, runtime, frame tensors, git SHA.
- Video-MME w/o subtitles (2700): Dense 63.26%; current per-frame L2 TopK Native + Q_ATTN 58.63%; QA B8 + all Q_ATTN 58.30%; QA B8 + RDRIFT_FIRST 58.30%.
- 64 frames, 169 raw patch tokens/frame, nominal T32 / 28 decoder layers: K per frame `64→11→4` after layers 12/21; keep the additional per-frame structural token outside patch budgets.
- A800 TTFT (warmup10/measure50; CPU decode excluded): Dense 2090 ms; Native proxy 984 ms; QA Q_ATTN 1601 ms; QA RDRIFT 1596 ms.
- Operator-counted FLOPs Dense 215 T, pruned 78.4 T; the latter does not count all selector overhead.
- Remote prior report: `LIVE_VIDEO_CALLMAP`; Stage1 currently `live_stage1.py` selects `NATIVE_L2_TOPK` and runs `select_question_anchored` once per frame in `qa_core.py`. It uses `dual=(pooled,pooled)` and includes CPU sync such as `float(std)`. Stage2 `live_stage2.py` uses `hidden_states.float().cpu()` and `txt @ vis` proxy. **These statements are supplied by the operator; verify in the actual source.**
- Predicted problem: ~615 ms QA-specific extra TTFT and shared Stage2 D2H overhead.
- Do not claim current video result evaluates exact Dual-depth BELA or official STAR-Pro QR.
- Every source file and run must be hashed and versioned. Avoid mixing old results and optimized runs without explicit source tags.

## 2. Comparable external reference, not assumed head-to-head reproduction

- STAR-Pro arXiv:2609.05916v1 reports 64-frame LLaVA-Video-7B, 28 decoder layers, **T32** `64→11→4`, `Video-MME w/o subtitles=61.6`; T16 `32→5→3`, `Video-MME=60.0`; Dense=63.6. Its T16 one-token A800 speedup 2.24× is NOT a matched T32 latency measurement. Paper code for video is not in the public LLaVA overlay.
- QCPruner arXiv:2609.19990 and CoRePrune arXiv:2609.26484 report strong LLaVA-Video-7B ~32/frame Video-MME (~63.3 and ~63.1) under their own benchmark/harness; different Dense and score protocols can shift absolute scores.
- Published FrameFusion ICCV 2025 / EarlyTom CVPR 2026 / MeToM CVPR 2026 show video-centric spatial-temporal reuse and often compress within/preceding vision encoder; BELA does not currently save the full ViT computation.
- Use paper numbers in a clearly marked *reported literature* section. A strict result requires common checkpoint, frames, prompts, decode/scorer, K, profiler.

## 3. P0: Instrumentation + causal isolation (no algorithm changes)

### P0.A Reference/manifest
- Lock `model_sha, git_sha, tokenizer_sha, vision_features_sha, frame_timestamps_sha, sample_id, video_id, num_frames, input_patch_ids, prompt_sha, attn_backend`.
- Separate offline `frame_decode/preprocess`, `vision+projector`, `native_cover`, `qa_query`, `qa_similarity`, `qa_swap`, `stage2_cut1`, `stage2_cut2`, `decoder_prefill`, `decode`.
- Time whole generate via synchronized CUDA events; exclusive per-op times should not be blindly summed if overlapping asynchronous work. Also capture CPU wall time and timeline of kernel gaps/CUDA sync.
- Count `num_qa_frames`, attempted/accepted swaps, donor/drop evaluations, `.item/.cpu/.tolist` hot-loop calls; record GPU allocation/peak.
- Save baseline immutable result table.

### P0.B Three-run isolation on frozen input
1. `NATIVE_ONLINE`: existing native keep ID selection + same Stage2 as C.
2. `QA_ONLINE`: existing QA keep IDs computed live + same Stage2 as C.
3. `QA_REPLAY`: obtain QA keep IDs **before timed run**, then replay exactly the same IDs without live QA scoring; preserve sequence order/hidden tensors/Stage2/answer.
- If QA_REPLAY moves close to NATIVE_ONLINE: live selector is the dominant extra work. If not, profile differing survivor gather/attention/caches.
- QA_REPLAY is *diagnostic only*, never report it as an online method speed.
- For QA_ONLINE vs QA_REPLAY, exact original patch IDs and generated answers must match for each paired sample.

**P0 deliverable:** `VIDEO_CAUSAL_TIMING.md` + raw trace and 20-sample timing panel.

## 4. P1: exact-equivalent QA selection optimization (highest priority)

All steps maintain the **same score formulas, search candidates, thresholds, tie-break behavior and max B**. If an optimization changes selected identities, treat it as failed equivalence (unless a separately documented precision tie is rigorously resolved).

1. `query_embedding`: construct once per sample on GPU, apply correct text content mask, then broadcast per frame. No repeated tokenization, projection or `.cpu()`.
2. `dual=(pooled,pooled)`: when source shows identical features and equal fusion, calculate the same Gram similarity once, not twice. This duplication removal is only valid in this current video proxy, **not** in the restored genuine dual-depth selector.
3. Group all same-length 64 frames as `[64,169,d]`; compute batched cosine normalization/Gram once (`torch.bmm`), retain on GPU. Avoid Python `for frame in range(64)`.
4. Batch donor (<=16) and drop (<=12) candidate sets, including Query gain, novelty, coverage guard and pair objective. Select exact same pair with identical stable tie semantics. For `coverage C(S)=mean_i max_{j in S} sim(i,j)`, use the first/second best representatives for **exact one-swap marginal** in O(N×donors×drops), not a full recomputation for each pair. **Before using this optimization verify that production `qa_core.py` actually uses this facility-location coverage objective.** The accompanying reference implementation only proves this mathematical sub-operation, not BELA as a whole.
5. Carry a fixed-size `[F, K]` kept-ID table and on-GPU Boolean `active_frames`. Iterate at most B rounds *across the frame batch*; failed/stopped frames update by identity mapping (`torch.where`), not Python `if` driven by CUDA Tensor scalar. It is okay to do B=8 GPU rounds; do not do F×B Python-controlled rounds.
6. Do not use `.item()`, `float(cuda_tensor)`, `bool(cuda_tensor)`, `.cpu()`, `.tolist()`, CPU NumPy or dynamic `nonzero` in the CUDA hot path. Use debug-only validations outside timed inference.
7. Preserve original sequence order after selecting `keep` IDs; never alter frame marker positions or patch identity. Ensure accepted swap histories replay to exactly the same final IDs.
8. `torch.compile` / CUDA Graph / custom Triton kernel are **conditional optimizations** only after batched correctness and basic profiling. Do not hide additional warmup/compile overhead from production TTFT.

**P1 correctness gate:** reference vs optimized `QA B8` yields same original Patch-ID set and order, same accepted swap trace/threshold decisions, same answers on a frozen sample panel (1, 2, 8 and 64 frames, including flat Query, duplicate candidates, ties, empty donor, B0 and near-threshold guard cases).

**P1 efficiency gate (target, not a guarantee):** optimize incremental QA-only overhead from observed ~615 ms to <=100 ms on representative 64-frame paired input, without accuracy or selection regression. If not met, collect an exclusive CPU/GPU profile and fix the dominant hotspot rather than changing B under the guise of optimization.

## 5. P2: faithful Stage1 fidelity (separate from zero-change speed optimization)

**The currently faster `NATIVE_L2_TOPK` is not Dual-depth Native COVER. `dual=(pooled,pooled)` is not actual dual-depth features.**

- Obtain the *actual* ViT intermediate and penultimate features used in the frozen V100 method if the video backbone supports them; align their patch identities after pooling to the **169 video patch outputs per frame**. No invented 169↔576 index mapping or nearest-neighbor resampling without explicit documented correctness.
- Port the production Native COVER score/normalization/fusion as implemented. If the video vision tower has incompatible feature topology, label this an adaptation and verify representation fidelity; never call a single-layer proxy a true Dual COVER.
- Apply QA-COVER on the genuine Dual COVER candidates using the original global patch IDs, same q scoring semantics, same novelty/coverage guard, frozen per-frame B. Batch per-frame operations to preserve performance.
- First run a fidelity test comparing F=1 with the image selector on aligned features. Then test F=64 without cross-frame aliasing.
- **New algorithmic arm identities** (must not reuse the baseline labels): `V_NATIVE_L2`, `V_DUAL_COVER`, `V_DUAL_QA_B2/4/8`.
- Correctness to production *image* selection is evaluated only when the same features, query and score exactly align. Otherwise report explicitly which adapter changes were required.

## 6. P3: remove video Stage2 CPU offload and make attention semantics correct

- Current `hidden_states.float().cpu()` of shape roughly `[4160,3584]` is ~56.9 MiB of float32 moved to host at first cut. It is a shared execution tax, not the ~615 ms QA-specific causal gap.
- Remove D2H and CPU-side `txt @ vis` proxy; calculate candidate ranking **on GPU** at the relevant decoder layer using the actual text Q and visual K projections. STAR-Pro's paper also allows direct Q/K scoring without materializing full quadratic attention.
- For correctness, capture the **same layer's Q/K** before the layer's attention is applied (or record correct attention probabilities from that layer), not Q/K recomputed from next-layer post-attention hidden states.
- Account for `RoPE`, Q/K scaling, grouped-query attention head expansion when needed, causal mask, frame separator and question-only Query mask; to recover actual attention *probabilities*, normalize over all valid keys (not only selected visual keys) for each text query/head before aggregating across queries/heads.
- Compute only relevant text→visual entries and required softmax normalizers, not full `[S,S]` attention maps. Preserve SDPA/Flash main attention backend; selection instrumentation should not force whole-model `eager`.
- Keep token identity, frame markers, `position_ids`, `cache_position`, pruning boundaries, seq length and KV cache consistent. Preserve per-frame 64→11→4 patch budget; 64 structural frame markers remain separately tracked.
- Verify short-video exact Q_ATTN ranks against native attention probabilities at a toleranced FP32 reference (`topk` ties audited); `RDRIFT_FIRST` uses aligned original patch IDs for `old/now`. Do not call raw hidden dot product attention.
- This **changes the current video proxy score**, so evaluate accuracy afresh. It is *not* the same kind of correctness-preserving micro-optimization as P1.

**P3 deliverable:** `VIDEO_STAGE2_QK_AUDIT.json` + `VIDEO_PHYSICAL_PRUNE_TRACE.jsonl` + `VIDEO_ATTENTION_BACKEND_DIFF.md`.

## 7. Minimal quality candidate matrix (development only)

**First freeze one true per-frame T32 protocol: F64 / N169 / L28 / K0=64 / L12→11 / L21→4.** Keep nonpatch structural tokens outside the quota. Do not scan F or prune-layer locations now.

| Arm | Stage1 | Stage2 | Reason |
|---|---|---|---|
| R0 | Old L2 Top-K | GPU Q_ATTN | New Stage2 vs old 58.63 reference |
| R1 | **Actual Dual COVER** | GPU Q_ATTN | Entry restoration effect |
| R2 | Dual COVER + QA B2 | GPU Q_ATTN | Conservative per-frame Query correction |
| R3 | Dual COVER + QA B4 | GPU Q_ATTN | Intermediate |
| R4 | Dual COVER + QA B8 | GPU Q_ATTN | Original max-swaps strength |

- Do **not** immediately run full 2700 for every candidate. First use a disjoint, frozen development panel of short/medium/long videos (e.g., 120 videos × 3 questions, if legally available). Model and frame sampling fixed. Results already inspected on the full Video-MME should be labeled exploratory, not an untouched blind test.
- Score overall + per-duration + latency. Select **one** T32 candidate by a predeclared accuracy/latency tradeoff. If same accuracy within statistical uncertainty, prefer lower B or Native.
- If actual Dual COVER+QA improves T32 meaningfully, optional first-selector RDRIFT vs Q_ATTN tiny development check; previous proxy C/D had identical overall Video-MME, so no reason to default to expensive new Stage2 sweeps.
- Only after T32 yields promising accuracy/latency, test one T16 port using `32→5→3`, B0/B2/B4 on the development panel and one final winner.
- No automatic new temporal guard, frame compression/metadata decoder, motion network or additional learned component. They change the scientific method and scope.

## 8. Success thresholds and honest decision rules

- **Engineering success:** online QA selection produces exactly the same indices/predictions as old QA, with much lower isolated overhead. The earlier 58.30% stays 58.30% unless another clearly labeled semantic fix is applied.
- **T32 accuracy goal, not forecast:** Video-MME **≥61.6%** (STAR-Pro's reported T32 result), ideally closer to the 2026 query-conditioned coverage competition. A larger gap remains a valid negative finding, not permission to relabel proxy as BELA.
- **Efficiency goal, not forecast:** return toward the current Native proxy `~984 ms` TTFT; with 40–100 ms online QA overhead and otherwise identical execution, arithmetic scenario is `~1024–1084 ms`, i.e. `~2.04–1.93×` vs our 2090-ms Dense. Genuine Dual COVER and corrected Q/K attention may change these numbers, so **re-measure both**.
- Do not use STAR-Pro's `807.4 ms` **T16** as the latency goal for our **T32**. Compare 2.24× T16 speedup only as a clearly different-budget literature reference.
- **Full accuracy:** Dense, original proxy reference and *one frozen optimized T32* on Video-MME; optional MVBench/MLVU/LongVideoBench if data and protocol ready. Include short/medium/long, per-sample manifest & valid denominators.
- **Full efficiency:** same A800 UUID, batch1, 10 warmups + 50 measures, one generated token, CPU decode excluded; Fixed32 separately, operator-counted FLOPs separately, peak memory, per-stage exclusive host/GPU wall and selector cost. Run latency quietly, not mixed with other GPU inference.
- **Publication:** say `same backbone, matched frames, matched average visual token budget` when true, and mark published/reference scores versus same-harness reproductions; never report literature speed as our speed or cached QA_REPLAY as real speed.

## 9. Required deliverables (keep concise)

1. `VIDEO_REPRO_LOCK.json` — config hashes, F/K/layers/B, frame and question manifest, backend.
2. `VIDEO_CAUSAL_TIMING.md` — Native vs online QA vs QA keep-id replay; breakdown of ~615 ms.
3. `VIDEO_QA_BATCH_EQUIVALENCE.json` — old/new keep IDs and answers, boundary tests, selection traces.
4. `VIDEO_DUAL_COVER_MAPPING_AUDIT.md` — whether true dual-depth is possible and correct at F=64.
5. `VIDEO_STAGE2_QK_AUDIT.json` — correct Q/K attention, no CPU transfer, physical prune/cache checks.
6. `VIDEO_DEV_MATRIX.md` — only the 5 listed T32 candidates + their per-duration/TTFT.
7. `VIDEO_FINAL_VIDEO_MME_AND_EFF.md` — selected arm accuracy, TTFT/Fixed32/FLOPs, honest reference rows.

## 10. Immediate Cursor action

> Work only in the A800 video worktree. First lock the existing 64-frame source and read `live_stage1.py`, `qa_core.py`, `live_stage2.py`. Run a 20-sample `NATIVE_ONLINE vs QA_ONLINE vs QA_REPLAY` timing audit to isolate online selection cost. Implement **exact-equivalent, batched 64-frame QA** and verify equal original patch IDs / QA exchange histories / answer strings before making any semantic change. Next port the **real Dual-depth Native COVER** with explicit 169-patch identity mapping and replace `hidden_states.float().cpu()`/`txt@vis` with same-layer GPU Q/K-based attention scoring while preserving physical pruning and cache behavior. Then perform only R0–R4 on a frozen development panel, freeze one T32 candidate and run full Video-MME + matched A800 EFF. Do not change the V100 image path; do not silently use old L2/pooled proxies under the BELA name. If a scientific protocol or model topology prevents exact port, fail closed with a precise report.

## 11. Source links verified 2026-10-31

- STAR-Pro original: https://arxiv.org/html/2609.05916v1
- STAR-Pro public release (video code not in LLaVA overlay): https://github.com/EasonAI-5589/starpro
- QCPruner original: https://arxiv.org/html/2609.19990
- CoRePrune original: https://arxiv.org/html/2609.26484
- FrameFusion ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Fu_FrameFusion_Combining_Similarity_and_Importance_for_Video_Token_Reduction_on_ICCV_2025_paper.html
- EarlyTom CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_EarlyTom_Early_Token_Compression_Completes_Fast_Video_Understanding_CVPR_2026_paper.html
- MeToM CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Wu_MeToM_Metadata-Guided_Token_Merging_for_Efficient_Video_LLMs_CVPR_2026_paper.html

**Important limitation:** The provided QA coverage-delta reference implements the facility-location per-swap formula for benchmarking the *mathematical incremental update*. Its actual match to production QA must be checked against the unseen current `qa_core.py`. It is not a complete QA-COVER implementation, and the CPU test is not a GPU performance measurement.
