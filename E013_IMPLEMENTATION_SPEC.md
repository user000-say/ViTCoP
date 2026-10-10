# E013-P1 Implementation Spec — Stage1 Headroom / Transient Processing Utility

**Owner:** Code Agent | **Date:** 2026-10-09 | **Approval:** User explicitly requested complete code and a bounded 8-GPU Cursor handoff. **Status:** Complete source generated; CPU validated; GPU not yet run. Scientific mechanisms are hypotheses.

## Goal

Separate two different possible Stage1 improvements while **freezing the proven Stage2** (LLaVA-1.5-7B, FP16, SDPA, batch1, E012 S=D=8, K96, frozen E006 QED scoring). No QED-RS, no new training, no new pruning layer or decoder edits unrelated to eligibility transport.

* H1: fixed-budget Candidate Repair: replace b of original CLS Top172 with b candidates OUTSIDE original Top172, maintaining exactly 172 prefill vision tokens. Matched b=16 controls RANDOM/ATTN/COVER.
* H2: transient Processing Utility: keep original CLS Top172 intact and append b real/NULL/DUPLICATE candidates; they participate in true SDPA until POST-layer D=8, must NOT be eligible for final K96, and must be masked ONLY from the QED statistic side-copy. Matched b=16 ATTN/RANDOM/COVER/NULL/DUPLICATE; b=0,8,16,32 curve.

Candidate generation uses 576 existing CLIP vision patches, raw CLS attention, and existing CLIP patch representations. It requires no second CLIP run and no extra cross-modal model.

## Exact files and code insertion

* `LLaVA/llava/model/multimodal_encoder/clip_encoder.py`: add opt-in `E013_*` config, guards, Stage1 budget/patch selection, real or placebo extra embedding slots, and eligibility labels 0 (persistent) / −3 (transient). No change when E013 off. Critical insertion points are immediately before existing E009/E007 Stage1 branch selections and the cluster_labels sentinel constructor.
* `LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py` (NEW): pure torch candidate selection and cheap prototype COVER scorer; no data-dependent `.item()` in regular GPU path.
* `LLaVA/llava/model/language_model/vitcop_modeling_llama.py`: use already validated **E012-SD-P1 code** as base (SHA256 `614b86c9...`), add E013 enable/invariants, accept varying 172+b context-only label masks in the decoder, and route existing `CONTEXT_ONLY` mask to already validated E008/E009 QED side copy. No change to real SDPA, RoPE, existing QED score, cache deletion machinery, position IDs, generate or mm_projector.

`clip_encoder_E013.py` and `vitcop_modeling_llama_E013.py` are full replacement Python files. `e013_install.sh` verifies exact input hashes BEFORE writing and saves backups.

## Frozen math

Let `a_i` be original CLS attention sum, `P=Top172(a)` in patch-index order and `R={0,...,575}\P`.

**REPLACE** (budget b): `A=Top(172−b)(a)`, `B=Top_b(score_i)_{i∈R}`, final ordered candidate set `A∪B`, size 172. Original Core172's lowest b positions have been replaced, not appended. QED may select any of these 172.

**EXTRA**: real input sequence `[P, B]` with B from R, or same b appended NULL embeddings or b duplicate Core embeddings. Metadata `[0]*172 + [-3]*b`. Physically run all 172+b until post-D, then QED's final selected 96 indices MUST be in P. QED side attention sees only P visual keys for the selector, but original SDPA sees P+B as normal.

`ATTN`: `score_i = a_i`. `RANDOM`: seeded torch rand scores (accuracy-only control). `COVER`: region prototype `m_g = normalize(sum_{j∈A:g(j)=g} normalize(x_j))` on a fixed 4×4 partition of original 24×24 CLIP patch grid; score `a_i [1−clamp(cos(normalize(x_i),m_{g(i)}),0,1)]`, with empty cells assigned cosine=0. `NULL`: zero CLIP-level input slots (projector biases can make nonzero projected embeddings). `DUPLICATE`: duplicate b most salient Core tokens (no new evidence, matched b positions).

**No learned weights, no SVD/eigvalsh, no 576×576 matrix, no Python greedy loop.** COVER is a cheap test heuristic with prior-art overlap, NOT claimed as final algorithmic contribution. Its sole purpose is to measure whether candidate complementarity predicts marginal usefulness beyond top-k and random.

## Computational exposure and latency

EXTRA b adds b projector slots, b LLM visual positions for layers 0..D, side-score masking and small Stage1 selection overhead. Core172 and final K96 remain fixed. No vision encoder work is saved for EXTRA. REPLACE retains exactly 172 LLM visual positions but incurs the selection operator's extra cost. Actual wall-clock benefit MUST be measured; FLOPs are not enough.

## Risks and hard controls

* Real CUDA/SDPA integration unverified in this environment: run image+prompt→prefill→decode1→decode2 before any accuracy jobs.
* Do not compare source-swapped E013 with E012 server unless deployment SHA is recorded and BASE/B0 outputs match. Restored E012 implementation in this package originates from the previous E012 Code Agent package, not a re-uploaded live server file. Server hash mismatch => HARD STOP, do not merge automatically.
* Ensure N=172+b and candidate labels exactly as documented, with selected IDs only inside persistent 0-label slots. For REPLACE N=172 and all 172 QED-eligible.
* `E013_AUDIT=True` / `E013_STRICT=True` causes GPU sync and logging. They are required only for smoke and strictly forbidden for publication wall-clock.
* Random sampling uses request order/seed, NOT stable per-sample identity; runner must use frozen ordering. Retain several seed controls if RANDOM competes with proposed method.
* LLaVA self-attention caches early visual positions even after later pruning; label 'transient' means ineligible final hidden-state survivor, not guaranteed complete information erasure from every layer's KV.
* REPLACE changes position indices of Core patches; compare matched REPLACE arms and report this caveat. Do not call it a pure representation-only causal intervention.
* NULL embeddings can become nonzero after mm_projector bias. Duplicate changes attention normalization. Neither is a perfect causal placebo, but both are important controls.
* Do not derive question-dependent oracle labels on test sets. Per-sample gold-logprob may be used only as offline diagnostic/calibration on strictly separate data, if existing runner genuinely reports it.
* **E012 severe timing equivalence anomaly:** two nominally equivalent L12 arms differ by ~94 ms Fixed32; run ABBA same-GPU crossover before asserting new wall-clock speedups. ABBA failing affects timing validity, not correctly measured task accuracy.

## Expected validation / rollback

CPU: `python test_e013_cpu.py` (real CLIP Stage1 method extracted from original class, CPU fake CLIP outputs, tests eligibility, B0 identity, budgets/controls). GPU: same exact image/question, E012 baseline vs E013 B0; several EXTRA/REPLACE variants, verify QED prefill exactly once, decode0 selections, correct L0..D KV and L(D+1).. KV, full generated output. Rollback: `bash e013_rollback.sh <backup_dir>`.

**Promotion:** only after paired panel identifies improvement vs corresponding budget-matched controls and strict Full-VLM timing is reliable. Counterfactual Utility predictor is a possible E014 module, NOT implemented now.
