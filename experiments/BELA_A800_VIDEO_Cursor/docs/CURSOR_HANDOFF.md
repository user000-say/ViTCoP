# BELA / CVPR 2027 — Two-A800 video port: Cursor Handoff
**Prepared 2026-10-28. State: offline reference code only; no SSH login, no A800 GPU jobs, no video model accuracy measurements.**

## 0. Security
User shared remote login information in chat. **No password, username, IP, or SSH host is saved here.** Rotate the exposed password and use SSH public-key login; don't use `sshpass` or credentials on command lines. Connect only in an authorized Cursor SSH session. Record no secrets in logs.

## 1. Decision
- Required video model: `lmms-lab/LLaVA-Video-7B-Qwen2`, freeze checkpoint revision. Backbone: SigLIP SO400M, 26 vision layers, Qwen2-7B language model with 28 decoder layers; 64 video frames maximum.
- Benchmark input: **64 uniformly sampled frames, 169 actual pooled visual tokens per frame** = 10,816 visual candidates; exclude timestamps and all structural tokens from visual selection.
- Algorithm: **existing Dual COVER (frame-wise) → unchanged QA-COVER residual admission (frame-wise) → original RDRIFT adapted to video 28-layer schedule**. Never port LLaVA-1.5 CLIP patch indices 0..575 to this model; the relevant visual selection unit is one of the *169 post-spatial-pooling tokens per frame*.
- Keep original fixed QA-COVER equation and hyperparameters from previous user-validated LLaVA-1.5 result. Only scale `max_swaps` proportionally to entry K: 8 at K64, 4 at K32; this is an explicit portability assumption, not tuned video evidence.
- No global temporal token router, text tower, separate video head module, or new architecture before paper submission. Video is a generalization test, not an excuse to reopen algorithm search.

## 2. Primary-paper evidence and methods
- **STAR-Pro**, arXiv:2609.05916, Table 2 / 5 / 10 and Efficiency Table 4: 64×169 on LLaVA-Video; T32/T16 are **layer-average** visual tokens/frame; evaluates MLVU m-avg, LongVideoBench val, Video-MME *without subtitles*; uses 10 warmups/50 measures, one A800-80GB, frame-wise selection. https://arxiv.org/html/2609.05916
- **CoRePrune**, arXiv:2609.26484, Table S1 / S6: 64×169, **final** K64/K32/K16 per frame, on MVBench test, LongVideoBench val and Video-MME w/o subtitles. It prunes in the ViT at seven layers [9,11,13,15,17,20,22], then refines after zero-based Decoder layer7; therefore **its final K is NOT STAR-Pro's layer-average T**. https://arxiv.org/html/2609.26484
- **HieraVid**, arXiv:2604.01881 (Apr 2026), tests LLaVA-Video-7B, LLaVA-OneVision-7B and Qwen2-VL-7B; evaluates MVBench, EgoSchema, NExT-QA, Video-MME, and uses segment-, frame- and decoder-layer-level reduction. https://arxiv.org/html/2604.01881
- **FastVID**, NeurIPS 2025, uses temporal segmentation + dynamic density across frames, including LLaVA-Video, LLaVA-OneVision and Qwen video models. Fixed per-frame allocation is a *controlled choice*, not globally optimal or novel. https://proceedings.nips.cc/paper_files/paper/2025/hash/b2e63e36c57e153b9015fece2352a9f9-Abstract-Conference.html
- **Unified Spatiotemporal Token Compression**, CVPR 2026, merges/compacts spatiotemporal video tokens and LLM tokens at ultra-low retention (~1–2% on some protocols). Incomparable budget/architecture to a frame-wise BELA Video-T16 until exact conditions match. https://openaccess.thecvf.com/content/CVPR2026/html/Du_Unified_Spatiotemporal_Token_Compression_for_Video-LLMs_at_Ultra-Low_Retention_CVPR_2026_paper.html
- **Model checkpoint**: https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2 (Qwen2 + SigLIP, max64 frames, official `load_pretrained_model`, `modalities=['video']`, `qwen_1_5` chat template).
- **Reference implementation**: https://github.com/LLaVA-VL/LLaVA-NeXT
- **Eval toolkit**: https://github.com/EvolvingLMMs-Lab/lmms-eval. In installed version discover task IDs with `python -m lmms_eval --tasks list`; do not invent model wrapper names.
- **Video-MME**: https://github.com/MME-Benchmarks/Video-MME (900 videos/2700 QAs, multiple question IDs per video; academic-use licensing).

## 3. Schedule exactly matching STAR-Pro's nominal video budgets
Cuts are AFTER 1-based decoder layers12 and21 (after zero-based layer indices11 and20).
| Budget | Decoder layers 0-11 | layers12-20 | layers21-27 | W (64 frames, 28 layers) | Avg visual tokens/frame/layer | W reduction vs dense |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 169×12 |169×9 |169×7 |302848 |169.00 |0%|
| Video-T32 | 64×12 |11×9 |4×7 |57280 |31.96 |81.09%|
| Video-T16 | 32×12 |5×9 |3×7 |28800 |16.07 |90.49%|

**STAR nominal T32 != final32.** Actual final framewise K is4 (T32) or3 (T16).
**CoRe final32 != layer-average32.** If its candidate K64/frame persists for first8 decoder layers before finalK32, a *rough* LLM-layer W proxy is 64×(8×64+20×32)=73728, vs STAR T32 W57280; plus CoRe saves ViT computation separately. No cross-method compute superiority claim without identical instrumentation.

## 4. Live model engineering contracts; do not infer GPU accuracy from CPU checks
1. Pin actual model checkpoint, repo git SHA, mm projector, tokenizer, Eval commit, BF16/FlashAttention backend. Verify A800 model and per-GPU VRAM using `scripts/a800_preflight.sh`; two physical GPUs do not merge memory unless you implement model parallelism.
2. Inspect real `llava.model.llava_arch.get_2dPool`, SigLIP feature select layer, spatial pooling mode/stride. Capture middle and penultimate SigLIP views; map them using the **exact native video pooling operation** onto the same `[F,169,D]` pooled units as final decoder-visible features. The checkpoint's pooling may be bilinear; verify actual code and all shapes, not a guessed reshape or average pooling.
3. Every candidate ID is a tuple **(frame_idx, pooled_patch_idx 0..168)**. The `frame*169+patch` ID in the reference helper is **only a visual-only flattening ID**, never a full LLM sequence offset. The real model must track separate full-prompt positions including timestamps and nonvisual text. Never delete time separators or question tokens.
4. Compute native Dual COVER separately for each frame; preserve original semantics; initial K64/frame at T32 or K32/frame at T16. Native intermediate embeddings are descriptors only; gather the ORIGINAL post-pooling projected visual embeddings for actual decoder input.
5. QA-COVER uses `bela_video_a800.video_entry.select_framewise`, with original QA core copied unchanged. Its query score is cosine between already projected 169-per-frame LLM visual embeddings and the real question-stem input embeddings, **without answer labels or generated output**. Alignment is not guaranteed across CLIP→SigLIP and Vicuna→Qwen2. Mandatory matched-random and different-video-query negative controls.
6. RDRIFT: real 28-layer Qwen2 attention, cut after layers12/21, retain **per frame** 64→11→4 or 32→5→3. Gather original IDs at each cut, preserve text/time embeddings, RoPE positions and layer-local KV/cache state. Do NOT materialize full [10k,10k] attention; compute the needed text-query × video-key scores at the current layer with correct grouped-query attention and RoPE.
7. Test `model.generate(max_new_tokens=1)` and `max_new_tokens=32` after prefill; mismatched per-layer caches or changed `cache_position` is a fatal correctness failure. Physical decoder sequence lengths must actually shrink; changing masks only is not real TTFT saving.
8. `BELA_VIDEO_MODE=off` must be token-for-token identical to OFFICIAL Dense outputs on multiple videos (at least short, medium, long; 8/16/64 frames). If not, STOP before benchmarking.
9. Production must not use the old request-global `LAST_ENTRY_KEEP` / `LAST_ENTRY_METRIC` across concurrent video requests; keep per-request/per-frame state.
10. This archive **does NOT contain a patched real `llava_arch.py` or `Qwen2` decoder forward**. Cursor must create `LIVE_VIDEO_CALLMAP.md` and actual `VIDEO_PATCH_DIFF.patch` in the remote repo and prove off/real-S2 equivalence. Do not call this package a turnkey working video model.

## 5. Four required video benchmarks + exact paper-aligned scores
| Dataset | Official split & metric | Main comparison | Extra report |
|---|---|---|---|
| Video-MME | without subtitles, 2700 QAs / 900 videos | STAR-Pro, CoRePrune, CDPruner | overall + short/medium/long |
| LongVideoBench | val, multiple-choice Accuracy | STAR-Pro, CoRePrune | temporal relationship category |
| MLVU | multiple-choice m-avg (not open-ended) | STAR-Pro, CDPruner | perception / relation categories when available |
| MVBench | test, Accuracy | CoRePrune, QCPruner | temporal/action subtypes |

Optional only after these four: EgoSchema, PerceptionTest, NExT-QA. Do not spend the rented A800 on open-ended GPT-judged video tasks before completing matched multiple-choice benchmarks.

The reference table `configs/paper_reported_video_scores_NOT_OURS.json` separates paper scores by provenance:
- STAR Dense: MLVU67.7 / LVB59.0 / VideoMME63.6; STAR T32:65.4/57.0/61.6, STAR T16:60.3/56.1/60.0.
- CoRe Dense: MVBench60.8 / LVB58.9 / VideoMME64.3; CoRe *final* K32:60.4/57.2/63.1; *final* K16:59.5/56.2/62.3.
- Different Dense scores are **not automatically comparable**, even at the same nominal video model. Record exact checkpoint, transcript/subtitle exclusion, sampling, prompts, scorer, and model/backend before claiming any direct lead.
- Never copy paper scores into actual BELA predictions or `configs/run_matrix_PENDING.json`.

## 6. Experiment matrix and fast promotion criteria
Full formal rows: `DENSE` once per dataset + `BELA_VIDEO_NATIVE` and `BELA_VIDEO_QACOVER` at T32/T16 on each of four benchmarks = **20 full combinations**. Two pilot-only negative controls at T32/T16 on Video-MME and MVBench = **8 additional pilot rows**.

**Preflight smokes** (batch1, fixed video+question, no scoring claims):
- official Dense 8/16/64-frame decoding works with same frozen model & backend;
- `off` patch is byte/token identical to official Dense;
- 169 pooled tokens per frame actually exist, native ID maps and actual sequence positions align;
- T32 physical K 64×64 ->64×11 ->64×4, T16 physical K 64×32 ->64×5 ->64×3;
- zero label leakage, valid timestamp order, length-32 decode after both cuts, no OOM/NaN;
- verify query mismatched across DIFFERENT videos, not via duplicate `question_id`; the evaluation prompt **remains original**.

**Accuracy pilot**: predefine Video-MME 90–150 QAs stratified by duration and MVBench 80–120 paired QAs, and record each sample key, official answer, generation text, scorer, frame timestamp hash, native/final IDs. Only advance to 4-benchmark formal runs if QA beats native plus random or wrong-video on at least one primary score without obvious regression on the other. Do not tune to test labels. Pilot results are diagnostic only.

**Full evaluation**: same videos, same extracted frames, same prompt, same decoder, same K/W. Official multiple-choice answer parsing. Video-MME without audio/subtitles. Paired differences + cluster-by-video uncertainty, not four unrelated score columns. MLVU/Video-MME aggregates never mix unrelated subsets.

## 7. Practical two-A800 resource strategy
**Before starting expensive GPU jobs**:
- Download legal video data/checkpoint and cache 64 uniformly sampled RGB frames + timestamps using CPU/V100 or off-GPU time. Validate all required clips exist and decode. Avoid automatic downloading on the leased GPUs.
- Prepare fixed dataset JSONL manifest with composite key `(dataset, row_index, video_id, question_stem)`; not raw question_id. `bela_video_a800.shuffle_control` creates deterministic different-video *selector-query-only* controls.
- Freeze a compatible `LLaVA-NeXT` checkout + official model revision + isolated Python environment. Make sure official Dense source runs and capture `torch`, `transformers`, CUDA, FlashAttention, lmms-eval versions.
- GPU memory check: each GPU must report A800 and its **actual** VRAM. Two cards in the machine do NOT merge memory. Prefer one inference process per GPU and BF16/FlashAttention if verified compatible.
- First prove Dense / off / native / QA work on tiny inputs; do not start full4-dataset jobs while model hooks are broken.

**Once both A800s are confirmed free and smokes pass**:
- GPU0: 64-frame official Dense/OFF regression + Video-MME paired pilot, then full Video-MME.
- GPU1: MVBench pilot + LongVideoBench / MLVU full accuracy, once its own single-GPU correctness smoke passed.
- Use explicit `CUDA_VISIBLE_DEVICES=0` and `=1` according to confirmed local `nvidia-smi` mapping; never kill unrelated PIDs. Keep each run's process/precision/GPU ID and CPU/GPU video decode costs in its metadata.
- One video question = one atomic JSONL result line; restart/resume must use **composite sample ID** and skip completed rows. Stage trace per sample stored separately.
- Prefer all full accuracy tasks before extended optional model work, so short A800 availability yields a valid appendix even if time expires.
- For controlled latency: avoid concurrent timing load and use batch=1, 10 warmup + 50 measured calls; report one-token TTFT and separate fixed32 generated-token latency, p50/p90, selector milliseconds and peak GPU memory. STAR-Pro's published A800 one-token timing excludes CPU video preprocessing, so give both GPU-only and inclusive pipeline times, never a stitched speedup.
- If video port fails position/cache/off-path equivalence, lacks actual physical pruning, or exceeds manageable overhead: STOP, preserve main LLaVA image submission and explicitly mark video as incomplete.

## 8. Cursor must return before any video result is citable
- `LIVE_VIDEO_CALLMAP.md`: precise original video class paths and pooled feature tensor shapes for SigLIP middle/deep, Qwen2 text token indices, `get_2dPool` mode, stage2 Q/K, video prompt position mapping.
- `VIDEO_PATCH_DIFF.patch`: actual live production code, mode off exact with official checkpoint.
- `VIDEO_GPU_PRECHECK.json`: no secrets; GPU name, BF16/kernel, free VRAM, checkpoint/scorer hashes.
- `VIDEO_OFF_EQUIVALENCE.json`: exact official/off answer/token equality, 8/16/64 frames, with failing cases if any.
- `VIDEO_TOKEN_TRACE.jsonl`: original `(frame, patch)` IDs, all per-frame survivor IDs, observed per-layer K, W, physical cache/position proof.
- `VIDEO_PREDICTIONS_*.jsonl`: real model output with exact video and question composite IDs and answer parser.
- `VIDEO_SCORES.json`: Video-MME/LVB/MLVU/MVBench official scorer original values and paired errors.
- `VIDEO_LATENCY.json`: synchronized components + full inclusive timing, same device arms, TTFT/fixed32, VRAM.
- `VIDEO_FINAL_DECISION.md`: generalization evidence, failure cases, algorithm baseline version, exact paper claims allowed.

## 9. What is actually demonstrated vs still UNKNOWN
- **OBSERVATION:** User reported LLaVA-1.5-7B QA-COVER at W1024 improves GQA56.52, POPE86.16 and mean6 95.34%. Not independently reproduced here.
- **OBSERVATION:** Published STAR-Pro and CoRePrune test 64×169 visual tokens on the LLaVA-Video-7B benchmark protocol, with **different** compression definitions.
- **HYPOTHESIS:** Question-conditioned admission into a *per-frame* SigLIP candidate pool will improve temporal VQA when coupled to native RDRIFT, with no increase in nominal W.
- **UNKNOWN:** Does the original QA query proxy work with Qwen2 embeddings? Can SigLIP pooled descriptors support true dual-depth coverage? Is staged Qwen2 caching correct? Are end-to-end A800 latency and Accuracy improved?
- **No novelty claim** from simply applying an existing still-unverified QA formula to every frame. Show independent matched controls, an unchanged algorithm, and honest compute costs.

## 10. Optional reproducible job wrapper
`scripts/run_matrix_template.sh` accepts a real integration script in the environment variable `BELA_VIDEO_RUNNER`. Cursor must write this real runnable script and document its CLI; the template intentionally **refuses to run** if there is no production integration. A valid runner must accept `--model-id`, `--dataset`, `--tier`, `--arm`, `--max-frames`, `--output-dir`; it must load the official model only once per GPU process, write per-item JSONL atomically, and never quietly fall back to the Dense model when the requested pruning arm is unsupported.
