# NeXT AnyRes / Stage1 / Physical Stage2 – exact integration requirements

## Critical: The shipped package does not contain the V100 production repo
No exact `git apply` patch can be authored without real live `llava_arch.py`,
`prepare_inputs_labels_for_multimodal`, BELA Stage1 / `modeling_llama.py`.
The modules `src/bela_next_budget` are tested, standalone components.
Cursor must use them to implement actual patches against its live tree and
return `ACTUAL_SOURCE_DIFF.patch` and a git commit/hash (if Git is present).

## 1. NeXT vision inputs, five-crop paper mode

- Check model and tokenizer, vision tower and projector SHAs; must be
  LLaVA-NeXT Vicuna checkpoint, 32 (7B) or40 (13B) decoder layers.
- Official comparability mode: force *actual* 672×672 image tiling,
  five 336×336 crops, 576 patch tokens per crop.
  `image_grid_pinpoints=[[672,672]]` as used in the STAR repo is a
  REFERENCE CONFIG; verify the processor and `mm_patch_merge_type`,
  model doesn't necessarily handle pinpoints as a simple CLI flag.
- Inspect original visual sequence after the projector and
  `unpad_image()` + image_newline insertion. Write a token-type map
  `[seq_len]` identifying **every** patch vs image_newline, text,
  padding, image-start, separators.
- Forcing five crops must be done **equally for dense and pruned**;
  do not silently change dense input resolution.
- Never gather a `crop*576+patch` tensor as if it were identical to
  sequence positions after unpadding. Track:
  `crop_group_id`, `raw_patch_id`, `post_unpad_patch_id`,
  `current_sequence_absolute_index`, `is_patch`, `is_newline`.
- Every sample logs `actual_valid_patch_n`, per-crop valid counts,
  number of special tokens, image aspect, unpadding/grid args.
  Reject any paper-mode sample where counts are not exactly
  five crops×576 patches; if official code obtains different layout,
  record a separate protocol rather than fabricating 2880.
- If native AnyRes variable crop count is used later, do not
  label the compression percentage relative to 2880 when N differs.
  Use `budget.dynamic_ratio_plan` to scale under an explicit
  `native_anyres` protocol, record achieved W/real ratio, and
  evaluate corresponding Dense with identical crops.

## 2. Stage1 candidate distribution

The **exact same Dual COVER baseline implementation** should be
adapted to crop-wise 5 groups. Crucial design choice:
- `K0=320` → 64 candidate patches/crop for five groups (uniform);
- `K0=640` → 128 per crop. Source group size 576.
- For BELA A3/A4, e.g. K0=288 can divide unevenly [58,58,58,57,57],
  K0=256 [52,51,51,51,51], with original stable crop ID.
- Avoid monolithic [2880×2880] pairwise construction if it exceeds
  V100 RAM. Crop-wise similarity [576×576] × 5; or chunked kernels.
- ALWAYS compare one baseline:
  exact legacy BELA `Dual COVER` on its intended true mid+pen
  without question versus B8 version, same crop quotas.
- Reproduce bona fide SigLIP/CLIP or vision tower mid+pen capture
  on NeXT checkpoint: these **are not the same layer numbering**
  as image LLaVA-1.5 7B unless verified from actual vision config.
  Print names/shapes and sample local patch identities.
- Do not use `(pooled,pooled)` to fake true dual-depth features.
- QA donor must be a real rejected-original pool (`576-Kcrop` per crop),
  not merely resort already kept tokens.
- QA B8 from historical BELA (max_swaps8, donor16, drop12,
  slots32, seed17) is **a starting anchor**, not a proven NeXT optimum.
  The per-crop swap semantics must be recorded: B8 globally vs B8
  per crop yields 5×different work. For the primary screen lock
  the **same semantics in all 4 budgets**, and log total swapped IDs,
  QA runtime and per-crop quotas.

Executable crop split:
```python
from bela_next_budget.crops import crop_quota_from_ids
q=crop_quota_from_ids([576]*5,total=320,policy="uniform")
assert q == [64,64,64,64,64]
q640=crop_quota_from_ids([576]*5,total=640,policy="uniform")
assert q640 == [128,128,128,128,128]
# Optional after schedule+QA frozen:
boosted=crop_quota_from_ids([576]*5,total=320,
                            policy="thumbnail_boost",global_weight=1.5)
assert sum(boosted)==320
```
The crop ids and original patch IDs **must be preserved** through
Stage1 QA and two decoder stages. Restored order is the original
image spatial/crop ordering.

## 3. Decoder Stage2 and full sequence preservation

Import `source_schedule` from `budget.py`, validate actual 32 vs40
and exact cut boundaries:
```python
from bela_next_budget.budget import source_schedule
schedule=source_schedule(arm)
assert schedule.w==arm["budget_T"]*arm["layers"]
# For example, next13/T160:
# after L15 -> 80 patches; after L30 -> 40 patches;
# first 15 blocks see K0=320, next 15 K1=80, last10 K2=40.
```

Never switch to "initial/final T160" semantics.
Compute same-layer Q/K AFTER RoPE, with correct head GQA
(if present), text query IDs/vision patch keys and causal mask;
evaluate Q attention over full current key set, softmax then
filter visual keys, no hidden-state-dot fallback.

Call the safe index planner:
```python
from bela_next_budget.preserve import plan_physical_keep
plan=plan_physical_keep(
    sequence_length=int(hidden_states.shape[-2]),
    patch_positions=live_patch_seq_positions,  # ONLY patch tokens
    selected_patch_relative=selected_from_bela_QATTN,
    original_patch_ids=original_patch_ids,
    crop_ids=current_crop_ids,
)
# Pass full_keep_indices into the EXISTING validated physical prune
# implementation and gather all dependent states / cached K/V / masks.
# It only returns the selection plan; it cannot manage RoPE/position/cache.
```

If preserving newline tokens across all stages increases the REAL
sequence budget, do NOT claim same full-model FLOPs as patch-only W.
Record patch_W, newline_W, text_W, total actual sequence lengths per layer.
If the original model has special tokens that may be dropped safely,
run a completely separate "special token policy" ablation with independent
Dense/parity. Never silently drop them for the BELA main comparison.

## 4. Correctness gates before accuracy
P0a: Dense with no-prune feature-flagged forward vs original Dense
same input images/preprocess identical logits/generation within numerical
FP16 tolerance. Save max abs, logit top1 agreement and exact prompt ids.
P0b: Fixed image NeXT input: 5*576 patch IDs, per crop total
K, newline count/positions, monotonic nested selection Stage1 ⊇L12⊇L20
or L15⊇L30, materialized exact W per layer.
P0c: independent same-layer after-RoPE Q/K vs Q_Attn ranking, correct
text-query selection, GQA, attention masks, padding & merged cropping.
P0d: causal position, cache and first decode generated tokens correct;
verify no source patch from a deleted token sneaks into a future cache.
P0e: 16 distinct real examples across image aspect ratios / OCR and
charts; verify correct crop mapping and QA source IDs.
Any gate failure → no model-accuracy report, return smallest reproducer.

## 5. What changes when model scale grows
- NeXT7B: 32 decoder layers, cuts after L12/L20.
- NeXT13B: 40 layers, cuts after L15/L30.
- NeXT language family in paper is **Vicuna**, not Video-Qwen.
- Stage1 K0=320/640 may be > CLIP per-crop 576 only if grouped
  incorrectly. Group quotas remain ≤576.
- All script flags in `print_arm_env.py` are NEW conventions; they
  have NO effect until actual live source is patched.
