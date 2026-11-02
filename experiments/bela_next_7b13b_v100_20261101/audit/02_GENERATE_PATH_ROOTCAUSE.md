# Root cause: ACC did not prune (2026-11-02)

## Symptom
NeXT-7B A0–A4 full ACC produced **byte-identical** predictions; scores ≈ Dense.

## Root cause (Transformers 5.20 + our hook)

1. `GenerationMixin._prepare_multimodal_encoder_kwargs_for_generation` calls
   `base_model.get_image_features`, stores **stock dense** packs in
   `mm_encoder_outputs["image"]`, then **pops `pixel_values`**.

2. Our old `hooked_llava_forward` treated `pixel_values is None` as
   “decode / skip BELA” and called `orig_llava_fwd` → **Dense scatter + Dense LM**.

3. Smoke used `model(**inputs)` (pixels present) so Stage1/Stage2 traces looked
   correct; smoke `generate()` was only checked for “does not crash”, not for
   arm differentiation. ACC uses `model.generate()` → never pruned.

## Fix
1. Patch `get_image_features` → Dual COVER Stage1 pack (K0).
2. Prefill with `mm_encoder_outputs` **or** `pixel_values` runs placeholder
   rewrite + Stage2 (fail closed if metadata missing).
3. Replace `generate` when ENABLE=1: BELA prefill via `forward(pixel_values=...)`,
   greedy decode on pruned KV, return `cat([orig_input_ids, new_tokens])` so
   lmms `out[:, dense_len:]` still works.

## Proof
`scripts/diag_generate_path_prune.py` → `DIAG_PASS=True`:
- A0 `320→91→46`, A4 `256→160→64`
- `prefill_via=pixel_values`, `a0_eq_a4=false`, both ≠ Dense
