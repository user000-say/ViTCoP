# Runtime and source audit — hard blockers
Current production BELA source is NOT available in this ChatGPT runtime. Cursor must NOT assume historical E014 paths remain authoritative.

## Required source call map (CUT_CALL_MAP.md)
For each of first and second (and proposed third) cut, record:
- exact file:function:line, event after which 1-based decoder layer, original candidate K, outgoing K;
- whether uses RDRIFT, QAttn, original visual attn, selection from a precomputed cache, or another scorer;
- timestamp / layer at which attention or hidden signal is available (no future-layer leakage);
- surviving ORIGINAL patch IDs; mapping from local candidate IDs;
- real gather on hidden states, attention mask, position ids and past_key_values;
- prefill-only action vs cached decode, physical vs masked deletion;
- whether prompt and all query tokens are correctly selected and attention is comparable when K changes.
Provide a small call-stack/print trace plus SHA256 of modified files.

## Required runtime JSONL
One record PER image-question; selected prompt tokens remain unchanged:
{
  "sample_id": "...",
  "image_id": "...",
  "run_id": "...",
  "active_visual_tokens_by_decoder_layer": [32 integers],
  "prune_events": [
    {
      "after_layer": 4,
      "original_patch_ids_before": [...],
      "original_patch_ids_after": [...],
      "score_function_path": "model.py:fn",
      "callsite": "model.py:line",
      "qattn_scores_file": "optional path to cached score vector",
      "kv_cache_checked": true,
      "mask_checked": true
    }
  ]
}
`tools.schedule.validate_trace` checks monotonic count/ID mapping and source provenance for a supplied named schedule; it cannot prove GPU KV correctness by itself.

## Mandatory SMOKE before new production code
1. Three cases: minimal prompt, many question tokens, and image with several small objects/text.
2. Compare default-OFF mode against unmodified baseline predictions and shape/KV trace; no change expected.
3. Test no-removal and aggressive low K boundary; every chosen ID must be unique, present in earlier set, and physical hidden sequence must match.
4. Test `use_cache` true and false, short-output decoding, batch-size one; if code only supports one, report explicitly.
5. Deterministic tie handling for scores; no accidental new random behavior.
6. If cut3 inserted, ensure score after layer20 is generated before first layer21 reads the new set; any inter-layer cache aliases must be adjusted.
7. No `.item()` per token, CPU/GPU round trips, per-patch Python loops in hot inference code. Run timed profiling with selector enabled/disabled.

## Run metadata
`git_sha`, `model_sha`, `dataset_manifest_sha`, `evaluator_sha`, `CudaDevice`, `seed`,
`batch_size`, `dtype`, `prompt_template`, `generated_tokens`, `W_expected`, `W_observed`, `elapsed_ms`, `cut_event_counts`, `status`.
Raw predictions MUST use official benchmark scripts and same parser, no separate ad hoc accuracy metric.
