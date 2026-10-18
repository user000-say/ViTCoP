# E020.3 Incident Log

Auto-recorded pipeline failures during wiring / ACC / TTFT.

Format: timestamp · symptom · root cause · impact · fix · status.

## INC-001 — fp16 index_add dtype mismatch in recycle

- **Timestamp:** 2026-10-18T19:25+08:00
- **Symptom:** `index_add_(): self (Half) and source (Float)` during B merge smoke.
- **Root cause:** `torch.zeros(...)` for mass/total followed ambient default dtype (fp16) while donors were cast to float32.
- **Impact:** Blocked merge arms only; A off-path (alpha=0) unaffected.
- **Fix:** Explicit `dtype=torch.float32` accumulators in `torch_backend.recycle`.
- **Status:** Fixed; unit tests + B smoke green.

## INC-002 — inject used RoPE visual_positions as row indices

- **Timestamp:** 2026-10-18T19:26+08:00
- **Symptom:** CUDA device-side assert in `inject_visual` / `index_copy_`.
- **Root cause:** `s.visual_positions` are original RoPE ids; after gather, recycled rows must write to compacted `[visual_start, visual_start+K)`.
- **Impact:** Merge path only.
- **Fix:** Inject via `arange(visual_start, visual_start+K)` after `_drop_ids`.
- **Status:** Fixed; B merge GQA limit=2 smoke OK.
