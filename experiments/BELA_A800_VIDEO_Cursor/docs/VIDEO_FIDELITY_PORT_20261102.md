# Video fidelity port from NeXT image standards — 2026-11-02

## Verdict

Same fidelity checklist that lifted NeXT-7B maps to video with **three concrete patches** (applied in `bela_video_a800/` + local `LLaVA-NeXT` hooks). Dual-COVER remains a **SigLIP adaptation**, not image Dual.

## Checklist → video status

| Standard | Image NeXT fix | Video before | Video after |
|---|---|---|---|
| Dense-absolute RoPE after physical cut | gather `position_ids`; decode `rope_next=dense_len` | Stage2 **recompacted** to `0..L-1` | gather original ids; rotary returns full cos/sin table; decode uses `_bela_rope_next_pos` |
| Mid layer = Layer9 → `hs[10]` | CLIP `mid_layer=10` | SigLIP already `mid_idx=1+9` | unchanged (already correct) |
| Stage2 Q_ATTN = suffix text | orphan-nl / abs-pos filter | `visual_end:` suffix | unchanged (already correct) |
| Dual COVER k≈N / empty residual | clamp + top-up | could `RuntimeError` unique≠K | clamp + score top-up |
| Stage1 dense-absolute at LLM entry | gather from dense embeds | feature-level gather then pack short seq | **deferred** (larger arch change; Stage2 fix is the cross-layer RoPE bug) |
| No-op Dense vs BELA_NOP | separate | `mode=off\|dense` | unchanged |

## Host wiring (V100 paths)

- `llava_arch.py`: Stage1 hook + Stage2 visual_start/ctx publish → `/home/xiangyue/ViTCoP-a800-video/experiments/BELA_A800_VIDEO_Cursor`
- `llava_qwen.py`: generate mask rewrite uses **physical** `_bela_prefill_seq_len` + **absolute** `_bela_rope_next_pos`

## Test plan

1. Unit: COVER clamp + position gather subset (PASS locally).
2. `scripts/v100_fidelity_smoke.py` Dual+Q_ATTN T32, fp16, 8 frames, GPU ∈ {0,2,4,7}.
3. Limited Video-MME (`--limit 32`) then full if smoke green — do not claim A800 R1 numbers from V100.

## Constraints

- V100: **fp16 only** (no bf16).
- Do not kill foreign jobs; NeXT v5 holds 4/7 while other users may hold 0/2.
- Full Video-MME / LVB ACC historically A800; V100 = fidelity bring-up + limited ACC.
