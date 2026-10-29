# LIVE_VIDEO_CALLMAP — LLaVA-Video-7B-Qwen2 on A800 (2026-10-29)

Live checkout: `/data1/xiangyue/src/LLaVA-NeXT`  
Checkpoint: `/data1/xiangyue/models/LLaVA-Video-7B-Qwen2`  
Eval overwrite: `mm_spatial_pool_mode=average`, `mm_spatial_pool_stride=2`, `attn=sdpa`, `dtype=bfloat16`.

## Observed shapes (Dense smoke)

| Stage | Shape |
|---|---|
| pixels | `[F,3,384,384]` |
| SigLIP tower | `[F,729,1152]` (27×27) |
| after `mm_projector` | `[F,729,3584]` |
| after `get_2dPool` avg stride2 | `[F,169,3584]` |
| decoder layers | 28 |

## Official packing path (`prepare_inputs_labels_for_multimodal`)

1. `encode_images` = vision_tower → mm_projector (no pool yet).
2. For `modalities[i]=="video"`: `get_2dPool(feat)` → `[F,169,H]`.
3. Checkpoint `mm_newline_position=grid` → `add_token_per_grid` → ~`F*182` tokens (13×14 with per-row newline). Structural newlines are **not** visual candidates.
4. Question / time instruction tokens stay outside the visual block.

## Stage1 hook (this delivery)

- File: `llava/model/llava_arch.py` method `_bela_video_stage1_maybe_flat`.
- Env `BELA_VIDEO_MODE=off|dense` → **None**, official grid path unchanged.
- Env `native|qacover|matched_random` → gather K patches/frame from pooled `[F,169,H]`, then `add_token_per_frame` (one newline/frame), flatten, skip grid packing.
- K from `BELA_VIDEO_TIER=T32|T16` → 64 or 32 entry tokens/frame.
- Native selector is **NATIVE_L2_TOPK** (not Dual COVER). QA-COVER residual uses frozen `select_framewise` when mode=`qacover`.
- Stage2 RDRIFT / cuts after layers 12 and 21: **not hooked**. Do not report these runs as full BELA Video-T32/T16.

## ID contract

- Visual selection unit: `(frame_idx, pooled_patch_idx 0..168)`.
- Flat visual-only id = `frame*169 + patch` (audit only).
- Do not treat flattened LLM positions as patch ids (timestamps / newlines / text intervene).
