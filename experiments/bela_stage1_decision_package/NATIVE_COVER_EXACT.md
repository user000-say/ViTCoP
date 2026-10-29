# NATIVE_COVER_EXACT — production source audit (P0)
**Date:** 2026-10-29  
**Host:** V100 node, repo `/home/xiangyue/ViTCoP`, branch `env/v100-4model-qacover` (working tree also holds prior LLaVA-1.5 arms under `logs/`).  
**Frozen schedule claim:** L2 `64×12→17×8→10×12` W1024; Dual COVER → optional QA-COVER B8 → RDRIFT.

## Checkpoint / runtime

| Item | Value |
|---|---|
| Model | `liuhaotian/llava-v1.5-7b` (metrics `pretrained` field) |
| Eval profile | `e014_boot.py --profile ANCHOR_A --layer 8` |
| Python env | `/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python` |
| Attention / dtype | production E014 path (SDPA/eager as configured by arm overlay; not re-probed this turn) |

## Exact call chain

1. `LLaVA/llava/model/multimodal_encoder/clip_encoder.py` (~1263): when `ELVA_R3_ENABLE=True`, calls `llava_stage1.select_existing_forward`.
2. `experiments/elva_r3/llava_stage1.py:select_existing_forward`:
   - `pen = hidden_states[-2][0,1:]` must equal `tokens_0` (penultimate payload).
   - Early view: `hidden_states[9][0,1:]` when Dual/`l9` needs it (0-based layer index **9** = Layer9).
   - Scores: `_sum_cls(attentions, -2)` and optionally layer 8 attentions for L9.
3. Native COVER: `cover_family=legacy_native` → `NativeCover(legacy_callback)` in `experiments/elva_r3/native_cover.py`.
4. `legacy_callback` → `e013_choose(..., mode="REPLACE", family="COVER", slots=32, core_k=K)` with `K∈{32,64,72,80,96,128,160,172}`; for K64 slots **must** be 32.
5. Output: `keep` = K unique **original patch IDs** (local indices into the 576 grid), sorted later by QA path.
6. Dual metric tensor `selected["metric"]` shape `[576, D]` — full-grid descriptor used by QA and QCP; not a fused scalar view name alone.
7. Stage2: unchanged RDRIFT/Q_Attn via E018 events from arm overlay; L2 documented as cuts yielding `11:17,19:10` with W=`64*12+17*8+10*12=1024` (`LIVE_CALLMAP.md`).

## SHA256 (files read for this audit)

```
native_cover.py     e2f2fe19a252d07a0dba108b27844443a169e12540439d231d72b9d59761a224
llava_stage1.py     274e07899f372b66b052424644abc0054aaced0efd4ccd798a32aff237561f13
query_anchor.py     78ee057e595f3f14e87cd737a055559cf639dc892222e9cf035433c4d2453581
clip_encoder.py     4d4b074fa11a778b956e3e4dd5fa937a14c8091188ff4b76a8b63ac1c64c15a7
```

## What is NOT claimed here

- Byte-identity of `e013_choose` COVER objective to any external paper formula.
- Fresh GPU re-measurement of L2/QA in this P0 document (historical metrics already on disk under `logs/e014_aee/full/bela_stage1_queryanchor_20261027` and `bela_final_p1_20261025/L2_LATE_64_17_10`).
