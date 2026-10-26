# CODE_CALL_MAP — live host (independent handoff P0)

## BELA / E3 three cuts (0-based post-block → 1-based after-layer)
E3 stages `[4,172]+[8,120]+[10,24]+[10,16]` → `E018_EVENTS=3:120,11:24,21:16`
(= after L4→120, L12→24, L22→16).

| Cut | after layer | Scorer | Callsite |
|---|---|---|---|
| 1 | 4 | Q_ATTN + **E020 RDRIFT refine (first only)** | `llava/model/e018_stage2_support.py` `_pick` (~390) → `e020_lib/dispatch.refine` |
| 2 | 12 | Q_ATTN | same `_pick` (no E020) |
| 3 | 22 | Q_ATTN | same `_pick` (no E020) |
| Physical gather | each cut | `_drop_ids` hidden + layer_local KV | `_event` → `_drop_ids` |

Stage1 COVER: `experiments/elva_r3/llava_stage1.py` → `native_cover.legacy_callback` → `e013_choose(COVER)`.

## L2 / F1024_BB two cuts
`64×12+17×8+10×12` → events `11:17,19:10` (after L12, L20). RDRIFT on first only.

## E5 / F2048_BB two cuts
`128×12+37×8+18×12` → events `11:37,19:18`.

## STAR-Pro QS (official tree)
| Stage | Path |
|---|---|
| Stage1 QR | `third_party/LLaVA-starpro` / star modelling (`STAGE1_SCORER=qr`, pool=2T) |
| Stage2 | progressive topk `[(12,17),(20,10)]` for T=32; `[(12,37),(20,18)]` for T=64 |
| Runner | `experiments/starpro_x0_t64/run_one_task.py` + env `/home/xiangyue/envs/starpro` |

## Unverified
`RDRIFT at both cuts` — **false** on current host; RDRIFT is first-event only.
