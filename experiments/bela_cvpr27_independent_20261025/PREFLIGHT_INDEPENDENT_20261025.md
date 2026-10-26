# Preflight — BELA_CVPR27_INDEPENDENT_CURSOR (2026-10-25)

## Inventory
| Package | Repo owns |
|---|---|
| 2×2 factorial matrix + budget/factorial tools | E014/E018 COVER+RDRIFT; STAR-Pro isolated env + T64 R9 done |
| Day plan: QS repro → 2×2 → Stage1 ablations | fair-256 TTFT; gpu_exclusive 0/2/4/7 |
| **No** hybrid QR↔BELA transplant code | |

## Gap / gate
| Item | Verdict |
|---|---|
| F2048_QS (STAR T64) | **REUSE** existing `logs/starpro_x0_t64/full/STARPRO_T64` (9 tasks) |
| F1024_QS (STAR T32) | **LAUNCH** — schedule `64×12+17×8+10×12` via `visual_token_num=32` |
| F*_BB | **REUSE** L2 / E5 (same schedules) as B/B |
| F*_QB / F*_BS | **BLOCKED** until clean Patch-ID + attention/KV transplant (SOURCE_AUDIT). Do not invent scores. Use full-system QS + native Stage1/S2 ablations instead. |
| E3 Stage1 pen/L9 | Runnable after QS T32 queued (native `guide=pen` / `guide=l9`) |
| Prior 2-stage 48-task | **PAUSED** (see sibling package `PAUSED_2STAGE_FOR_INDEPENDENT.md`) |

## Algorithm identity
- QS = official STAR-Pro (`METHOD=star_pro`, QR Stage1, topk Stage2) in `/home/xiangyue/envs/starpro` + `third_party/LLaVA-starpro`.
- BB = BELA COVER+RDRIFT on matched STAR schedule shapes (L2 / E5).
- Hybrids are diagnostics only if/when source contract passes — not paper method.

## Verdict
**READY** for P0: audit artifacts + F1024_QS full-6 ACC on GPUs 0/2/4/7 (smoke T32 first). Hybrids remain gated.
