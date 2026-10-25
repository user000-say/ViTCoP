# BASELINE_LOCK — 2026-10-26

| Field | Value |
|---|---|
| git SHA (pre-expand parent) | `da71409f082c1b1e388c2bc32c424cf848075a23` |
| Host | E014 arm `E018_COVER_TWO32` + ELVA_R3 overlay COVER+RDRIFT |
| Model | LLaVA-1.5-7B (fair-256 / e014 local tree) |
| Evaluator | `e014_boot` → lmms-eval official task scripts |
| GPUs | physical **0, 2, 4, 7** via `gpu_exclusive` |
| Handoff | `BELA_ELVA_CVPR2027_FINAL_CURSOR_HANDOFF_20261025.zip` → `experiments/bela_cvpr27_final_handoff_20261025/` |
| Unit tests | `python -m unittest discover -s tests -v` → 23 OK |
| Dirty intentional | COVER K0 allowlist expand `{64,72,80}` for L2/L3 (same e013 COVER) |

## Locked references (do not regen)
| Arm | Notes |
|---|---|
| A2 172/120/20 W2048 | GQA57.62 POPE85.44 MME1774.9 MMB63.14; TTFT122.5 / F32 1217.7 |
| Wave1 best W2048 | `S_2048_172_125_18` mean6=96.74 |
| Wave1 best W1024 | `S_1024_96_60_8` mean6=93.24 |
| B1 GQA | 57.03 (complete); TTFT119.6 / F32 1178.9 |
| B2 | TTFT120.5 / F32 1177.6 (non-iso W2288) |
