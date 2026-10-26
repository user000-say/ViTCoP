# F2048_QS / F1024_BB / F2048_BB — reuse map

## F2048_QS = STAR-Pro T64 (already on disk)
Path: `logs/starpro_x0_t64/full/STARPRO_T64/`
Schedule: Stage1 pool 128 → (12,37),(20,18); W=2048.

| Task | Abs | Rel% |
|---|---:|---:|
| GQA | 58.42 | 94.38 |
| POPE F1 | 87.26 | 101.70 |
| MME Total | 1707.9 | 92.22 |
| MMBench | 61.68 | 96.22 |
| SciQA | 64.50 | 92.79 |
| TextVQA-OCR | 49.93 | 85.79 |
| **R4 / mean6** | | **96.13 / 93.85** |

vs BELA E5 (F2048_BB shape): E5 mean6=96.08 GQA57.88 POPE84.96 — STAR higher GQA/POPE, lower MME/Sci/TVQA → mean6 lower.

## F1024_BB = L2 (reported)
`logs/e014_aee/full/bela_final_p1_20261025/L2_LATE_64_17_10` mean6=93.56

## F2048_BB = E5 (reported)
`logs/e014_aee/full/bela_final_p1_20261025/E5_DELAY_128_12_20` mean6=96.08

## F1024_QS
Launching STAR T32 → `logs/starpro_x0_t64/full/STARPRO_T32/`
