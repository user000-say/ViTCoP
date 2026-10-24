# Preflight — BELA_ELVA_CURSOR_NEXT_HANDOFF_20261024

## Verdict: READY (2 new arms) / SKIP (done) / BLOCKED (W4096)

| Item | Status |
|---|---|
| P0 B1 GQA + B1/B2 TTFT | **DONE** overnight (skip regen) |
| A0/A2/A3/B2 6-task ACC + A* TTFT | **DONE** (skip) |
| W1024 historical PCOV60 60/28/12 | **DONE** in `bela_gate1_8task/W1024_V4_PCOV_QATTN` (skip) |
| **A25** 172/125/18 W2048 COVER+RDRIFT | **READY** — launch 6 ACC + TTFT |
| **W1024_probe** 96/50/12 COVER+RDRIFT | **READY** — launch 6 ACC + TTFT |
| W4096 256/184/80 | **BLOCKED** — legacy COVER K0∉{96,128,160,172} |

## Algorithm identity
Both new arms: **original** host COVER Stage1 + RDRIFT(first) + Q_ATTN(later); schedule-only.
