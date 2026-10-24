# Preflight — BELA_CVPR27_ONE_DAY 136-arm handoff (2026-10-24)

## User override vs handoff CSV pilot
Handoff P1 suggests fixed **subset** pilots. User forbids subset sampling (high noise).  
**Judgment unit = full 6-task ACC** (GQA/POPE/MME/MMB/SciQA/TextVQA). Progressive selection still applies across *arms*, not within a dataset.

## Queue inventory
| phase | n | action |
|---|---:|---|
| REUSE | 5 | A0/A2/A3/B1/B2 — **skip** (already MEASURED) |
| PILOT | 95 | schedule sweep; only COVER-legal entry K |
| GATED | 36 | selector probes — **BLOCKED** until cut2 audit |

COVER-legal entry K: `{96,128,160,172}`. K60 needs PCOV (separate). K192/224+/256 BLOCKED on legacy COVER.

Eligible now: **W2048 ×24**, **W1024 ×8** (=96 entry). W4096 deferred.

## Host fix (required)
ELVA final-K allowlist was `{12,16,20,24,32}` → A25 kdeep=18 crashed. Expanded to `8..64` when `ELVA_R3_FINAL_K` matches.

## Paper-informed progressive Wave-1 (not random 96)
From STAR-Pro / QCPruner / CoRePrune / our B1/B2:
- Keep Stage1 COVER; do not swap selector yet (GATED blocked).
- Prefer mid↑ for POPE, deep↑ for GQA; search around A2 and STAR-like entry128.
- Exact W only; full 6 ACC; then TTFT for promoted arms.

## Wave-1 arms (8 × W2048 + remaining W1024@96)
See `scripts/launch_136_wave1_full6.py`. GPUs: free cards from `{0,2,4,7}` via `gpu_exclusive` (GPU0 may be foreign-occupied — skip if busy).

## Verdict
**READY** for Wave-1 full-6. GATED/W4096/K60-PCOV not in this wave.
