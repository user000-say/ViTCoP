# NeXT-7B 94.4% A0–A4 Preflight — 2026-11-01

## Verdict: **READY for 6-task ACC** (smoke PASS)
Previous tag `PRUNING_NOT_ACTIVE` cleared for the new overlay path.

## Smoke (GPU 4, 2026-11-01)
| Check | Result |
|---|---|
| Dense parity (`BELA_NEXT_V2_ENABLE=0`) | PASS |
| A0 320→91→46 | PASS |
| A1 320→100→40 | PASS |
| A2 320→82→52 | PASS |
| A3 288→136→48 | PASS |
| A4 256→160→64 | PASS |
| generate after prune | PASS all five |

Evidence: `logs/smoke_a0_a4/SMOKE_REPORT.json`, `SMOKE_VERDICT.md`.

## Forward wiring (real)
| Component | Path |
|---|---|
| Dual COVER (e013 REPLACE/COVER, crop-wise) | `bela_next_forward/dual_cover.py` |
| Stage2 Q_ATTN physical prune @L12/L20 | `bela_next_forward/stage2.py` |
| HF LlavaNext overlay (TF 5.20) | `bela_next_forward/apply_llava_next.py` |
| lmms_eval entry | `scripts/lmms_bela_next_entry.py` |

Stage1: simple_crop_concat of Dual COVER survivors (K0 patches). Newlines excluded from patch budget. Placeholders rewritten to match K0. Stage2 physically shortens hidden + DynamicCache.

## GPUs
0–3 foreign — do not touch.
4–7 free — ACC queue target.

## Launch
`scripts/run_next7_a0_a4_six.sh` → 5 arms × 6 tasks on GPUs 4/5/6/7.
