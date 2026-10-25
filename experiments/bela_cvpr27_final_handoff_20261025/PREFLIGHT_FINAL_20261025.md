# Preflight — BELA_FINAL_CURSOR_HANDOFF 2026-10-25

## Inventory
| Package provides | Repo already owns |
|---|---|
| Research freeze + P1 schedules (E1–E5, L2–L3) | E014/E018 COVER+RDRIFT host, `apply_llava.e018_events` |
| CPU schedule/ledger/results tools + unit tests | Official scorer via `e014_boot` / lmms-eval |
| CUT_CALL_MAP / OUTPUT contracts (templates) | fair-256 TTFT bench, gpu_exclusive 0/2/4/7 |
| **No** new prune kernel / Guard | Wave-1 full6 ACC done (15/15) |

## Gap check (resolved before launch)
1. **L2 K0=64 / L3 K0=80 blocked** by legacy COVER allowlist `{96,128,160,172}`.
   - Fix (capacity only, same REPLACE/COVER formula): expand allowlist to `{64,72,80,96,128,160,172}` in `e013_stage1_support.py`, `native_cover.py`, `cover_binding.py`, `llava_stage1.py`.
   - **Not** a new Stage1 algorithm; labeled capacity expansion for FINAL late-first-cut.
2. **3-cut schedules (E1/E2/E3)**: E018 already allows ≤3 events; `e018_events` maps variable `lengths/counts`. No Stage2 kernel edit.
3. **GATED** selector/cross arms: not launched (handoff: after P1 + call-map).
4. **Evidence Guard / retirement_probe**: absent from zip; not implemented (handoff: only after paired ID-error proof).

## Algorithm identity
| Arm | Operator | Identity |
|---|---|---|
| E1/E2/E3 | COVER Stage1 + RDRIFT@first + Q_ATTN later; 3 physical cuts | Original host; new **schedule only** |
| E5 | Same selector; STAR-Pro **shape** `128×12+37×8+18×12` | Schedule shape diagnostic; **not** STAR-Pro code |
| L2/L3 | Same selector; delayed first cut | Schedule + COVER@K64/80 capacity |
| S_1024_128_24_16 / S_1024_160_28_8 | Early-cut higher entry @W1024 | ACC-boost schedule probe (Wave1 never tried K128/160 @W1024) |

## Prior evidence (reuse, do not regen)
- A0/A2/A3/B1/B2 ACC+TTFT; Wave1 best W2048 `172/125/18` mean6=96.74; W1024 best `96/60/8` mean6=93.24.
- User targets: W1024 ~94–95, W2048 ~97–98 relative — gaps ≈+1–2 / +0.3–1.3 vs current bests; P1 tests late-retirement / late-first-cut hypotheses.

## Verdict
**READY** for P1 full-6 ACC (+ACC-boost W1024 entry probes) on GPUs 0/2/4/7 after smoke (limit=2) confirms entry K, event layers, and final K for E1 and L2.
