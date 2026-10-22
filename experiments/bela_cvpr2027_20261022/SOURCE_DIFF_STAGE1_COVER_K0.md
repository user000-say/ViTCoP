# SOURCE DIFF — Stage1 COVER accepts budget=K0 (handoff §P3)

**Date:** 2026-10-22  
**Intent:** match `02_CURSOR_HANDOFF_CN.md` Stage1 req §1 — real dual COVER call accepts `budget=K0` without changing feature definition or COVER gain. **Not** PCOV / COVER_PORT / pool172.

## What changed

| File | Change |
|---|---|
| `LLaVA/.../e013_stage1_support.py` | `e013_choose(..., core_k=None)`; default `None→172`; allow `{96,128,160,172}` |
| `experiments/elva_r3/native_cover.py` | Pass `core_k=k`; slots fixed at **32** |
| `experiments/elva_r3/elva/cover_binding.py` | `legacy_native` may declare those K0 values |
| `experiments/elva_r3/llava_stage1.py` | `supported_k=[96,128,160,172]`, slots 32 each |

## Binding contract

- Entry K comes from `ELVA_R3_CONFIG` → `schedule.counts[0]` → `ELVA_R3_ENTRY_K` (existing R3 overlay). No E014 `STAGE1_K=172` helper for these arms.
- COVER family remains `legacy_native` (Layer9+pen dual path already in `llava_stage1.select_existing_forward`).
- `BELA_NEXT_ENABLE` stays **False**; G1 is static fixed K0 via config, not QuantilePolicy.

## G1 configs (matrix)

- **F2** `dual_COVER_K160` + RDRIFT: counts `160/96/32`, lengths `4/8/20`, W=2048  
- **F3** `dual_COVER_K128` + RDRIFT: counts `128/64/32`, lengths `8/8/16`, W=2048  

## Explicitly NOT this patch

- Adaptive QuantilePolicy / shuffled controls (G2) — later.  
- `greedy_order` exposure for residual-difficulty (Stage1 req §2) — later; G1 does not need it.  
- PCOV60/K80 low-budget family — different algorithm, W1024 era.
