# NeXT high-res ACC drop — engineering root cause (2026-11-02)

## Why higher-res + more tokens still lost ACC

Not a paradox: Dual COVER / QA on NeXT were **not** the same operators as 1.5-7B.

| Bug | Effect | Fix |
| --- | --- | --- |
| `original_dual` applied `slots=32` **per crop** | 5 crops → ~160 COVER / 320 keep (~50% novelty); 1.5 uses 32/K once | distribute global 32 slots across crop quotas |
| `mid_layer=9` | CLIP `hs[9]` = Layer8, not Dual L9 (`hs[10]`) | `mid_layer=10` |
| `_question_ids_from_input` kept USER/ASSISTANT | QA cosine flat → `qa_swaps=0` always | strip template; question-only ids |

## Smoke proof (`logs/smoke_fix_slots_mid_qa/A0_SMOKE.json`)

- `dual_slots_per_crop = [6,6,6,7,7]` (sum 32)
- `mid_layer = 10`
- `qa_swaps = 8`, `qa_reason = swapped`
- `question_n_tokens = 13`

## Retest

`scripts/run_next7_engfix_gated.sh` → `logs/acc_a0_a4_engfix_20261102/` A0 R4 first (GPUs 4–7).
Prior v4 A0 R4≈90.25 is the comparison baseline for lift.

## Follow-up from NeXT ACC-drop audit (Stage2 contract)

Audit: highest-ROI bug is Stage2 scoring post-layer states via a second Q/K probe.
Production Dual L9 is `hidden_states[9]` — reverted mid_layer 10→9.
QA now slices original USER…ASSISTANT token ids (no decode/re-encode).
Kept global slots redistribution (COVER inflation fix).

Retest: `run_next7_s2qk_gated.sh` → `logs/acc_a0_a4_s2qk_20261102/`.
