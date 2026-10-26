# Preflight — BELA_FINAL_HANDOFF_20261026

## Verdict: READY (partial five-arm; A-module candidate launched)

| Item | Status |
|------|--------|
| Offline `check_plan` + 27 unittest | PASS |
| CD K32 six-task recovered | mean6=**94.41** (reuse) |
| L2_NATIVE | reuse mean6=**93.56** |
| COVER32_FLAT | **BLOCKED** — e013/COVER allowlist K∉{32} |
| CD64_L2_NATIVE/STATIC | **BLOCKED** — no clean CD-ID→BELA transplant yet |
| E3 Dual vs L9 pairing | DONE (GQA complementary headroom) |
| L2_ADMIT_B2 (handoff A, b=2) | **LAUNCH** 6-task |
| L2_L9_ONLY control | **LAUNCH** 6-task |

## Algorithm identity

- **L2_ADMIT_B2**: original Dual COVER@64 + RDRIFT/Q_ATTN L2 schedule; **new** bounded admission swaps ≤2 IDs using L9–late disagreement (METHOD §3). Default E3 unchanged.
- Not permanent object-track / trait-rail (forbidden by handoff).

## Dual∪L9 evidence (E3 W2048, full GQA)

- dual_only=472, l9_only=587, oracle_union ACC=62.43 vs Dual 57.76 → answer-level complementarity supports A.

## W1024 → ~95 path (honest)

- CD32 already 94.41; L2 93.56. Need ~+0.6–1.4 mean6.
- Admission b=2 is small; if L2_ADMIT/L2_L9 stall below 95, next is CD64_L2 transplant (not renamed CD).
