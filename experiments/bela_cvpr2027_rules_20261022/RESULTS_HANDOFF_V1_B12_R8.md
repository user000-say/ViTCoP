# Handoff V1 — B1/B2 mechanism ACC (8-task, no VizWiz) + TTFT

Date: 2026-10-23. Scorer: official lmms metrics via `e014_boot` / Dense refs frozen in `elva_scores` protocol.  
rel = 100 × abs / Dense. **mean8** = mean of 8 relatives (VizWiz excluded). **R4** = GQA/POPE/MME/MMB.

Dense: GQA 61.9, POPE 85.8, MME 1852, MMB 64.1, SciQA 69.51, MMMU 35.60, AI2D 55.18, TextVQA-OCR 58.20.

Arms (COVER + RDRIFT first / Q_ATTN later):
| Arm | schedule | W | role |
|---|---|---:|---|
| A0 | 172×4 + 90×8 + 32×20 | 2048 | isobudget anchor |
| A2 | 172×4 + 120×8 + 20×20 | 2048 | main isobudget baseline |
| A3 | 172×4 + 130×8 + 16×20 | 2048 | deeper cut (T1) |
| B1 | 172×4 + 90×8 + 20×20 | **1808** | non-iso: A0 mid, A2 deep |
| B2 | 172×4 + 120×8 + 32×20 | **2288** | non-iso: A2 mid, A0 deep |

## Absolute / relative (8-task)

| Arm | GQA | POPE F1 | MME | MMB | SciQA | MMMU | AI2D | TextVQA | R4 | mean8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 57.84 / 93.44 | 83.31 / 97.10 | 1767.83 / 95.46 | 62.20 / 97.03 | 68.96 / 99.21 | 36.00 / 101.12 | 54.79 / 99.30 | 55.97 / 96.16 | **95.76** | **97.35** |
| A2 | 57.62 / 93.08 | 85.44 / 99.58 | 1774.85 / 95.83 | 63.14 / 98.51 | 68.67 / 98.79 | 35.80 / 100.56 | 54.50 / 98.77 | 54.89 / 94.32 | **96.75** | **97.43** |
| A3 | 57.35 / 92.66 | 85.68 / 99.86 | 1775.86 / 95.89 | 62.80 / 97.97 | 68.67 / 98.79 | 36.00 / 101.12 | 54.57 / 98.89 | 54.88 / 94.30 | **96.59** | **97.43** |
| B1 | *rerun* | 83.45 / 97.26 | 1747.84 / 94.38 | 62.37 / 97.30 | 68.86 / 99.07 | 35.90 / 100.84 | 54.44 / 98.65 | 54.51 / 93.66 | — | — |
| B2 | **58.44 / 94.40** | 85.36 / 99.49 | **1804.20 / 97.42** | 62.80 / 97.97 | 68.37 / 98.36 | 35.30 / 99.16 | **54.92 / 99.53** | 55.13 / 94.73 | **97.32** | **97.63** |

## TTFT / Fixed32 (fair-256, n=1024)

| Arm | TTFT ms | Fixed32 ms |
|---|---:|---:|
| A0 | 123.7 | 1234.6 |
| A2 | 122.5 | 1217.7 |
| A3 | 120.3 | 1177.1 |
| B1 | pending exclusive | pending |
| B2 | pending exclusive | pending |

## Mechanism read (handoff P1)

1. **B1 ≈ A0 on POPE (83.45)** — cutting deep alone does **not** reproduce A2’s +2 POPE; mid 90→120 is the main POPE lever.
2. **B2 recovers GQA (58.44 > A0/A2)** and lifts MME (1804) while keeping POPE ≈ A2 — extra deep capacity helps GQA/MME under A2’s mid schedule (non-isobudget; not a fair W ranking).
3. TextVQA: A0 55.97 > B2 55.13 > A2 54.89 — B2 partially closes the OCR gap vs A2, still below A0.
4. Latency: lower final K slightly lowers Fixed32 (A3 best among measured); still ~1180–1235 floor on this host.

## Open

- B1 GQA refill + B1/B2 TTFT (exclusive `gpu_exclusive` leases).
- Paired ID / POPE subset diag still recommended before GO_C1 vs GO_IDENTITY.
