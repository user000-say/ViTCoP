# E013-P1 Final Summary (2026-10-10)

**Status:** COMPLETE (smoke / panel / full×10 arms / ABBA@15% / Stage1 latency / bridge12).  
**Locks:** ACC = official full N only. Latency gate = |rel Fixed32| ≤ 15%; headline HOLD until ABBA narrative (gate PASS).  
**Artifacts:** `FULL_ACCURACY_TABLE.json`, `latency/CROSSOVER_GATE.json`, `E013_RETURN.md`, `DELTA_CORRECT.json`, `PANEL_SUMMARY.json`.

## Full accuracy (official N)

| arm | GQA% | POPE F1% | POPE acc% | MME | MMBench |
|---|---:|---:|---:|---:|---:|
| BASE_S8 | 58.785 | 84.835 | 86.100 | 1777.45 | 63.402 |
| B0_S8 | 58.785 | 84.835 | 86.100 | 1777.45 | 63.402 |
| REPLACE_RANDOM16 | 58.761 | 84.864 | 86.156 | 1779.72 | 63.316 |
| REPLACE_ATTN16 | 58.809 | 85.282 | 86.489 | 1779.27 | 62.973 |
| REPLACE_COVER16 | **59.008** | 85.234 | 86.444 | **1798.94** | 63.144 |
| EXTRA_RANDOM16 | 58.952 | 85.014 | 86.211 | 1761.88 | 63.746 |
| EXTRA_ATTN16 | 58.833 | 85.138 | 86.322 | 1772.96 | 63.488 |
| EXTRA_COVER16 | 58.968 | 85.054 | 86.267 | 1793.11 | 63.488 |
| EXTRA_NULL16 | 57.656 | 84.111 | 85.589 | 1737.45 | 60.395 |
| EXTRA_DUPLICATE16 | 58.881 | 84.610 | 85.978 | 1762.85 | 63.144 |

N: GQA 12578 / POPE 9000 / MME 2374 / MMBench 4329. BASE≡B0 on all four.

### Δ vs BASE (pp except MME raw)

| arm | ΔGQA | ΔF1 | ΔMME | ΔMMB |
|---|---:|---:|---:|---:|
| REPLACE_COVER16 | +0.223 | +0.399 | +21.5 | −0.26 |
| REPLACE_ATTN16 | +0.024 | +0.447 | +1.8 | −0.43 |
| EXTRA_COVER16 | +0.183 | +0.220 | +15.7 | +0.09 |
| EXTRA_ATTN16 | +0.048 | +0.304 | −4.5 | +0.09 |
| EXTRA_NULL16 | −1.13 | −0.72 | −40.0 | −3.01 |

## Decisions (no novelty claim for COVER)

- **H1 (REPLACE):** **inconclusive / partial.** COVER best on GQA (+0.22pp) and MME (+21.5) vs BASE/RANDOM/ATTN, but POPE F1 slightly below REPLACE_ATTN; MMBench not improved.
- **H2 (EXTRA):** **inconclusive / partial.** COVER strong on MME (+15.7 vs BASE) and beats NULL/DUPLICATE broadly; real EXTRA ≫ NULL supports “transient real context helps”; COVER does **not** cleanly beat ATTN/RANDOM on POPE/MMBench → **not** sufficient to claim a new diversity module.
- COVER remains a **test heuristic**, not a validated algorithm.

## Latency

- ABBA gate **PASS** @ 15%: pair1 rel ≈ +4.3%, pair2 ≈ +0.13% (E012 vs LEGACY Fixed32).
- Stage1 Fixed32 vs BASE: REPLACE_ATTN −1.0%, REPLACE_COVER −1.8%, EXTRA_ATTN −6.1%, EXTRA_COVER −8.1% (all within 15%).
- Addendum: do not headline algorithm speedups until E012 ~90ms story is fully closed; numbers above are measured under same runner.

## Bridge12 (panel N=512 only — not full ACC)

S12 panel completed; compare Δ within same S only. Use for depth/E008 check, not for official scores.

## Not done (by addendum design)

- CROSSIMAGE control (no verified patch).
- Gold log-prob / teacher-forcing utility (phase-2).
- Fixed-survivor-ID causal intervention (new branch forbidden in main run).
