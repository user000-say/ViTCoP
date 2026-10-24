# Progressive schedule search — paper takeaways → Wave-1

## What the papers imply (not copied as modules)
| Paper | Useful claim | Our constraint |
|---|---|---|
| **STAR-Pro** | Broad Stage1 coverage + **nested progressive** refine; LLaVA schedules keep meaningful deep K (not →0) | Keep COVER; search entry **128/160/172** at W2048; do not collapse deep to 8 first |
| **QCPruner** | Query-aware coverage helps POPE; Top-K already strong | Do not blame RDRIFT alone; Stage1 pool + mid K matter (matches B1≈A0 POPE) |
| **CoRePrune** | Same final K, **set identity** repair > static Top-K | Selector probes stay **GATED** until cut2 audit; Wave-1 is schedule-only |
| **B1/B2 (ours)** | Mid↑ → POPE; deep↑ → GQA/MME; non-iso B2 not same-cost SOTA | Wave-1 densifies **isobudget** mid/deep around A2 and STAR-like entry |

## How handoff selects (and how we obey + override)
Handoff: 96 PILOT schedules → successive halving on a **pilot subset** → promote → full eval.  
**User override:** no subset sampling; each arm judged by **full 6-task ACC**.  
We still progressive on *which arms* run: Wave-1 = 8×W2048 curated + 7×W1024@96 (skip measured), not all 136, not GATED, not W4096 yet.

## Already known baselines (skip)
A0/A2/A3/B1/B2 full metrics+TTFT; W1024_PROBE 96/50/12 six-task done (R4≈91.5, weak vs W2048).
