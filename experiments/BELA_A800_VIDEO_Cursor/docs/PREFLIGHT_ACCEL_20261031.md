# Preflight — Video Accelerate & Compete Kit 2026-10-31

## Verdict: **GO for P0+P1** (efficiency); P2 Dual COVER / P3 QK semantic deferred until P1 gates pass.

## Inventory
| Item | Status |
|---|---|
| Kit handoff + coverage-delta reference | Installed under `accelerate_kit_20261031/` |
| Live `qa_core.py` facility-location ΔC | **MATCHES** kit assumption (`top1/top2` one-swap) |
| Live Stage1 | `NATIVE_L2_TOPK` + per-frame QA; `dual=(pooled,pooled)` |
| Live Stage2 | Physical 64→11→4; CPU `.float().cpu()` + hidden proxy |
| Baseline ACC/EFF | Locked in handoff §1 (Dense 63.26 / C,D 58.30 / Native 58.63; TTFT 2090/1601/984) |
| STAR proxy | Schedule-matched only — **not** official QR |
| GPU | 0/1 busy foreign; **2/3 free** — use these only |
| V100 image path | Untouched |

## Gaps / non-goals this round
- P2 true Dual-depth ViT features: not in current video tower hook → label adaptation later
- P3 same-layer Q/K attention: semantic change → after P1 equivalence
- Full 2700 ACC: only after P1 efficiency gate + optional R0–R4 dev panel
- MVBench: still no mp4

## Algorithm identity this round
- **P1**: exact-equivalent batched QA (same formulas/thresholds/B) — must match old keep IDs
- Stage2 GPU same-formula move (no D2H) — micro-opt; ranks should match FP32 proxy
- No new temporal module
