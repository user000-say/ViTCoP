# CUT_CALL_MAP — production host (git `da71409f` + COVER K capacity expand)

Decoder indices below are **0-based post-block** as consumed by `E018_EVENTS` (`apply_llava.e018_events`: cut after completing `sum(lengths[:i])` blocks → event key `sum-1`).

## Stage1 (pre-decoder entrance)
| Item | Value |
|---|---|
| Path | `experiments/elva_r3/llava_stage1.py:select_existing_forward` → `elva.cover_binding.select_checked` → `native_cover.legacy_callback` → `e013_stage1_support.e013_choose(..., pick=COVER, core_k=K0)` |
| Signal | CLS→patch attention sum on penultimate (+ layer9 when guide=dual); **no future decoder leakage** |
| Payload | Penultimate CLIP patch rows (projector input) |
| K0 | Config `schedule.counts[0]`; allowlist `{64,72,80,96,128,160,172}` |

## Stage2 cut #1 (first progressive event)
| Item | Value |
|---|---|
| Path | `llava/model/e018_stage2_support.py` `_tail` → `_event` → `_pick` |
| Scorer | Baseline `Q_ATTN` (`side["q_attn"]` = question→vision mean); if `first_selector=RDRIFT`, **E020 rank_drift refines FIRST event only** (`_pick` lines ~390–418; `layer == events[0][0]`) |
| E020 | `e020_lib/dispatch.refine` kind=`rank_drift`; `E020_NEXT_AFTER=last_event_layer+1` |
| Physical | `_drop_ids` gather on hidden + optional layer_local KV scrub |

## Stage2 cut #2 / #3 (later events)
| Item | Value |
|---|---|
| Path | Same `_pick`; **no E020** on non-first events |
| Scorer | `Q_ATTN` (`later_selector`) |
| Cap | `e018_contract`: ≤3 events; runtime raises on 4th physical prune |

## Example mapped schedules
| Arm | E018_EVENTS (0-based) | Meaning (1-based after-layer) |
|---|---|---|
| A2 / Wave1 3-stage | `3:120,11:20` | after L4→120, after L12→20 |
| E1_RET32_TO12 | `3:120,11:32,19:12` | after L4→120, L12→32, L20→12 |
| L2_LATE_64_17_10 | `11:17,19:10` | after L12→17, L20→10 (K0=64 from Stage1) |
| L3_MID_80_24_12 | `7:24,15:12` | after L8→24, L16→12 |

## Provenance rules
- Surviving IDs are original patch IDs carried in `state.visual_ids`; prune must be subset.
- Prefill-only progressive prune; decode uses shortened cache after physical gather.
- RDRIFT is **not** re-applied at the final cut under current COVER+RDRIFT host.
