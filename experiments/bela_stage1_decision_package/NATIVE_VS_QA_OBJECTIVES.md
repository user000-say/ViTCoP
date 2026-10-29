# NATIVE_VS_QA_OBJECTIVES — P0

## Native Dual COVER (production)

- **Input features:** Dual-depth metric from Layer9 + penultimate path inside `select_checked` when `guide=dual`, `beta=0.5` (config `S1_QANCHOR_L2` / `L2_NATIVE`).
- **Importance scores:** CLS→patch attention sum at penultimate (`_sum_cls`), plus early scores when Dual needs them.
- **Operator:** `e013_choose` REPLACE/COVER with `core_k=64`, `replacement_slots=32`.
- **Output:** fixed-K unique patch IDs; no question text enters Native COVER.
- **Saliency guard in QA:** `saliency_guard_slots=0` on frozen L2 QA configs (Native saliency not passed into QA unless guard>0).

## QA-COVER (production)

- **When:** `BELA_QA_ENTRY_MODE=s1_anchor` (or `matched_random`).
- **Query construction:** request-local question token ids from `e006_span.explicit_span` — user question only; image placeholder `-200`, system prefix, and `ASSISTANT:` tail excluded; frozen template-piece content mask drops function words (`LIVE_CALLMAP.md`).
- **Visual side:** full 576 penultimate rows through live `mm_projector` for the query proxy (timed separately); Dual `metric` supplies coverage/novelty geometry.
- **Frozen AnchorConfig (L2):** `max_swaps=8`, `donor_pool=16`, `drop_pool=12`, `min_raw_std=0.008`, `min_query_gain=0.08`, `min_novelty=0.05`, `max_coverage_loss=0.005`, `coverage_weight=0.5`, `saliency_guard_slots=0`.
- **Contract:** final keep stays K=64 unique IDs in ascending spatial order; `LAST_ENTRY_KEEP` overwritten; `LAST_ENTRY_METRIC` remains full-grid Dual descriptor.
- **GPU sync risk:** historical reference used `float(std)` / `argmax` style host syncs; production path is `experiments/bela_query_anchor/query_anchor.py` — profiler still required under P4 before latency claims.

## Objective difference (falsifiable)

| | Native COVER | QA-COVER |
|---|---|---|
| Uses question | No | Yes (question-only embeds) |
| Changes K | No (fixed 64) | No (bounded swaps ≤B) |
| Can change ID set | Yes (selects the 64) | Yes (replaces ≤B members) |
| Stage2 | Same RDRIFT schedule | Same RDRIFT schedule |

Kit `joint_baseline.py` is an **authored surrogate** one-pass greedy (coverage gain + weighted query), **not** CoIn/SCoRe/MMTok/QCPruner official code.
