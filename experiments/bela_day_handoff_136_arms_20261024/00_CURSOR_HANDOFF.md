# BELA / ELVA CVPR 2027 — 24-hour GPU Screening Handoff
Version: 2026-10-24 · **one frozen queue**, not permission for blind full-benchmark search.

## Core goals
1. Prioritize absolute GQA, POPE F1, MME Total and MMBench; protect ScienceQA-IMG and TextVQA; exclude suspect VizWiz from primary decisions.
2. Preserve COVER and RDRIFT until real cut-event code has been audited. For each module demand standalone and cross-stage ablations.
3. Require task-conditional evidence-lifetime story to be supported by causally appropriate comparisons, not invented after the scores.
4. Report W, Q_visual, actual TTFT, Fixed32, selection overhead and peak memory.
5. Treat user/AI/auditor suggestions as hypotheses. Never cherry-pick a dataset or use test-set performance to tune final hyperparameters.
6. Submit CVPR 2027 by official deadline; do not spend the whole GPU day on low-information full validations.

## Baseline lock / known state
Reported A0=172/90/32 W2048: GQA57.84, POPE83.31, MME1767.8, MMB62.20, SciQA68.96, TextVQA55.97, TTFT123.7, Fixed32=1234.6.
Reported A2=172/120/20 W2048: GQA57.62, POPE85.44, MME1774.9, MMB63.14, SciQA68.67, TextVQA54.89, TTFT122.5, Fixed32=1217.7.
Reported A3=172/130/16 W2048: GQA57.35, POPE85.68, MME1775.9, MMB62.80, SciQA68.67, TextVQA54.88, TTFT120.3, Fixed32=1177.1.
Reported B1=172/90/20 W1808: GQA PENDING; POPE83.45, MME1747.8, MMB62.37, SciQA68.86, TextVQA54.51; timings PENDING.
Reported B2=172/120/32 W2288: GQA58.44, POPE85.36, MME1804.2, MMB62.80, SciQA68.37, TextVQA55.13; timings PENDING.
B2 is diagnostic, not same-cost victory. VizWiz suspect; MME Total and Perception must never be mixed.

## What to do when the user is unavailable
P0 (blocking; finish before GPU fanout): collect B1 GQA & B1/B2 timing already running; fingerprint checkpoint/model/evaluator/data/seed/GPU; read CURRENT decoder source to name actual selectors at cut after L4 and L12; trace original patch IDs/KV/mask; verify physical pruning. No new module code.
P1 (broad but cheap): run pilot only on a FIXED and reproducible balanced subset of GQA+POPE+TextVQA. The CSV contains 96 exact-budget schedule arms (32 per W tier). **Do not run 96 full test sets**. Submit only arms with supported entry K and valid selector paths, sorted by phase+budget and then Q cost. Batch multiple GPU workers, but obey a GPU-hour cap measured from first calibration.
P2 (successive halving): evaluate the top 4-6 pilot schedules *per budget*, plus the best fixed baseline, on a larger **disjoint development subset**. Use predeclared multiobjective rules; reject task regressions; no task-special casing from test labels.
P3 (selector probes, conditional): 36 GATED arms are only eligible after call-map audit. Isolate **the real second-cut selector**. Hold entry, two cut positions, W, and pre-cut candidate set fixed. Random controls with multiple seeds are mandatory. If score source unavailable, status=BLOCKED and spend remaining GPU budget on Stage1 entry diagnostics instead.
P4 (confirmatory): run complete eight-task official evaluation for at most two new promoted arms per budget, if GPU-hours remain, and real timing. If throughput is insufficient, finish W2048 first, then W1024, then W4096. These final sets must not be used for further tuning.
P5: save JSONL predictions, POPE TP/FP/TN/FN by subset, GQA question type, TextVQA failure cases, layer IDs, actual W and error logs. Report failures. Generate STOP_GO.md.

## Ranking (precommitted)
Gate 1: valid exact W + original patch consistency + official evaluator.
Gate 2: no *material* GQA/POPE/ScienceQA/TextVQA regression relative to same-W strong baseline; require paired stats not just mean.
Gate 3: prioritize Pareto dominance for GQA/POPE/MME/MMBench and real latency. Avoid scalar R9 or suspicious VizWiz.
Gate 4: candidate must beat random budget assignment or random Token IDs when claiming an adaptive/selector effect.
No claim of faster inference from W alone. All reported result labels must distinguish REPORTED / MEASURED / NOT_RUN / BLOCKED.

## Budget note
Primary tiers are W1024/W2048/W4096; user's earlier "4028" must not be silently substituted. This queue uses 4+8+20 decoder layers and only changes the three K values. It is *not* an investigation of alternative cut times; those require separate code plumbing and controls.

## Novelty constraints
2025 ATP-LLaVA and PyramidDrop covered dynamic and progressive ideas; 2026 STAR-Pro, CoRePrune, QCPruner already cover stagewise attention, conditional deletion, query-aware coverage. Demonstrate an independently verified low-cost answer-relevant evidence-lifetime mechanism or admit if current results support only a strong fixed schedule.

## Deliverables
BASELINE_LOCK.md, REAL_CUT_CALL_MAP.md, RUN_MANIFEST.csv, PILOT_RESULTS.csv, PROMOTED_RESULTS.csv, RAW_TIMING.csv, PAIRED_DIAG.json, POPE_DIAG.json, SELECT_TRACE.jsonl, STOP_GO.md, SOURCE_DIFF.patch.
Do not represent the offline queue generator as an integrated GPU model patch.
