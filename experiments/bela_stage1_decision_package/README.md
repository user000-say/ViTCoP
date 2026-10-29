# BELA CVPR2027 Stage1 — Decision Experiment Kit

**This package was created from the 34-round chat export and public literature (2026-10-29).**

## What is actually validated

- Extracted `query_anchor_history_reference.py` from the **verbatim reference code block in the uploaded chat export**. This is not the latest production repository.
- Offline QA reference behavior checked at `N=96, K=24` and at the LLaVA `N=576, K=64` scale.
- 12 CPU/unit tests passed, including B8 vs B16, continued 8+8 exchange, invariants, empty/flat query, invalid IDs, POPE metrics, image-cluster bootstrap and trace replay.
- The controlled `joint_baseline.py` is a **newly authored basic surrogate**, not a reimplementation of CoIn, SCoRe, MMTok or QCPruner. For publishable literature comparison use the original method and official code.
- No production BELA checkout, raw test predictions, V100, A800 or real profiler results were supplied. Therefore **no new GPU accuracy or latency claims** are made.

## Files

- `CURSOR_HANDOFF.md`: full priorities, factorial arms, controls, prerequisites, reviewer-grade evidence and chart plan.
- `query_anchor_history_reference.py`: historical QA-COVER PyTorch reference extracted from chat.
- `joint_baseline.py`: one-pass greedy visual coverage and query score comparator with an optional K−B native-overlap constraint.
- `bela_eval.py`: paired image-cluster bootstrap of real per-question results and replay/audit of patch-ID traces.
- `plot_real_trace.py`: real query-proxy heatmap and patch selection masks; **requires real patch scores**, will not manufacture data.
- `test_stage1.py`: runnable regression, edge-case and statistical consistency tests.
- `TEST_REPORT.json`: historical numeric verification and local CPU tests, clearly separate from model measurements.

## Run

```bash
python test_stage1.py
python bela_eval.py audit --trace REAL_TRACES.jsonl --out audit.json
python bela_eval.py compare --metric pope --baseline L2_POPE.jsonl --candidate QA_POPE.jsonl --bootstrap 2000 --out pope_report.json
python bela_eval.py compare --metric gqa --baseline L2_GQA.jsonl --candidate QA_GQA.jsonl --bootstrap 2000 --out gqa_report.json
python plot_real_trace.py --trace REAL_TRACES.jsonl --sample-id YOUR_UNIQUE_ID --height 24 --width 24 --out trace.png
```

Requires Python >=3.10, numpy, torch, matplotlib.
`bela_eval.py` alone only requires numpy and the standard library.

**Important:** GQA comparisons should export each sample's `correct` from the exact original evaluator. Otherwise the script explicitly labels its internal fallback as *simple EM, not official*. POPE `question_id` is not unique; create a stable unique `sample_id`. Cluster bootstrap by `image_id`, not by question ID.

## Disclosure

The referenced QA source is taken from the uploaded conversation's historical assistant reply. Verify the actual source code, tensor semantics, and git SHA on the server before running formal tests or citing its internals in your paper.
