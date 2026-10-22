# BELA / ELVA — CVPR 2027 Research-to-Cursor Handoff

**Date:** 2026-10-22  
**Nature:** executable *research prototype* and GPU-integration instructions; **not a model-integrated production patch**.

## Start here

1. Read **`01_研究负责人独立判断.md`** (research position and why these experiments are worth running).
2. Read **`02_CURSOR_HANDOFF_CN.md`** (precise integration gates, runtime invariants, source cautions and STOP rules).
3. Run the CPU tests below, then inspect `configs/experiment_matrix.csv`.
4. On the actual server, execute the read-only `scripts/server_preflight.sh` BEFORE editing any LLaVA file.
5. Only after all gates pass, implement a new off-by-default branch in the real model.

## What works NOW

- Exact post-block 32-layer W calculator; integer feasibility and nested patch-ID checks.
- Stage1 feature-space residual coverage diagnostic (NumPy + Torch reference); a separate CPU diagnostic can use multiple ACTUAL, potentially non-nested K-sets without inventing a greedy order.
- Unlabeled DEV difficulty-threshold calibration and heldout image-budget plan.
- Two EXACTLY equal-per-example-W cut12 vs cut20 Stage2 candidate lifetimes.
- Baseline-score interface (does not reimplement production RDRIFT).
- Randomly shuffled budget assignment control keeping exact profile histogram.
- Strict POPE yes/no diagnostic, a paired Dense-vs-BELA error flip summary.
- JSONL patch-ID + actual 32-block Token trace validation.
- CPU unit/edge-case tests and synthetic example inputs.

## What has NOT been demonstrated

- No official GPU smoke, live LLaVA integration, trained model run, benchmark accuracy, TTFT or FLOPs are created by this package.
- The actual current production dual-COVER and RDRIFT source is not in the uploaded E014 archive; its presence must be verified in the current server checkout.
- The proposed image complexity signal and online decision thresholds are **hypotheses**, not measured accuracy gains.
- The POPE diagnostic does not replace the team's original/official scoring implementation.

## Commands

```bash
python -m compileall -q bela_next scripts tests
python -m unittest discover -s tests -v

python -m scripts.plan_offline calibrate \
  --input examples/dev_signals.jsonl \
  --policy examples/frozen_policy.json --model-revision SYNTHETIC

python -m scripts.plan_offline plan \
  --input examples/heldout_signals.jsonl \
  --policy examples/frozen_policy.json \
  --output examples/heldout_plan_fixed.jsonl --mode fixed

python -m scripts.plan_offline plan \
  --input examples/heldout_signals.jsonl \
  --policy examples/frozen_policy.json \
  --output examples/heldout_plan_dynamic.jsonl --mode adaptive

python -m scripts.make_matched_controls \
  --input examples/heldout_signals.jsonl \
  --policy examples/frozen_policy.json \
  --output examples/shuffled.jsonl

python -m scripts.analyze_pope --dense examples/pope_dense.jsonl \
  --bela examples/pope_bela.jsonl

python -m examples.make_trace_example
python -m scripts.audit_trace --input examples/sample_trace.jsonl
```

Run **from this root**. Python dependencies: `numpy`; `torch` for the optional Torch reference/tests. Use the same PyTorch environment as the production model when integrating. No training, GPU or model weights are required for offline unit tests.

## Profile identity

- EASY: entry96 → middle64 → final32, cut4/12, W1536.
- NORMAL: entry160 → middle96 → final32, cut4/12, W2048.
- HARD: entry172 → middle154 → final32, cut4/12, W2560.
- Online Stage2 alternative cut4/20 allocates EXACTLY the same respective W with intermediate sizes 48/64/93.
- If their frequency on new data differs from the calibration sample, the mean W may differ. Always audit realized W and compare matched budget distributions.

## Scientific gate

**Do not conclude “image-adaptive is useful” until a complexity-based budget allocation beats both the best verified FIXED schedule and profile-histogram-matched shuffled allocation on an image-disjoint confirmation set, with real latency and official scorer.**

All source papers and project evidence are linked in `01_研究负责人独立判断.md`.

## Directory map

```text
01_研究负责人独立判断.md
02_CURSOR_HANDOFF_CN.md
03_历史实验与近邻核查索引.md
README.md
bela_next/
    budget.py
    signals.py
    stage1_torch_adapter.py
    policy.py
    stage2_adapter.py
scripts/
    plan_offline.py
    make_matched_controls.py
    analyze_pope.py
    audit_trace.py
    server_preflight.sh
configs/
    experiment_matrix.csv
    run_manifest_template.json
examples/
    dev_signals.jsonl
    heldout_signals.jsonl
    pope_dense.jsonl
    pope_bela.jsonl
    make_trace_example.py
tests/
    test_budget_and_policy.py
    test_pope_and_torch.py
    test_audit_trace.py
```
