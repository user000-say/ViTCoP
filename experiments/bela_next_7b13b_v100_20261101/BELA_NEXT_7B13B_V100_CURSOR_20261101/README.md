# BELA NeXT-7B/13B × 94.4% / 88.9% on Four V100 GPUs

A research-backed handoff bundle with **executable, synthetic-tested**
budget/crop-aware index planning code. Designed for Cursor on the
user's EXISTING V100 LLaVA/BELA repository.

## Open first
1. [`CURSOR_RUN_FIRST.md`](CURSOR_RUN_FIRST.md)
2. [`docs/00_MASTER_HANDOFF_CN.md`](docs/00_MASTER_HANDOFF_CN.md)
3. [`docs/01_RESEARCH_PAPER_PARAMETER_MATRIX.md`](docs/01_RESEARCH_PAPER_PARAMETER_MATRIX.md)
4. [`docs/02_ENGINEERING_NE_XT_ANYRES.md`](docs/02_ENGINEERING_NE_XT_ANYRES.md)
5. [`docs/03_V100_FOUR_GPU_RUNBOOK.md`](docs/03_V100_FOUR_GPU_RUNBOOK.md)
6. [`docs/04_SELECTION_AND_RESULTS_PROTOCOL.md`](docs/04_SELECTION_AND_RESULTS_PROTOCOL.md)

## What is included
- `configs/arms.json` and `arms.tsv`: 20 integer-verified, exact W
  model×budget×schedule arms.
- `src/bela_next_budget/budget.py`: exact visual patch token-layer
  accounting and calibrated dynamic actual AnyRes N ratio planner.
- `src/bela_next_budget/crops.py`: deterministic 5-crop quota with
  hard caps and optional global-thumbnail boost.
- `src/bela_next_budget/preserve.py`: safe physical-gather index
  plan keeping special/newline/text tokens intact.
- `tools/v100_preflight.py`: read-only real four physical V100 card
  occupancy check, 7B/13B actual config layer lock.
- `tools/print_arm_env.py`: prints configuration flags (not a model run).
- `tools/plan_priority_queue.py`: prints bounded run order
  (not a GPU run).
- `tools/verify_run_manifest.py`: checks real visual patch counts
  and expected actual per-layer W in paper five-crop protocol.
- `tools/score_dev_matrix.py`, `tools/paired_dev_accuracy.py`:
  independent dev ranking and native-scorer paired analysis.
- `tests/`: synthetic CPU tests; doesn't prove A800/V100 performance.
- `evidence/`: historical user-provided LLaVA-1.5-7B parameter facts.

## Run local Python tests
```bash
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
python -m unittest discover -s tests -v
python tools/print_arm_env.py --arm NEXT7_T160_A0_PAPER
```

## Critical limitations
- NO V100 user-host source / checkpoint bytes in this bundle.
- New flags don't work on the unmodified repo.
- No exact patch against V100 live code can be certified here.
- Real V100 GPU inference, new ACC, model speedup, and KV parity
  have NOT been run and must NOT be reported as achieved.
- Paper STAR-Pro T160/320 refers 5-crop layer-average patch count;
  MMTok final cap160/320 is not the same compute budget.
- V100 does NOT support BF16 or FlashAttention2; use FP16
  and verified eager/compatible SDPA.
