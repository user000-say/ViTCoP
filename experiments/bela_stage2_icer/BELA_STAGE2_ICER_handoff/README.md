# BELA Stage2 ICER handoff package

This bundle provides a research prototype and a Cursor integration plan for the BELA Stage2 question: can bounded, interaction-conditioned complementarity repair improve the current Q-Attn/RDRIFT selectors without changing the token-layer budget?

## Local checks
```bash
python -m unittest discover -s tests -v
python -m compileall -q bela tests
```

The unit tests validate mathematical/selection invariants only. They do not validate LLaVA integration, accuracy, CUDA latency, or FLOPs.

## What this is
- `bela/stage2_icer.py`: bounded swap selector plus CoRePrune-style equation-level diagnostics.
- `bela/latency_audit.py`: timing helpers to insert around real model stages.
- `CURSOR_HANDOFF_CN.md`: full algorithm design, integration contract, timing diagnosis, and staged experiment plan.
- `experiment_matrix.csv`: executable research matrix for Cursor.

## Critical limitation
The current uploaded artifact contains handoff/core code, not the production LLaVA checkout and exact active model hook. Therefore this package does **not** claim to patch `llava_arch.py`, update real KV cache/position IDs, or produce GPU results. Cursor must apply the selector at the verified existing pruning hook and use the existing validated tensor/cache path.
