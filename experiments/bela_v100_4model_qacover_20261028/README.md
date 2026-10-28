# BELA Cross-Backbone V100 Validation Pack

Start with **[Cursor Handoff](docs/CURSOR_HANDOFF.md)**. This archive is a **CPU-verified preflight/selector-contract pack, NOT a production Hugging Face integration and NOT new GPU model scores**.

Includes an unchanged snapshot of the prior 23-test QA-COVER implementation under `qa_cover_original/`, 16 new cross-backbone tests, model-specific schedules, model-config/DeepStack/crop validators, a strict `results_pending.csv` with 192 missing scores, and **separate** published Dense references.

Commands:
```
python -m unittest discover -s tests -v
(cd qa_cover_original && python -m unittest discover -s tests -v)
python -m tools.budget_planner --model qwen3-vl-8b --tier stress --observed-n 1296
python -m tools.model_config_probe --config-json /path/to/actual/checkpoint/config.json
```

**No model weights or GPU evaluation predictions are included.**
