# BELA Gate1 server results (in progress → finalize by watcher)

## Scope
- **Not** BELA `LifetimeState` model hook (zip has no production forward patch).
- Gate1 frozen baselines on current ELVA host: W2048 V5 COVER+Q_ATTN, W2048 V5 COVER+RDRIFT, W1024 V4 PCOV+Q_ATTN.
- ACC = 8 tasks (`elva_acc_tasks.ACC_TASKS`); R4 uses POPE **F1**.
- Latency = SparseVLM-style model-level `generate` on GPU7 (wall + CUDA Event, warmup 12, Fixed32, peak mem). Not service E2E.
- FLOPs = visual W + transparent O(d²·W) proxy; fused kernels not counted.

## Commands
```bash
python experiments/bela_20261020/BELA_CURSOR_20261020/scripts/launch_gate1_8task.py
python experiments/bela_20261020/BELA_CURSOR_20261020/scripts/launch_gate1_ttft.py
python experiments/bela_20261020/BELA_CURSOR_20261020/scripts/summarize_gate1.py
```

## Outputs
- ACC: `logs/e014_aee/full/bela_gate1_8task/<arm>/<task>/metrics.json`
- Timing: `server_results/timing/<arm>/latency_summary.json`
- FLOPs: `server_results/flops/<arm>.json`
- Aggregate: `server_results/official_scores.json` (written when watcher sees 24+3)
