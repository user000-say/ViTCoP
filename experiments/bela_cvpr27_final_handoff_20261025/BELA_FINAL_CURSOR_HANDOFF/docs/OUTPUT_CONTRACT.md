# Cursor mandatory output files
Under `runs/FINAL_20261025/`:
- `BASELINE_LOCK.md` — production git SHA, dirty state, model/config/dataset/evaluator hashes.
- `CUT_CALL_MAP.md` — actual selector file:fn:line for each cut with original Patch identity and causal signal availability.
- `PATCH_TRACE.jsonl` — documented in docs/ENGINEERING_CONTRACT.md; three smoke cases + winning validation cases.
- `RUN_MANIFEST.csv` — task, dataset revision, GPU, run time, seed, checkpoint hash, job status, exit_code, stderr path.
- `RESULTS_MASTER.csv` — absolute task scores; POPE F1; MME Total and Perception separate; W, actual timings; NOT_RUN not 0.
- `POPE_CONFUSION.json` — random/popular/adversarial TP/FP/TN/FN/precision/recall/F1, official answer parser.
- `PAIRED_CI.json` — per-image cluster bootstrap for GQA and TextVQA; F1 bootstraps for POPE.
- `LATENCY_RAW.csv` — warmup 10+ measured repeated runs p50/p90; include vision and selector time separately.
- `CROSS_ABLATION.csv` — Stage1×Stage2 factorial, matched W/stages; why any missing cell.
- `SOURCE_DIFF.patch` — small, tested, default-OFF experiment hook.
- `STOP_GO.md` — winner, task-level gains and losses, mechanism evidence, latency cost, next decision.
- `paper/main.tex` — 8p CVPR draft + BibTeX; no unverified results.
Fatal mistakes: writing complete where only a pilot subset ran; renaming MME reference; silently changing test prompt; overfitting on test; mixing GPU machines when comparing ms.
