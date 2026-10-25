# Execution and job dispatch protocol
This is not an already-running GPU scheduler. Cursor must construct job commands using the REAL deployed runner and verify 3 smoke samples.

## GPU lanes
- GPU 0: W1024 late-first-cut experiments (L2, L3).
- GPU 2: W2048 staged retirement (E1, E2, E3).
- GPU 4: official STAR-Pro fair baseline + task/selector diagnostics, only after installing in a separate environment.
- GPU 7: real end-to-end speed, E1 instrumentation, second scorer audit and trace.
If insufficient GPUs or hardware differs, reallocate with GPU, memory, batch info logged; no mixing timing on dissimilar GPUs.

## No more brute-force full 8-task searches
Current `config/experiments.json` has 30 configs including 7 historical references, stage probes, and GATED controls. **Only six P1 schedule probes are the main initial candidates**. Each new experiment needs precise W and measured Q.
- Pilot: fixed, hash-locked DEVELOPMENT subset of GQA/POPE/TextVQA, not the final full benchmark set.
- Promotion: best 1–2 per budget; candidate vs best old baseline paired, report per-image bootstrap CI; no adaptive retuning from confirmation set.
- Confirmation: official full GQA/POPE/MME Total/MMBench/SciQA/TextVQA and, if possible, AI2D/MMMU.
- Stop tests whose token ID trace fails, RDRIFT missing, actual W mismatch, or error bars/point estimate show severe broad degradation.

## Priority safety stop / go rules (operational heuristics, not hypothesis-test thresholds)
At W2048, proposed ERS should aim at GQA recovery over A2 57.62, maintain POPE near or above 85.44, not worsen TextVQA below A2 54.89, and not increase p50 true end-to-end TTFT materially. Tiny gains < noise are not convincing; a method that consumes appreciably more latency must show proportional ACC utility.
At W1024, prioritize large relative recovery from best reproducible W1024 fixed baseline; an improvement limited to one dataset is insufficient.
No test-set-specific hyperparameter selection, no tuning against suspect VizWiz.
If every algorithmic hypothesis fails, reserve GPU time for the strongest fair baseline and complete the paper with qualified claims rather than fabricating results.

## Every 4-hour log artifact
`RUN_STATUS.md`: completed, blocked and remaining jobs, exact command, failure stack if any, allocated GPU hours, best dev candidates per W, current STOP/GO.
All metrics must distinguish POINT_ESTIMATE, CONFIRMATION, and PAPER_REPORTED.
