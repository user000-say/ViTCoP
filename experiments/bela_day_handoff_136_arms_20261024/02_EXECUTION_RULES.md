# 24-hour dispatch policy (not automatic cluster commands)
- First calibrate actual jobs/hour using 2 representative pilot arms per budget and reserve GPUs for B1 GQA plus B1/B2 latency. Do not extrapolate from nominal FLOPs.
- Every queue row is a candidate, NOT an instruction to run complete eight-task evaluation. Never submit the 36 GATED selector arms until the actual second-cut scorer is audited.
- Keep one stable pilot sample-ID manifest, fixed seed, identical official evaluator and image preprocessor. Do not optimize using the untouched full benchmark results.
- Allocate the measured GPU-hour pool: 10% source audit/baseline & timing; 50% PILOT spread across W1024/W2048/W4096 (W2048 first if backlog); 25% successive-halving reruns of top 4–6 per tier; 15% confirmation/timing. If audit fails, repurpose unspent time to reproducing baselines only.
- Pilot stop if patch identity is invalid, NaN/shape mismatch, missing token layers, or selection repeats; log BLOCKED reason and skip.
- Pilot ranking: within each W, paired GQA/POPE/TextVQA improvement with no major ScienceQA regression, tie-break measured TTFT and Q_visual. Validate MME/MMB after first halving; do not rely on suspicious VizWiz/R9.
- Production source stays unchanged by default. Cursor may implement a local runner adapter only after documenting exact CLI invocation and demonstrating three smoke inputs.
- Save raw RUN_MANIFEST (seed, model sha, task split, data hash, actual W, GPU, start/stop, failures), jsonl predictions and timing. STOP_GO.md must identify a single next algorithm hypothesis or explicitly NO_GO.
