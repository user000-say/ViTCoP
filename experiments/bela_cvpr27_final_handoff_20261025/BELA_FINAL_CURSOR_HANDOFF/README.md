# BELA / ELVA CVPR 2027 — Final Research Freeze V1

**Give Cursor this entire ZIP and paste/point it to `00_CURSOR_START_HERE.md`.**
The latest GPU model source and prediction files were not available to this environment. This package contains **real, tested research planning/diagnostic Python code**, NOT a modified or GPU-validated BELA production forward.

- `config/experiments.json` / `.csv`: 30 prioritized experiment plans (some references, gated controls, 6 main P1 arms).
- `tools/schedule.py`: strict W and Q auditing and runtime original-Patch-ID checking.
- `tools/results.py`: official-scored per-sample paired error analysis, image-cluster bootstrap, POPE confusions.
- `tests/`: synthetic boundary-condition tests. `python -m unittest discover -s tests -v`.
- `01_RESEARCH_DECISION_AND_NOVELTY.md`: two falsifiable mechanisms and reviewer attack risks.
- `paper/`: actual 8-page paper writing outline, abstract draft with TBD, ablation/figures and mandatory admin requirements.
- `docs/`: engineering contract, GPU weekly dispatch, reporting and official 2027 compliance.
Nothing is allowed to silently mutate the original evaluation scripts or append external LLM results as if self-reproduced.
