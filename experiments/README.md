# Experiment git checkpoints

After each completed experiment (or substantive code change for that experiment):

1. Commit **code + compact result summaries** (not full `logs/` dumps).
2. Message / tag naming: `eNNN-...` matching the experiment id, e.g.
   - `e013-p1: Stage1 REPLACE/EXTRA full ACC + ABBA latency`
   - `e012-sd-p1: ...`
3. Put key tables under `experiments/<exp_id>/` so they are tracked even if `logs/*` is gitignored.
4. Tag the commit: `git tag eNNN-...`
5. **Always `git push origin main` and `git push origin <tag>`** after commit (full remote save).
6. Do not commit `__pycache__`, `.pyc`, secrets, or multi-GB raw sample dumps.
