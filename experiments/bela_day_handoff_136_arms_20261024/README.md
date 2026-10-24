# Daily BELA Experiment Queue
Run `python -m unittest discover -s tests -v` and `python -m scripts.generate` from this directory.
Queue columns: arm, phase, budget, k0, kmid, kdeep, cut_after_1, cut_after_2, selector, w_check, q_proxy, needs_stage1_check, status, note.
No runtime LLaVA code is modified. The 3-stage generator assumes 32 decoder layers with cuts AFTER layers 4 and 12.
Read `00_CURSOR_HANDOFF.md` before scheduling GPU experiments.
