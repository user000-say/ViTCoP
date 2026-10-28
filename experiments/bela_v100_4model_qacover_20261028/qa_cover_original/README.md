# BELA / CVPR 2027: Stage1 QA-COVER Cursor delivery

**Start with [`docs/CURSOR_HANDOFF_FINAL.md`](docs/CURSOR_HANDOFF_FINAL.md).**

This archive contains tested standalone PyTorch selector code, a request-local LLaVA integration helper, a POPE confusion audit, query-shuffle controls, configs, and unit tests. **It is not a production merge and contains no new GPU metric.**

Run: `python -m unittest discover -s tests -v` from archive root.

Default saliency guard is off; enable only for separate evidence-backed ablation.
