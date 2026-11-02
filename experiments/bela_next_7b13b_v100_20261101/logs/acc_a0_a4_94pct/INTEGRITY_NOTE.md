# NeXT-7B A0–A4 integrity note (2026-11-02)

All five arms produced **byte-identical** `*_samples_*.jsonl` on every task
(GQA/POPE/MME/MMBench/SciQA/TextVQA). Arm banners show different
`BELA_NEXT_ARM`, but predictions do not differ.

Prediction diffs vs Oct-28 Dense are small (0.1–3.8% of samples) and are
consistent with forced `image_grid_pinpoints=[[672,672]]` (Dense used the
default multi-pinpoint set), **not** with distinct Stage1/Stage2 schedules.

**Conclusion:** reported A0–A4 absolute/relative numbers are **not valid
schedule ablations**. Treat as a single nearly-Dense AnyRes-672 run until
forward/eval-path differentiation is proven (per-sample token traces in
lmms generate) and arms are re-run.
