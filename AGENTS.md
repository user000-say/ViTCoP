# ViTCoP Research Governance

This repository is governed as a research project, not a demo. The single source of truth is `research/STATE.md`.

- MASTER alone may edit `research/STATE.md` and approve gates.
- PAPER, EXPERIMENT, and CODE read STATE and submit timestamped handoffs; they must not edit STATE.
- No GPU benchmark may start without an approved Experiment Spec and implementation audit.
- Preserve historical states, logs, closed experiments, and frozen thresholds. Never tune on test benchmarks or silently alter the protocol.
- The final objective is real accuracy–latency Pareto improvement. Token counts and FLOPs are supporting evidence only.
