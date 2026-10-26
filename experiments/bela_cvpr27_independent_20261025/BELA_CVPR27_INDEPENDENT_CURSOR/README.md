# BELA/ELVA — Independent Evidence Handoff (2026-10-25)
Deliver this entire ZIP to Cursor and make `00_START_CURSOR.md` the only current instruction file.
All earlier 136-arm and 60-arm exploratory Handoffs are **historical archives**, not the final execution protocol.

**Not a live LLaVA patch.** Current GPU server production source and its scoring exports are not mounted here. This package includes a frozen experimental matrix, evidence-ranked research decision, source/implementation audit, an executable strict budget/Patch-ID verifier, and a paired 2×2 factorial tool; it does not claim to have generated new GPU ACC.

## Key files
- `00_START_CURSOR.md` — research verdict, exact 2×2/STOP-GO plan, GPU day-by-day, paper deadline.
- `configs/experiment_matrix.json` / `.csv` — 23 experiments (including references and gated arms), validated layer-average workload.
- `tools/budget.py` — exact per-layer W/Q and physical Patch identity audit (prefill batch1).
- `tools/factorial.py` — matched-question, image-cluster bootstrap for Stage1 × Stage2 GQA/POPE component attribution.
- `docs/PRIMARY_SOURCES_AND_CODE_AUDIT.md` — STAR-Pro, CoRePrune, QCPruner, MiCo, HAP, HiPrune primary-source distinctions and checked code knobs.
- `docs/SOURCE_AUDIT_AND_INTEGRATION.md` — exact source/score/Mask/KV acceptance gate before any transfer.
- `docs/STOP_GO_AND_EFFICIENCY.md` — explicit independent model decisions and latency gates.
- `paper/CLAIM_TO_EVIDENCE_BLUEPRINT.md` — CVPR 2027 eight-page framework and reviewer evidence obligations.
- `.cursor/rules/BELA_CRITICAL.mdc` — always-on project research standard.
- `tests/` — synthetic unit tests, never substitute for GPU accuracy reproduction.

## Command-line verification
```bash
python -m unittest discover -s tests -v
python -m tools.budget configs/experiment_matrix.json
python -m tools.factorial --BB bb.jsonl --QB qb.jsonl --BS bs.jsonl --QS qs.jsonl --mode acc
python -m tools.factorial --BB bb-pope.jsonl --QB qb-pope.jsonl --BS bs-pope.jsonl --QS qs-pope.jsonl --mode pope_f1
```
Input `.jsonl` lines require stable `sample_id`, `image_id`, `correct` boolean from the SAME official evaluator and for POPE the official `gold_yes_no`/`pred_yes_no` (`yes`/`no`). Correctness alone cannot calculate POPE F1; POPE false-positive/false-negative must be kept.

## Cite primary links, not the local research codes as achieved results
STAR-Pro paper https://arxiv.org/html/2609.05916
Code https://github.com/EasonAI-5589/starpro
CoRePrune paper https://arxiv.org/html/2609.26484
QCPruner paper https://arxiv.org/html/2609.19990
MiCo paper https://arxiv.org/html/2609.34330
CVPR Author Guidelines https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines

The primary scientific question is not "How many modules can we attach?" but "What dominates the precision gap under an identical pruning path, admission or retirement, and can one new low-cost mechanism mitigate it?"
