# QCP Preflight 2026-10-27

## Inventory
- Package: `experiments/bela_qcp/` (qcp.py, configs, launch, smoke)
- Host: LLaVA `e018_stage2_support._bela_qcp_refine` after RDRIFT first cut
- Stage1: Dual COVER stores `LAST_ENTRY_KEEP` + `LAST_ENTRY_METRIC` (elva_r3)

## Algorithm identity
- C0: historical L2 Dual+RDRIFT (reuse metrics, no re-run)
- C1: original RDRIFT + new `select_qcp` (not ICER rename)
- C2: matched-count random boundary swaps

## Gaps closed
- First-cut only; later cut native Q_ATTN
- QCP mutually exclusive with ICER
- Smoke: stage1=64, events=[[11,17],[19,10]], final=10; QCP swaps observed

## Verdict
**Ready** for W1024 pilot GQA/POPE/MMB/TextVQA on GPUs 0/2/4/7.
