# Preflight 2026-10-31 — V100 77.8% W4096 + cross-model

## Verdict
**GO** for LLaVA-1.5-7B W4096 schedule A (256→74→36) S1 vs P3 six-ACC on GPUs 0/4/7.
**BLOCKED** for progressive Native/QA on NeXT / Qwen3-VL / Qwen3.5 (no host prune binding).
**QUEUED** Dense NeXT-7B (+ remaining Dense script) after W4096 drains.

## Smoke
- Dual COVER K256 + QA B8 + RDRIFT: `entry_trace` K=[256,256], swaps=[8,8], e014_boot GQA limit=2 PASS.
- Events `11:74,19:36`, E020 rank_drift on, slots=32 unchanged.

## Live
- Queue: `launch_w4096_s1_p3_six.py` → `logs/e014_aee/full/bela_w4096_77p8_20261031/{S1,P3}/`
- GPU2 foreign busy (wangshirui); project set still 0/2/4/7 with auto-skip.
