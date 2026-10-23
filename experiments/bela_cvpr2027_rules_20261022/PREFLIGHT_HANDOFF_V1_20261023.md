# Preflight — BELA_ELVA_CVPR27_CURSOR_HANDOFF_V1 (2026-10-23)

## Inventory
| Source | Provides |
|---|---|
| Handoff zip | Research decision, runbook, schedules (A0/A2/A3/B1/B2/C1), offline audit/paired scripts — **no GPU model patch** |
| Repo host | E014/E018 + COVER Stage1 + RDRIFT first-event (proven on A0/A2/A3) |
| Scorer | `elva_scores.py` / official lmms metrics |

## Gap check
- Stage1 COVER, schedule env, K guards, question span, COVER/RDRIFT binding: **present** (same as T1 A*).
- New arms need only JSON schedules: B1 `172/90/20` W1808; B2 `172/120/32` W2288.
- C1 (3rd prune) **blocked** until P1 supports it — not launched.
- VizWiz: handoff marks `PROTOCOL_SUSPECT` — **excluded** per user (R8 / 8-task).

## Algorithm identity
| Arm | Operator | Identity |
|---|---|---|
| B1 / B2 | COVER + RDRIFT(first) + Q_ATTN(later) | **Original host** (same as A0/A2); schedule-only change |

## Already measured (skip ACC regen)
- A2/A3: full R9 done.
- A0: 6/8 (missing mmmu_val, ai2d) — fill those two only.
- A2/A3 TTFT: done. A0 TTFT: exclusive rerun in progress on GPU0.

## Verdict
**READY** for B1/B2 8-task ACC (no VizWiz) + B1/B2 TTFT. GPUs: 2/4/7 free; 0 reserved for A0 TTFT until drain.
