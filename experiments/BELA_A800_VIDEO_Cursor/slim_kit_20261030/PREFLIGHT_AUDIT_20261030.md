# Slim kit preflight audit — 2026-10-30

## Verdict: **CONDITIONAL GO** — Stage2 must be wired+smoke PASS before ACC/EFF.

## Inventory
| Item | Status |
|---|---|
| Model LLaVA-Video-7B-Qwen2 | Present; 28 layers; SigLIP→pool→169/frame |
| Stage1 native/qacover T32 | Live (`NATIVE_L2_TOPK` substitute, not Dual COVER) |
| Stage2 RDRIFT/Q_ATTN physical | **NOT hooked** (blocker) |
| off==dense | PASS (prior) |
| Video-MME media | 900 mp4 ready |
| MVBench media | **0 mp4** — ACC deferred until data or skip |
| lmms-eval | `/data1/xiangyue/src/lmms-eval` via PYTHONPATH |
| STAR-Pro official video overlay | **Absent** in public starpro; must label substitute |
| GPU 0/1 | Busy (foreign JarvisIR) — use **2/3 only**; do not kill |
| /data1 disk | 96% used, ~334G free |

## User scope vs slim matrix
User: only image-best two algorithms + Dense + STAR on A800.
Frozen arms (after Stage2 PASS):
- **DENSE** OFF/OFF
- **C** T32 QA B8 + Q_ATTN_ALL  (image P3)
- **D** T32 QA B8 + RDRIFT_FIRST (image S1)
- **STARPRO_T32** schedule-matched proxy: Stage1 NATIVE_L2_TOPK + Stage2 Q_ATTN_ALL (NOT official QR; labeled substitute)
- Optional T16 **F** after T32 winners if time

Pilot A/B/E/G skipped per user narrow scope.
