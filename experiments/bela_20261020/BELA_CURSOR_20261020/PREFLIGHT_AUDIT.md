# BELA 20261020 preflight (2026-10-21)

## Inventory
| Package provides | Host already owns |
|------------------|-------------------|
| `bela/core.py` Plan / LifetimeState / lifetime_scores | LLaVA + e014_boot + E018 prune + ELVA Stage1 COVER/PCOV |
| unit tests (15 PASS), CLI exact-W | R3 W2048 V5 / W1024 V4 configs + metrics (R4 four) |
| `model_hook_contract.py` (example only) | e020 RDRIFT first-event refine |
| `legacy_e020_4/` (reference, not auto-merged) | 8-task ACC wiring (`elva_acc_tasks.ACC_TASKS`) |

## Gaps (blocked for claiming “BELA model ACC”)
- No `task_entrypoint` / server_binding that calls `LifetimeState` inside decoder.
- `tensor_adapter` does not touch KV/mask/RoPE; must use repo gather path.
- `lifetime_scores` is an explicit surrogate, not verified = historical RDRIFT.

## Algorithm identity for this launch
| Arm | Operator | Label |
|-----|----------|-------|
| W2048 V5 Q_ATTN | Original ELVA Stage1 dual+COVER + E018 Q_ATTN | Gate1 fixed strong |
| W2048 V5 RDRIFT | Same Stage1 + E020 rank_drift FIRST | Gate1 REF/RDRIFT |
| W1024 V4 PCOV | Original R3 pen+PCOV60_B12 + Q_ATTN | Gate1 low-budget |

BELA `LifetimeState` host patch = **not in this zip**; do not report these arms as BELA selector.

## Smoke
- Package: 15/15 unittest + CLI W2048/W1024 exact plans OK.
- Host 8-task path: previously smoked scienceqa/mmmu/ai2d/textvqa_val_ocr.

## Verdict
**ready for Gate1 8-task ACC** on the three frozen arms above.  
**blocked** for end-to-end BELA LifetimeState accuracy claims until decoder hook lands.
