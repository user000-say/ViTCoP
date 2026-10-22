# Protocol install status — 2026-10-22

## Audit
- Unpacked `BELA_CVPR_2027_RESEARCH_RULES_CURSOR_20261022.zip`
- Offline: `unittest` 6/6 OK; `schedule_audit.py` 12 arms PASS
- Cursor rule installed: `.cursor/rules/bela_cvpr_research.mdc`

## Reuse (do not re-run)
| Prior arm | Maps to | Tasks |
|---|---|---|
| `F1_COVER_RDRIFT` (172/90/32 RDRIFT) | **A0** | 6/6 |
| F0 QATTN / F2 K160 / F3 K128 | prior G1 entry study | kept; **not** T1 matrix |

## This round GPU queue (T1 priority)
- **A2** 172×4+120×8+20×20 COVER+RDRIFT — 6-task ACC
- **A3** 172×4+130×8+16×20 COVER+RDRIFT — 6-task ACC
- A1/A4 configs written, **NOT queued** until A2/A3 decide (per DECISION)
- T2/T3 (B*/C*) — NOT_RUN this round

## Task set (user)
`gqa, pope, mme, mmbench_en_dev, scienceqa_img, textvqa_val_ocr` (R4 ⊂ 6)
