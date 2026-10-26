# Preflight — R9 extras fill (2026-10-26)

## Inventory

| Package | Provides | Repo already owns |
|---------|----------|-------------------|
| BELA E014/E018 host | Stage1 COVER + Stage2 RDRIFT/Q_ATTN schedules | `e014_boot`, `elva_r3`, P1 configs |
| STAR-Pro x0 | official T schedule runner | `experiments/starpro_x0_t64` |
| CDPruner x0 | fixed-K DPP selector | `experiments/cdpruner_x0` + `third_party/CDPruner` |
| Scorer | R4/R8/R9 | `elva_scores.py` Dense refs frozen |

## Gap check

- Missing host patches: none for this fill (reuse prior ACC paths).
- CD `run_one_task.FULL_N` extended with `mmmu_val` / `ai2d` / `vizwiz_vqa_val` + `mmmu_acc` field.
- Skip regen: arms with existing `metrics.json` skipped.

## Already complete R9

- `A2_COVER_RDRIFT` (BELA)
- `STARPRO_T64`

## Pending extras only (`mmmu_val`, `ai2d`, `vizwiz_vqa_val`)

- BELA: E3, E1, E5, L2
- STAR: T32
- CD: K64, K32

## Algorithm identity

- BELA arms: original COVER+RDRIFT schedules (configs under `p1_arms/`).
- STAR / CD: official source runners (not transplants).

## Verdict

**Ready** to launch fill queue on GPUs 0/2/4/7. Full R9 tables after drain.
