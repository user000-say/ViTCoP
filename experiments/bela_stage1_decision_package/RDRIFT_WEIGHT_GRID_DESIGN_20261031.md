# RDRIFT weight grid — after schedule A/B/C

## Operator (unchanged)

First Stage2 event uses E020 `rank_drift`; later event stays Q_ATTN.

```
score = w * rank_now + (1-w) * (rank_now - rank_old)
```

Env: `E020_CURRENT_WEIGHT=w`. Config field: `method.stage2.rdrift_current_weight`.

## Weight pairs (w, 1−w)

| label | w | 1−w | 77.8% | 94.4% | 88.9% |
|---|---:|---:|---|---|---|
| W20 | 0.2 | 0.8 | **run** | **run** | deferred |
| W40 | 0.4 | 0.6 | **run** | **run** | deferred |
| W60 | 0.6 | 0.4 | hist (S1/ABC) | hist (L2 `64→17→10`) | deferred |
| W80 | 0.8 | 0.2 | **run** | **run** | deferred |

Phase-1 queue = **6 arms** (3 new weights × {77.8%, 94.4%}). STAR T64 / 88.9% later.

## Compression × schedule (LLaVA-1.5-7B, N=576, L=32, lengths 12/8/12)

Schedule letter = ABC winner (same shape scaled). Cuts after layers 12/20.

| rate | T / W | A counts | B counts | C counts |
|---|---|---|---|---|
| 77.8% | 128 / 4096 | 256→74→36 | 256→80→32 | 256→68→40 |
| 88.9% | 64 / 2048 | 128→37→18 | 128→40→16 | 128→34→20 |
| 94.4% | 32 / 1024 | 64→17→10 | 64→20→8 | 64→17→10 |

Stage1: Dual COVER + QA B8 (controlled; only RDRIFT `w` varies).

## Eval suites (same scorer / Dense refs)

- **R4**: gqa, pope, mme, mmbench_en_dev  
- **R6**: R4 + scienceqa_img + textvqa_val_ocr  
- **R10**: R6 + mmmu_val + ai2d + vizwiz_vqa_val + mmvet (~10; R9 + MM-Vet)

Queue order: all arms finish R4 → then R6 remainder → then R10 remainder.

## Reuse

- Winner schedule @ 77.8% / W06: reuse ABC GQA/POPE/MME when present; if winner=A also reuse prior S1 six-task metrics where available.
- Do not claim cross-rate equality; report per-rate tables.
