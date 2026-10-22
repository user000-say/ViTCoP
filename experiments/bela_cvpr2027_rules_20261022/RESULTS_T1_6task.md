# T1 6-task ACC results (2026-10-22)

Protocol: fixed COVER@172 + RDRIFT, W2048, only mid/deep keep changes.  
Tasks: gqa, pope, mme, mmbench_en_dev, scienceqa_img, textvqa_val_ocr.  
Dense refs: GQA 61.9, POPE F1 85.8, MME 1852, MMB 64.1, SciQA 69.51, TextVQA-OCR 58.20.  
A0 reused from prior F1 (identical schedule).

| Arm | schedule | GQA | POPE F1 | MME | MMB | SciQA | TextVQA | R4 | mean6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 172/90/32 | 57.84/93.44% | 83.31/97.10% | 1767.8/95.46% | 62.20/97.03% | 68.96/99.21% | 55.97/96.16% | **95.76** | **96.40** |
| A2 | 172/120/20 | 57.62/93.08% | 85.44/99.58% | 1774.9/95.83% | 63.14/98.51% | 68.67/98.79% | 54.89/94.32% | **96.75** | **96.68** |
| A3 | 172/130/16 | 57.35/92.66% | 85.68/99.86% | 1775.9/95.89% | 62.80/97.97% | 68.67/98.79% | 54.88/94.30% | **96.59** | **96.58** |

**T1 read:** A2/A3 beat A0 on R4 (+0.99 / +0.83), mainly POPE+MMB+MME; GQA and TextVQA slightly down. Supports “less deep-32, more mid” at fixed W2048. A1/A4 / T2/T3 still NOT_RUN.
