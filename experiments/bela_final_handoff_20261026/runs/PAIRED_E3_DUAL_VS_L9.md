# Paired E3 Dual vs L9-only (full predictions on server)

## GQA (n=12578)

| | count |
|---|---:|
| both correct | 6793 |
| Dual-only correct | 472 |
| L9-only correct | 587 |
| both wrong | 4726 |
| Dual ACC | 57.76 |
| L9 ACC | 58.67 |
| Oracle union | **62.43** |

**INFERENCE:** Entry identity is complementary; bounded admission of L9-disagreement into Dual is a falsifiable recovery path (handoff A). Not yet proof that Stage2 exchange is required.

## POPE (joined n=3000 sample rows in jsonl; treat as diagnostic)

| | count |
|---|---:|
| Dual-only | 49 |
| L9-only | 57 |
| Oracle union ACC | 90.13 vs Dual 88.23 |

Complementarity smaller than GQA; do not claim POPE bottleneck equals GQA.
