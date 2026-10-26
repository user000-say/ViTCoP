# CDPruner X0 results (official source, 6-task + TTFT)

Protocol: lmms-eval same Dense refs as BELA; TTFT/Fixed32 warm=10 meas=50 synthetic 336 (same shape as STAR bench).

## Accuracy

| K | mean6 | R4 | GQA | POPE | MME | MMB | Sci | TVQA |
|---|------:|---:|----:|-----:|----:|----:|----:|-----:|
| 64 | 96.01 | 96.06 | 58.66 | 87.53 | 1712.0 | 60.91 | 67.38 | 55.23 |
| 32 | 94.41 | 94.53 | 57.00 | 87.67 | 1677.4 | 59.79 | 67.53 | 53.08 |

## Latency (ms)

| K | TTFT | Fixed32 |
|---|-----:|--------:|
| 64 | 103.8 | 891.1 |
| 32 | 98.3 | 883.5 |

## Context (same machine / similar protocol)

- E3 W2048 mean6 96.85; TTFT ~123 / Fixed32 ~1204 (BELA fair path)
- L2 W1024 mean6 93.56; STAR T32 mean6 93.53
- CDPruner K is fixed retain-all-layers, not BELA W=ΣK·L
