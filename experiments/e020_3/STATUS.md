# E020.3 Status

Updated: 2026-10-18T19:27+08:00

## Wiring

- Package: `experiments/e020_3` (41 unit tests OK; 1 skip).
- Runtime hook: `e018_stage2_support` → `RecyclingState.compress` + post-gather `inject_visual`.
- Arms: 79 `E0203_*` registered; wave-1 = 38 (A+B+D, ORIGINAL final).
- Smoke: A off-path GQA limit=2 OK; B both-event merge GQA limit=2 OK after INC-001/002.

## Running

| Queue | GPUs | Jobs | PID file |
|---|---|---|---|
| `full_e0203_wave1` ACC | 0,2,4 | 152 (38×4) | `state/acc_queue.pid` |
| TTFT anchors | 6 | 9 arms sequential | `state/ttft_runner.pid` |

## Not yet in this wave

- Blocks C/E/F/G/H ACC
- Cheap final selectors (POOLED_QK / KEY_ENERGY / RANDOM)
- Full 12h confirmation panel

## Denoms (R4)

gqa 61.9 / pope 85.8 / mme 1852 / mmbench_en_dev 64.1
