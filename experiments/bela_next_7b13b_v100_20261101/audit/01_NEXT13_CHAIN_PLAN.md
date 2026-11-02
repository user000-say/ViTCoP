# NeXT-13B A0–A4 chain (after NeXT-7B 30 jobs)

## Handoff schedules (T160 / 94.4%, W=6400, cuts **L15/L30**, 40 layers)
| Arm | stage_k |
|---|---|
| A0 PAPER | 320→80→40 |
| A1 PAPER_VAR | 320→84→34 |
| A2 LATE | 320→72→52 |
| A3 LESS_S1 | 288→112→40 |
| A4 STRONG_LATE | 256→128→64 |

## Forward
Same overlay `bela_next_forward` as 7B: Dual COVER crop-wise + Stage2 Q_ATTN.
Cuts/K read from `arms.json` (`NEXT13_T160_*`). Dense path unchanged when `BELA_NEXT_V2_ENABLE=0`.

## Gate
1. Wait `QUEUE_DRAINED` on 7B queue (GPUs 4–7 free)
2. Smoke `scripts/smoke_next13_a0_a4.py` on GPU4 — must print exact traces + cuts [15,30]
3. Only then `run_next13_a0_a4_six.sh` (30 jobs, same 6 tasks)

## Launcher
`scripts/chain_next13_after_next7.sh` (nohup’d)
