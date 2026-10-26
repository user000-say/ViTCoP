# STAR-Pro X0 T=64 status (2026-10-23)

## Audit
- Source: `https://github.com/EasonAI-5589/starpro` → `third_party/starpro`
- Overlay: LLaVA `@c121f043` + `install_overlay.sh` → `third_party/LLaVA-starpro`
- Isolated conda env: `/home/xiangyue/envs/starpro` (does **not** modify vitcopreback)
- Config: `METHOD=star_pro` `T=64` → Stage1 QR pool **128**, progressive `[(12,37),(20,18)]` avg **64**
- Scorer path: lmms-eval 9 tasks + `elva_scores.py` Dense refs (same as BELA R9)
- Smoke: generate OK (`SMOKE_OK`), schedule verified

## Difference vs BELA host
| | STAR-Pro | BELA/ELVA |
|---|---|---|
| Env | `/home/xiangyue/envs/starpro` | vitcopreback |
| LLaVA tree | `third_party/LLaVA-starpro` | `LLaVA/` + vitcop patches |
| Stage1 | pivoted QR (2T) | dual COVER |
| Stage2 | text-attn topk progressive | RDRIFT / Q-Attn |
| W proxy | layer-average T=64 | W2048 token-layers |

## Queue (started 2026-10-23 ~01:14 CST)
- R9 ACC: **RUNNING** on GPUs 0/2/4/7 (`launch_r9_acc.py`); first wave gqa/pope/mme/mmbench
- TTFT/Fixed32: watcher waits for ACC drain then runs `bench_ttft.py` on GPU 0
- Smoke: generate OK + pope limit=2 lmms path OK (`SMOKE_OK`)
- Full R9 metrics: **NOT_YET** (no completed `metrics.json` for full splits)

## Fair-256 latency lock (2026-10-26)

STAR T64 on same `speed_256.jsonl` as BELA E3, n=256, warm=8:
- STAR: TTFT mean≈147.5 / Fixed32 mean≈935 (see `logs/starpro_x0_t64/fair256_ttft/STARPRO_T64/`)
- E3 (n=1024, prior): TTFT≈123.0 / Fixed32≈1203.7
- Matched gap remains: E3 faster TTFT (~−24ms), slower Fixed32 (~+269ms). Old synthetic STAR F32≈939 ≈ fair256 — protocol noise was small for STAR; cross-method gap is real.
