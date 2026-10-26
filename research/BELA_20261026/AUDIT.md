# AUDIT — live repo map (2026-10-26)

Status: **in progress**. Paths verified on this server.

## Checkpoint / host

| Item | Value | Evidence |
|------|-------|----------|
| Model | LLaVA-1.5-7B snapshot `4481d270…` | fair-256 `models_llava`, STAR/CD paths |
| BELA host | E014 arm `E018_COVER_TWO32` + ELVA_R3 COVER+RDRIFT | `experiments/elva_r3/apply_llava.py` |
| Stage1 | dual COVER, payload pen, K0 from schedule | `native_cover` / `llava_stage1` |
| Stage2 first | RDRIFT @ first event only | `e018_stage2_support._pick` + E020 |
| Stage2 later | Q_ATTN | same |
| Cache policy default | `layer_local` (overlay hardcodes) | `apply_llava.py:E018_CACHE_POLICY` |
| Attn | SDPA required for E018 | `benchmark_llava_e2e_anygpu.py` |
| use_cache | True in latency/ACC generate | benches / lmms adapters |

## Schedules (0-based E018 events)

| Arm | counts×lengths | E018_EVENTS (post-block) |
|-----|----------------|--------------------------|
| E3 | 172×4+120×8+24×10+16×10 | `3:120,11:24,21:16` (via lengths) |
| L2 | 64×12+17×8+10×12 | `11:17,19:10` |

## Fixed32 finding (P0.3)

See `FINDINGS_FIXED32.md`. **OBSERVATION:** decode host sync (`.item()` + cache_pos cat) cost ~267 ms Fixed32; not progressive-schedule FLOPs. After fix, microbench Fixed32 **939.8** (was 1206.6). Official fair-256 remeasure running on GPU0.

## CDPruner

| Item | Path |
|------|------|
| Source | `third_party/CDPruner` |
| Adapter | `experiments/cdpruner_x0/` |
| K64/K32 ACC | `logs/cdpruner_x0/full/` complete |
| Fair-256 latency | GPU2/4 running `bench_fair256.py` |

## Open P0/P1

- [ ] Official E3 fair-256 after decode opt (GPU0)
- [ ] CD32/CD64 fair-256 (GPU2/4)
- [ ] E3 pope ACC regression after opt (GPU7)
- [ ] N32-FLAT / C32-REPLAY wiring (M2/M3) — not started
- [ ] Instrumentation TraceRecord (M1) — partial via KV profile only
