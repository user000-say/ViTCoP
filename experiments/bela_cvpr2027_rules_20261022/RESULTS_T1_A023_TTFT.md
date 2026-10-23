# T1 A0/A2/A3 fair-256 TTFT / Fixed32 (2026-10-23)

Panel: `e014_latency_fair_256/panel/speed_256.jsonl` (n=1024), same bench as H1.  
V100-PCIE-32GB. Warmups=8.

| Arm | schedule | final K | TTFT mean | TTFT med | Fixed32 mean | Fixed32 med | status |
|---|---|---:|---:|---:|---:|---:|---|
| A0 (1st try, GPU7) | 172/90/32 | 32 | 229.0 | 205.9 | 2011.5 | 1994.2 | **INVALID** — contention (p90 F32=2420, max=3834); STAR-Pro still draining |
| **A2** | 172/120/20 | **20** | **122.5** | 120.9 | **1217.7** | 1193.3 | OK (GPU0) |
| **A3** | 172/130/16 | **16** | **120.3** | 119.3 | **1177.1** | 1176.2 | OK (GPU0) |
| H1 historical (A0-equiv) | 172/90/32 | 32 | 129.1 | 127.7 | 1327.8 | 1304.1 | prior clean run |

**Read:** A2/A3 Fixed32 still ~1180–1220, only ~100–150 ms below clean H1/A0 (~1328), **not** down to DivPrune/STAR ~900. Final-K 20/16 helps a little vs 32, but does not escape the ~1200 decode floor on this path. A0 clean rerun queued separately.
