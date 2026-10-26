# Fixed32 bottleneck hunt (2026-10-26)

## Step 1 — where it stalls

Instrument E3 Fixed32 (`profile_e3_fixed32.py`, n=16, per-layer CUDA events):

| Signal | Result |
|--------|--------|
| Early layers (KV fat) vs tail (KV short) | **flat** ~1.22 vs ~1.19 ms/layer |
| `scrub_all` (all-layer KV→75) vs `layer_local` | Fixed32 **unchanged** (~1207 vs ~1218) |
| `cache_positions` torch.cat ×32/step | ~0.86 ms/step (~27 ms/Fixed32) |
| Outside layers (python/embed) | ~1.6 ms/step |
| vs STAR fair-256 | STAR Fixed32 ~944; E3 ~1204 → ~10 ms/token |

**Conclusion:** limitation is **not** “pruned but length kept”. Decode is **not KV-length bound** on this host. Cost is roughly **constant per decoder layer** + HF/python overhead. STAR also physically shortens hidden; shallow KV can stay longer too.

## Step 2 — first optimizations (numerically same RoPE)

1. Skip decode-time `cache_positions` cat (no further prune).
2. Pass `_e018_rope_seq_len` so SDPA/eager RoPE avoids `position_ids.max().item()` **CUDA sync every layer**.

Remeasure: `logs/bela_scrub_ab_20261026/` style fair-256 n=64 after patch.
