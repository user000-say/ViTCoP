# Fixed32 bottleneck — Step 1 finding + Step 2 first fix

**OBSERVATION (matched fair-256 style bench, n=64, same panel/script):**

| Config | TTFT ms | Fixed32 ms |
|--------|--------:|-----------:|
| E3 `layer_local` before opt | 125.9 | **1206.6** |
| E3 after decode host opt | 127.9 | **939.8** |
| Δ | +2.0 | **−266.9** |
| STAR T64 fair-256 (n=256) | 149.9 | 944.0 |

**Root cause (INFERENCE, code-backed):**
1. Per-layer decode time was **flat** (early fat KV ≈ tail short KV) → not “fake prune / length kept”.
2. `scrub_all` shortened all-layer KV to final K but Fixed32 did **not** drop → not KV-FLOP bound.
3. Production decode did `position_ids.max().item()` **every layer** (CUDA sync) + `torch.cat` on `cache_positions` every layer.
4. Fix: pass `_e018_rope_seq_len` (no `.item()` sync); skip decode `cache_positions` cat (no further prune).

**Status:** First host fix landed in `e018_stage2_support.py` + `vitcop_modeling_llama.py`. Official fair-256 remeasure + ACC regression queued. Do not claim paper efficiency until official harness + ACC gate pass.
