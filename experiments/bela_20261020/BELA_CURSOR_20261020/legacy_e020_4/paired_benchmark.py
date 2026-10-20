"""Interleaved ABBA real first-token benchmark. No synthetic TTFT fallback."""
import time
import numpy as np


def summarize_blocks(block_ms, resamples=2000, seed=17):
    a = np.asarray(block_ms, dtype=float)
    if a.ndim != 2 or a.shape[1] != 4 or len(a) < 2 or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError('positive [blocks, A B B A] timings required')
    baseline, candidate = a[:, [0, 3]].mean(1), a[:, [1, 2]].mean(1)
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError('positive bootstrap count required')
    rng = np.random.default_rng(seed)
    savings = np.empty(resamples)
    for i in range(resamples):
        rows = rng.integers(0, len(a), len(a))
        savings[i] = 1 - candidate[rows].mean()/baseline[rows].mean()
    return dict(blocks=len(a), baseline_mean_ms=float(baseline.mean()),
        candidate_mean_ms=float(candidate.mean()),
        baseline_p50_ms=float(np.median(a[:, [0,3]])),
        candidate_p50_ms=float(np.median(a[:, [1,2]])),
        baseline_p90_ms=float(np.quantile(a[:, [0,3]], .9)),
        candidate_p90_ms=float(np.quantile(a[:, [1,2]], .9)),
        relative_saving=float(1-candidate.mean()/baseline.mean()),
        bootstrap_saving_ci95=np.quantile(savings, [.025, .975]).tolist(),
        scope='paired timing uncertainty; not accuracy or cross-device uncertainty')


def benchmark(baseline, candidate, inputs, warmup=8, repeats=256):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required')
    if not inputs or warmup < 0 or repeats < 2:
        raise ValueError('invalid input/repetition counts')
    # Each callable must start a NEW request, reset KV, include preprocessing
    # under the chosen boundary, and return only after token 1 is available.
    for i in range(warmup):
        baseline(inputs[i % len(inputs)])
        candidate(inputs[i % len(inputs)])
    rows, cuda_rows = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(repeats):
        sample = inputs[i % len(inputs)]
        wall, gpu = [], []
        # Reverse middle labels on alternating blocks to balance which path
        # runs first over time; canonicalize back to A B B A for analysis.
        order = [baseline, candidate, candidate, baseline] if i % 2 == 0 else [candidate, baseline, baseline, candidate]
        for fn in order:
            torch.cuda.synchronize()
            before = time.perf_counter()
            start.record()
            fn(sample)
            end.record()
            end.synchronize()
            wall.append((time.perf_counter()-before)*1000)
            gpu.append(start.elapsed_time(end))
        if i % 2:
            wall, gpu = [wall[j] for j in (1,0,3,2)], [gpu[j] for j in (1,0,3,2)]
        rows.append(wall)
        cuda_rows.append(gpu)
    return dict(wall=summarize_blocks(rows), cuda=summarize_blocks(cuda_rows),
                raw_wall_abba_ms=rows, raw_cuda_abba_ms=cuda_rows)
