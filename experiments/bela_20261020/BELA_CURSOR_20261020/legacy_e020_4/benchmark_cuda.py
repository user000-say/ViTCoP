"""Time a supplied REAL single-input pipeline. No CPU timing passed as CUDA."""
import statistics
import time


def benchmark(function, inputs, warmup=20, repeats=1024):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no simulated TTFT fallback")
    if not inputs or warmup < 0 or repeats < 2:
        raise ValueError("nonempty real inputs, warmup>=0, repeats>=2 required")
    # Caller must reset per-input KV/cache and return after producing token 1.
    for i in range(warmup):
        function(inputs[i % len(inputs)])
    torch.cuda.synchronize()
    cuda_ms, wall_ms = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        begin.record()
        function(inputs[i % len(inputs)])
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - start) * 1000.)
        cuda_ms.append(begin.elapsed_time(end))
    def summarize(values):
        ordered = sorted(values)
        return dict(mean=statistics.mean(values), median=statistics.median(values),
                    p90=ordered[min(len(ordered)-1, int(.9 * len(ordered)))])
    return dict(n=repeats, cuda_ms=summarize(cuda_ms), wall_ms=summarize(wall_ms),
                raw_cuda_ms=cuda_ms, raw_wall_ms=wall_ms)
