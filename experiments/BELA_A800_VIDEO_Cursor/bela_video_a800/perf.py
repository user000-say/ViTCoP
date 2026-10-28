"""Reusable single-GPU inference timing wrapper; no inference is performed here.

Call from production model runner with the already-prepared input and inference
callback, and keep every accuracy arm on the same GPU/precision/backend.
"""
from __future__ import annotations
import statistics,time
import torch

@torch.inference_mode()
def measure_inference(fn, *, device, warmup=10, measured=50):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Timing requires actual CUDA GPU")
    if warmup<1 or measured<2:raise ValueError("insufficient measurement repetitions")
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples=[]
    for _ in range(measured):
        torch.cuda.synchronize(device)
        start=time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter()-start)*1000)
    s=sorted(samples)
    return {
        "ms_median":statistics.median(s),
        "ms_p90":s[min(len(s)-1, int(.9*len(s)))],
        "ms_mean":statistics.mean(s),
        "peak_allocated_gib":torch.cuda.max_memory_allocated(device)/(1024**3),
        "warmup":warmup,
        "measured":measured,
        "timing_scope":"callback; preprocessing intentionally EXCLUDED if not in fn",
        "warning":"Timing wrapper cannot distinguish TTFT vs full generation; fn defines scope",
    }
