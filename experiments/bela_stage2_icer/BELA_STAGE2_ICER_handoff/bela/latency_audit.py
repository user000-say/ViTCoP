"""Integration helpers for BELA timing. Call these around real model stages.

Important: these helpers do not magically split model internals. Insert them at
the real boundaries in the LLaVA code and record the raw output. For CUDA
asynchronous work, synchronize or use CUDA events. Do not mix TTFT with full
generation latency.
"""
from __future__ import annotations
import time, statistics
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import torch

@dataclass
class StageTimes:
    vision_encoder_ms: float = 0.0
    stage1_selector_ms: float = 0.0
    projector_ms: float = 0.0
    decoder_prefill_ms: float = 0.0
    stage2_selector_ms: float = 0.0
    gather_reindex_ms: float = 0.0
    first_decode_step_ms: float = 0.0
    full_generation_ms: float = 0.0
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0

def timed_cuda(fn, device="cuda"):
    """Return (result, elapsed_ms), measuring queued CUDA work on one device."""
    if not torch.cuda.is_available():
        t=time.perf_counter(); result=fn(); return result,(time.perf_counter()-t)*1000
    torch.cuda.synchronize(device)
    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    wall=time.perf_counter(); start.record()
    result=fn()
    end.record(); torch.cuda.synchronize(device)
    return result, {"cuda_ms": start.elapsed_time(end),
                    "wall_ms": (time.perf_counter()-wall)*1000}

def reset_peak_memory(device="cuda"):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

def memory_stats(device="cuda"):
    if not torch.cuda.is_available():
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    torch.cuda.synchronize(device)
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device)/1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device)/1024**2,
    }

def percentile(xs, p):
    xs=sorted(xs)
    if not xs: return None
    idx=min(len(xs)-1, max(0, int(round((len(xs)-1)*p))))
    return xs[idx]

def summarize_ms(xs):
    return {"n":len(xs), "median_ms":statistics.median(xs) if xs else None,
            "p90_ms":percentile(xs,.90), "mean_ms":statistics.mean(xs) if xs else None}
