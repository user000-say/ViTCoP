"""Paired end-to-end FIRST TOKEN timing; real VLM is a bound callback.

Input preparation belongs inside run_request. Dedicated measurement device.
No CUDA-event-only result is relabeled as user-observable wall TTFT.
"""
import math
import random
import time
import numpy as np


def paired_ttft(samples, methods, run_request, synchronize, *, rounds=3, warmup=3, seed=19):
    if len(methods) != 2 or methods[0] == methods[1] or not samples or type(rounds) is not int or rounds < 1:
        raise ValueError("two distinct methods, samples, positive rounds required")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be nonnegative")
    for i in range(warmup):
        for method in methods:
            run_request(samples[i % len(samples)], method)
    synchronize()
    rng, rows = random.Random(seed), []
    for repeat in range(rounds):
        order = list(range(len(samples)))
        rng.shuffle(order)
        for i in order:
            # Balanced order over rounds/input order, avoiding all-A then all-B.
            local = methods if (i+repeat) % 2 == 0 else methods[::-1]
            for method in local:
                synchronize()
                start = time.perf_counter_ns()
                first_token = run_request(samples[i], method)
                synchronize()
                elapsed = (time.perf_counter_ns()-start)/1e6
                if first_token is None or not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("callback must emit an actual first token")
                rows.append(dict(sample=i, repeat=repeat, method=method, wall_ttft_ms=elapsed,
                                 first_token=str(first_token)))
    return rows


def summarize(rows):
    output = {}
    for method in sorted(set(row["method"] for row in rows)):
        values = np.array([row["wall_ttft_ms"] for row in rows if row["method"] == method], float)
        if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("invalid timing rows")
        output[method] = dict(mean_ms=float(values.mean()), p50_ms=float(np.median(values)),
                              p90_ms=float(np.quantile(values, .9)), observations=len(values))
    return output
