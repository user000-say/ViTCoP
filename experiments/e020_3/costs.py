"""Token-work accounting and explicit latency hypotheses, not fake GPU data."""
import math
import numpy as np


PROFILES = {
    "SHIFT": [(3, 116), (13, 32)],
    "SHIFT_K108": [(3, 108), (13, 32)],
    "SHIFT_K100": [(3, 100), (13, 32)],
    "SHIFT_K132": [(3, 132), (13, 32)],
    "FINAL11": [(3, 116), (11, 32)],
    "FINAL15_WMATCH": [(3, 102), (15, 32)],
}


def token_trace(layers, initial, events):
    if not isinstance(layers, int) or not isinstance(initial, int) or layers < 2 or initial < 1:
        raise ValueError("integer model shape required")
    current, previous, result = initial, 0, []
    for done, keep in events:
        if not isinstance(done, int) or not isinstance(keep, int):
            raise ValueError("use COMPLETED block counts, not ambiguous layer IDs")
        if not previous < done < layers or not 0 < keep <= current:
            raise ValueError("invalid pruning boundary/count")
        result.extend([current] * (done - previous))
        current, previous = keep, done
    result.extend([current] * (layers - previous))
    return np.asarray(result, dtype=np.int64)


def work_report(events, layers=32, initial=172, text_tokens=None):
    t = token_trace(layers, initial, events)
    result = {"visual_token_layers": int(t.sum()),
              "visual_square_proxy": int(np.square(t).sum()), "trace": t.tolist()}
    if text_tokens is not None:
        if not isinstance(text_tokens, int) or text_tokens < 1:
            raise ValueError("positive text-token count required")
        result["sequence_square_proxy"] = int(np.square(t + text_tokens).sum())
    return result


def r4(metrics):
    denominators = dict(GQA=61.9, POPE=85.8, MME=1852., MMBench=64.1)
    if not set(denominators) <= set(metrics):
        raise ValueError("need named GQA/POPE/MME/MMBench metrics")
    values = [float(metrics[k]) / v for k, v in denominators.items()]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite metric")
    return 25 * sum(values)


def amdahl_hypothesis(total_ms, fixed_ms, decoder_work_ratio, added_ms):
    """Illustrative stage decomposition only, never a measured TTFT."""
    if not all(math.isfinite(x) for x in (total_ms, fixed_ms, decoder_work_ratio, added_ms)):
        raise ValueError("finite inputs required")
    if not 0 <= fixed_ms <= total_ms or decoder_work_ratio < 0 or added_ms < 0:
        raise ValueError("invalid decomposition")
    estimate = fixed_ms + (total_ms - fixed_ms) * decoder_work_ratio + added_ms
    return {"hypothetical_only": True, "estimated_ms": estimate,
            "savings_ms": total_ms - estimate}


def pareto_ids(records):
    """Compare only rows from the SAME runtime/split/scorer provenance."""
    if not records:
        return []
    if len({r["provenance_sha256"] for r in records}) != 1:
        raise ValueError("mixed measurement conditions")
    for r in records:
        if not math.isfinite(r["R4"]) or not math.isfinite(r["ttft_ms"]) or r["ttft_ms"] <= 0:
            raise ValueError("missing real quality/latency")
    return [r["id"] for r in records if not any(
        s["R4"] >= r["R4"] and s["ttft_ms"] <= r["ttft_ms"] and
        (s["R4"] > r["R4"] or s["ttft_ms"] < r["ttft_ms"])
        for s in records)]
