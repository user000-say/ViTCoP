"""Source-bound COVER with explicit capacity and no top-k fallback.

The real project supplies one thin callback with the standard keyword
signature. This module never invents the historical COVER formula.
"""
import hashlib
import importlib
import inspect
from pathlib import Path
import numpy as np
from .budget import integer
from .evidence import indices


FAMILIES = ("legacy_native", "parameterized_native", "legacy_pool172_then_topk", "protected_coverage")


def origin(family):
    return ("r3_protected_control" if family == "protected_coverage" else
            "none" if family == "none" else "historical_source_bound")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_fingerprint(spec):
    sources = spec.get("source_files", {})
    if not sources or not isinstance(sources, dict):
        raise RuntimeError("BLOCKED_SOURCE: bind original source files and the thin callback")
    actual = {str(Path(path).resolve()): expected for path, expected in sources.items()}
    for path, expected in actual.items():
        if not isinstance(expected, str) or len(expected) != 64 or file_sha(path) != expected:
            raise RuntimeError("BLOCKED_SOURCE: changed or invalid COVER source fingerprint")
    return hashlib.sha256(repr(sorted(actual.items())).encode()).hexdigest()


def check_sources(spec, callback):
    fingerprint = spec_fingerprint(spec)
    source = inspect.getsourcefile(callback)
    files = {str(Path(path).resolve()) for path in spec["source_files"]}
    if source is None or str(Path(source).resolve()) not in files:
        raise RuntimeError("BLOCKED_SOURCE: callback itself must be included in source_files")
    return fingerprint


def load_cover(spec):
    entry = spec.get("entrypoint", "")
    if not isinstance(entry, str) or entry.count(":") != 1:
        raise RuntimeError("BLOCKED_SOURCE: set a real module:function COVER entrypoint")
    module, name = entry.split(":")
    callback = getattr(importlib.import_module(module), name)
    return NativeCover(callback, spec)


def checked_keep(value, n, k):
    if isinstance(value, np.ndarray):
        return indices(value, n, k)
    if not hasattr(value, "device"):
        raise ValueError("native COVER must return NumPy or PyTorch local indices, not a list")
    import torch
    if (not isinstance(value, torch.Tensor) or value.shape != (k,) or value.dtype != torch.long
            or value.unique().numel() != k or int(value.min()) < 0 or int(value.max()) >= n):
        raise ValueError("native COVER must return unique device-local long active-row indices")
    return value


class NativeCover:
    def __init__(self, callback, spec):
        self.callback, self.spec = callback, dict(spec)
        if spec.get("family") not in FAMILIES:
            raise ValueError("explicit COVER implementation family required")
        self.family = spec["family"]
        self.supported = tuple(integer(k, "supported K", 1) for k in spec.get("supported_k", ()))
        if not self.supported or len(set(self.supported)) != len(self.supported):
            raise ValueError("declare unique verified supported K values")
        # Handoff COVER@K0 + sprint B1 K32 (slots bound separately per K).
        if self.family == "legacy_native" and set(self.supported) - {32, 64, 72, 80, 96, 128, 160, 172, 256}:
            raise ValueError("legacy_native COVER only allows K0 in {32,64,72,80,96,128,160,172,256}")
        if self.family == "legacy_native" and 172 not in self.supported:
            raise ValueError("legacy_native must still declare historical K172")
        if self.family == "legacy_pool172_then_topk" and any(k >= 172 for k in self.supported):
            raise ValueError("pooled route is an explicitly separate smaller-K method")
        self.source_sha = check_sources(spec, callback)
        self.events = []
        self.last_native_keep = self.last_requested_keep = None

    def __call__(self, *, metric_features, scores, patch_ids, k, context=None):
        k = integer(k, "requested K", 1)
        if k not in self.supported:
            raise RuntimeError(f"BLOCKED_CAPACITY: {self.family} has no verified K={k} implementation")
        n = len(scores)
        if len(metric_features) != n or len(patch_ids) != n or k > n:
            raise ValueError("aligned full active candidate pool required")
        native_k = 172 if self.family == "legacy_pool172_then_topk" else k
        if native_k > n:
            raise ValueError("native pool exceeds current candidate universe")
        slots = self.spec.get("replacement_slots", {}).get(str(native_k))
        local_context = dict(context or {})
        if self.family == "protected_coverage":
            from .low_budget_cover import POLICY
            slots = local_context.get("coverage_slots", slots)
            if slots not in self.spec.get("allowed_slots", ()) or self.spec.get("coverage_policy") != POLICY:
                raise RuntimeError("BLOCKED_SOURCE: unverified protected coverage configuration")
            local_context["coverage_policy"] = POLICY
        if type(slots) is not int or not 0 <= slots <= native_k:
            raise RuntimeError("BLOCKED_SOURCE: bind actual COVER replacement slots for this native K")
        local_context.update(cover_family=self.family, requested_k=k, native_k=native_k,
                             replacement_slots=slots)
        keep = self.callback(metric_features=metric_features, scores=scores,
                             patch_ids=patch_ids, k=native_k, context=local_context)
        keep = checked_keep(keep, n, native_k)
        self.last_native_keep = keep
        if self.family == "legacy_pool172_then_topk":
            if isinstance(keep, np.ndarray):
                keep = keep[np.argsort(-scores[keep], kind="stable")[:k]]
                keep = np.sort(keep)
            else:
                import torch
                chosen = torch.argsort(scores.index_select(0, keep), descending=True, stable=True)[:k]
                keep = keep.index_select(0, chosen).sort().values
        keep = checked_keep(keep, n, k)
        self.last_requested_keep = keep
        self.events.append(dict(family=self.family, source_sha=self.source_sha,
                                requested_k=k, native_k=native_k, replacement_slots=slots,
                                cover_calls=1, native_calls=int(self.family != "protected_coverage"),
                                origin=origin(self.family), fallback=False, candidate_count=n))
        return keep


def select_checked(pen, pen_scores, ids, config, *, early=None, early_scores=None,
                   early_ids=None, native_cover=None, context=None, backend="numpy"):
    """Actual selection using R2 kernels, with an enforceable COVER call audit.

    Keep the audit per sample. Export tensors/IDs only outside timed paths.
    No automatic native->pooled or COVER->top-k conversion is permitted.
    """
    stage = config["method"]["stage1"]
    k = config["method"]["schedule"]["counts"][0]
    family = stage["cover_family"]
    covered = family != "none"
    if covered and (native_cover is None or native_cover.family != family):
        raise RuntimeError("BLOCKED_SOURCE: required original COVER family is not bound")
    previous = len(native_cover.events) if native_cover is not None else 0
    if backend == "numpy":
        from .evidence import select_stage1
    elif backend == "torch":
        from .torch_ops import select_stage1
    else:
        raise ValueError("backend must be numpy or torch")
    local_context = dict(context or {})
    if covered and family == "protected_coverage":
        from .low_budget_cover import POLICY
        if stage.get("coverage_policy") != POLICY or type(stage.get("coverage_slots")) is not int:
            raise ValueError("explicit protected coverage slots and policy required in method fingerprint")
        local_context["coverage_slots"] = stage["coverage_slots"]
    kwargs = dict(guide=stage["guide"], beta=stage.get("beta") or 0., early=early,
                  early_scores=early_scores, early_ids=early_ids,
                  use_cover=covered, cover=native_cover if covered else None,
                  context=local_context, score_guide=stage.get("score_guide", "same"))
    result = select_stage1(pen, pen_scores, ids, k, **kwargs)
    events = native_cover.events[previous:] if native_cover is not None else []
    if len(events) != int(covered):
        raise RuntimeError("COVER native callback count disagrees with the requested method")
    result["stage1_audit"] = dict(cover_family=family, requested_k=k, cover_calls=len(events),
                                 native_calls=sum(e["native_calls"] for e in events), origin=origin(family),
                                 cover_source_sha=native_cover.source_sha if covered else None,
                                 native_k=events[0]["native_k"] if covered else None,
                                 replacement_slots=events[0]["replacement_slots"] if covered else 0,
                                 fallback=False, payload="pen")
    return result
