"""E018 contracts. No torch import: usable by registry/training/CPU tests."""
from dataclasses import dataclass
import hashlib
import json
import math
import os

SCHEMA = "e018-continuation-v1"
FEATURE_NAMES = (
    "layer_fraction", "current_fraction", "keep_fraction", "drop_fraction",
    "delta_work", "endpoint_fraction", "prior_events", "raw_mass_mean",
    "raw_mass_min", "response_entropy", "drop_response_mass", "drop_raw_mass",
    "drop_score_max", "boundary_gap", "drop_redundancy", "keep_redundancy",
    "response_drift", "has_history", "receiver_cosine_drift", "receiver_norm",
    "drop_hidden_norm", "keep_hidden_norm", "added_fraction", "drop_added_fraction",
    "drop_added_mass", "drop_cls_mass", "keep_cls_mass", "s1_exchange_fraction",
    "removed_signature_coherence", "cache_scrub_all",
) + tuple("receiver_sketch_%d" % i for i in range(8))


@dataclass(frozen=True)
class E018Config:
    enabled: bool = False
    mode: str = "STATIC"
    events: tuple = ((4, 102), (12, 32))
    selector: str = "Q_ATTN"
    cache_policy: str = "layer_local"
    controller: str = ""
    lam: float = 0.0
    protocol_id: str = ""
    trace: str = ""
    max_work: int = 0
    proxy_threshold: float = 0.02

    @classmethod
    def from_env(cls):
        events = tuple(tuple(map(int, x.split(":"))) for x in
                       os.environ.get("E018_EVENTS", "4:102,12:32").split(",") if x)
        return cls(os.environ.get("E018_ENABLE", "False") == "True",
                   os.environ.get("E018_MODE", "STATIC").upper(), events,
                   os.environ.get("E018_SELECTOR", "Q_ATTN").upper(),
                   os.environ.get("E018_CACHE_POLICY", "layer_local"),
                   os.environ.get("E018_CONTROLLER", ""),
                   float(os.environ.get("E018_LAMBDA", "0")),
                   os.environ.get("E018_PROTOCOL_ID", ""),
                   os.environ.get("E018_TRACE", ""),
                   int(os.environ.get("E018_MAX_WORK", "0")),
                   float(os.environ.get("E018_PROXY_THRESHOLD", "0.02")))

    def validate(self, layers, initial):
        if self.mode not in {"STATIC", "COLLECT", "ONLINE", "PROXY"}:
            raise ValueError("E018_MODE must be STATIC/COLLECT/ONLINE/PROXY")
        if self.selector not in {"Q_ATTN", "QED", "RANDOM"}:
            raise ValueError("E018_SELECTOR must be Q_ATTN/QED/RANDOM")
        if self.cache_policy not in {"layer_local", "scrub_all"}:
            raise ValueError("Unknown cache policy")
        if len(self.events) > 2 or (not self.events and self.mode != "STATIC"):
            raise ValueError("E018 supports up to two checkpoints; empty STATIC is the dense identity control")
        prev_layer, prev_k = -1, initial
        for layer, k in self.events:
            if not 0 <= layer < layers - 1 or layer <= prev_layer or not 0 < k < prev_k:
                raise ValueError("Require increasing post-block indices and decreasing positive K")
            prev_layer, prev_k = layer, k
        if not math.isfinite(self.lam) or self.lam < 0 or self.max_work < 0:
            raise ValueError("Invalid lambda/work cap")
        if not math.isfinite(self.proxy_threshold) or self.proxy_threshold < 0:
            raise ValueError("Invalid mass threshold")
        if self.mode in {"ONLINE", "COLLECT"} and not self.protocol_id:
            raise ValueError("Set E018_PROTOCOL_ID to your frozen weights/source/data protocol")
        if self.mode == "ONLINE" and not self.controller:
            raise ValueError("ONLINE requires a trained and calibrated controller JSON")
        if self.max_work and token_work(initial, self.events, layers) > self.max_work:
            raise ValueError("Work cap is below even the all-retire-now schedule")

    def fingerprint(self, initial, layers, stage1):
        obj = dict(schema=SCHEMA, protocol_id=self.protocol_id,
                   events=self.events, selector=self.selector,
                   cache_policy=self.cache_policy, initial=initial, layers=layers,
                   stage1=stage1)
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def token_work(initial, events, layers=32):
    """Visual token-block sum. Pruning at index t happens AFTER block t."""
    schedule = dict(events)
    n, work = initial, 0
    for layer in range(layers):
        work += n
        if layer in schedule:
            if not 0 < schedule[layer] <= n:
                raise ValueError("Schedule cannot resurrect/increase tokens")
            n = schedule[layer]
    return work


def group_split(image_id, seed=20261014):
    """All questions and counterfactual rows of an image share a split."""
    x = int(hashlib.sha256((str(seed) + ":" + str(image_id)).encode()).hexdigest()[:8], 16) % 100
    return "train" if x < 70 else "calibration" if x < 85 else "test"


def conformal_upper(residuals, alpha=0.1):
    if not 0 < alpha < 1 or not residuals:
        raise ValueError("Need residuals and alpha in (0,1)")
    k = math.ceil((len(residuals) + 1) * (1 - alpha))
    if k > len(residuals):
        raise ValueError("Too few calibration image groups for requested alpha")
    return sorted(residuals)[k - 1]
