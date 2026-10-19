"""Optional diagnostic signed-gain regressor for NEW transient-swap labels.

It is deliberately small. A working synthetic regression is not evidence
that VLM survival gains are predictable. No old E020 weights are accepted.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .counterfactual import ESTIMAND
from .evidence import finite, indices


def split_image(image_key, salt="elva-r2-frozen-split"):
    """Split by canonical decoded-image identity, not question/filename."""
    if not isinstance(image_key, str) or not image_key:
        raise ValueError("canonical image key required")
    value = int(hashlib.sha256((salt+image_key).encode()).hexdigest()[:16], 16) % 100
    return "train" if value < 70 else "validation" if value < 85 else "test"


def pair_features(token_features, common, group_a, group_b, *, begin, end, depth):
    """Features only from prompt and states available BEFORE this decision.

    Caller supplies aligned observables (current mass, rank drift, ViT CLS,
    layer-specific coverage descriptors, etc.). Do not feed NLL, future
    states, teacher-forced answer queries, or oracle-selected final IDs.
    Reversing groups negates the vector exactly, including context terms.
    """
    x = finite(token_features, 2).astype(float)
    c, a, b = (indices(v, len(x)) for v in (common, group_a, group_b))
    if set(c)&set(a) or set(c)&set(b) or set(a)&set(b) or len(a) != len(b):
        raise ValueError("disjoint equal groups required")
    if not 0 <= begin < end <= depth:
        raise ValueError("invalid observation/window depth")
    delta = x[a].mean(0)-x[b].mean(0)
    variance = x[a].var(0)-x[b].var(0)
    context = x[c].mean(0)
    return np.r_[delta, variance, delta*context, delta*(end-begin)/depth]


@dataclass
class PairModel:
    scale: np.ndarray
    weights: np.ndarray
    cache_mode: str
    feature_names: tuple
    catalog_hash: str
    stage: str = "decoder_transient_swap"

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim not in (1, 2) or x.shape[-1] != len(self.weights) or not np.isfinite(x).all():
            raise ValueError("finite features with trained schema required")
        return (x/self.scale)@self.weights

    def save(self, path):
        value = dict(schema=ESTIMAND, stage=self.stage, cache_mode=self.cache_mode,
                     feature_names=list(self.feature_names), catalog_hash=self.catalog_hash,
                     scale=self.scale.tolist(), weights=self.weights.tolist())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path, *, cache_mode, catalog_hash, feature_names):
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
        if (value.get("schema") != ESTIMAND or value.get("stage") != "decoder_transient_swap"
                or value.get("cache_mode") != cache_mode or value.get("catalog_hash") != catalog_hash
                or value.get("feature_names") != list(feature_names)):
            raise ValueError("incompatible weights/estimand/catalog; never reuse old E020 head")
        scale, weights = np.asarray(value["scale"], float), np.asarray(value["weights"], float)
        if (scale.ndim != 1 or len(scale) == 0 or weights.shape != scale.shape
                or len(feature_names) != len(scale) or not np.isfinite(scale).all()
                or not np.isfinite(weights).all() or (scale <= 0).any()):
            raise ValueError("invalid model arrays")
        return cls(scale, weights, cache_mode, tuple(feature_names), catalog_hash)


def fit_pairs(records, *, feature_names, catalog_hash, cache_mode, ridge=1.):
    """Train split only, no intercept/mean subtraction; antisymmetry preserved."""
    if not np.isfinite(ridge) or ridge <= 0 or not records:
        raise ValueError("positive ridge and nonempty real records required")
    if (not feature_names or len(set(feature_names)) != len(feature_names)
            or not catalog_hash or cache_mode not in ("native", "aligned_support")):
        raise ValueError("nonempty unique feature schema and valid catalog/cache mode required")
    xs, ys = [], []
    images = set()
    for record in records:
        if (record.get("schema") != ESTIMAND or record.get("cache_mode") != cache_mode
                or record.get("catalog_hash") != catalog_hash
                or record.get("feature_names") != list(feature_names)
                or record.get("feature_cutoff") != record.get("begin")
                or split_image(record["image_key"]) != "train"):
            raise ValueError("invalid schema/cache/catalog/causal cutoff or split leakage")
        if record.get("fixture", False):
            raise ValueError("synthetic fixtures cannot train a deployable VLM model")
        x = np.asarray(record["x"], float)
        y = float(record["delta"])
        if x.shape != (len(feature_names),) or not np.isfinite(x).all() or not np.isfinite(y):
            raise ValueError("nonfinite or mismatched training data")
        xs.append(x)
        ys.append(y)
        images.add(record["image_key"])
    if len(images) < 2:
        raise ValueError("at least two independent training images required")
    x, y = np.asarray(xs), np.asarray(ys)
    scale = np.sqrt((x*x).mean(0))
    scale = np.where(scale > 1e-12, scale, 1.)
    z = x/scale
    weights = np.linalg.solve(z.T@z+ridge*np.eye(z.shape[1]), z.T@y)
    return PairModel(scale, weights, cache_mode, tuple(feature_names), catalog_hash)


def choose_swap(predicted_gains, *, threshold):
    """One complete frozen candidate catalog, index 0 is the no-op."""
    gains = finite(predicted_gains, 1).astype(float)
    if gains[0] != 0 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("zero no-op gain and calibrated nonnegative threshold required")
    best = int(np.argmax(gains))
    return best if gains[best] > threshold else 0


def calibrate_threshold(predicted, true_gains, noop_differences):
    """VALIDATION images only, equal-work actions incl. true no-op column 0.

    Accepts complete [states, actions] catalogs. Threshold candidates use
    predicted maxima, not individual-action thresholds. Actual VLM generation
    confirmation is a separate requirement even if NLL gain is positive.
    """
    p, y = finite(predicted, 2), finite(true_gains, 2)
    noise = finite(noop_differences, 1)
    if p.shape != y.shape or (p[:, 0] != 0).any() or (y[:, 0] != 0).any():
        raise ValueError("complete matched catalogs with zero no-op column required")
    floor = float(np.quantile(np.abs(noise), .95))
    maxima = np.maximum(0., p.max(1))
    candidates = np.unique(np.r_[floor, maxima[maxima >= floor]])
    best = None
    for threshold in candidates:
        actions = np.array([choose_swap(row, threshold=float(threshold)) for row in p])
        gains = y[np.arange(len(y)), actions]
        score = float(gains.mean())
        key = (score, float(threshold))  # conservative tie: fewer accepted actions
        if best is None or key > best[0]:
            best = key, actions
    return dict(threshold=best[0][1], validation_gain=best[0][0],
                accepted=int((best[1] != 0).sum()), states=len(p),
                noise_floor=floor,
                status="CANDIDATE" if best[0][0] > floor and (best[1] != 0).any() else "NO_SUPPORTED_DECISIONS")
