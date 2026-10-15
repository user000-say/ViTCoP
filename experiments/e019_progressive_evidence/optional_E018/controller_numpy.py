"""Executable offline controller and feature definitions; NumPy only.

This is a new portable module, not a patch against unavailable E014/E017 code.
No model answers, labels, future layers or dense predictions enter online inputs.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'cf_defer_v1'


def finite(x, ndim=None):
    x = np.asarray(x, dtype=np.float64)
    if (ndim is not None and x.ndim != ndim) or not np.isfinite(x).all():
        raise ValueError('nonfinite array or wrong rank')
    return x


def projection(d, r, seed):
    if d < 1 or r < 1:
        raise ValueError('projection dimensions must be positive')
    return np.random.default_rng(seed).normal(size=(d, r)) / np.sqrt(r)


def region_summary(features, region_ids, selected, proj, groups=16):
    """[count fraction, projected mean, projected second moment] per region.

    Moment inputs are L2-normalized ORIGINAL encoder patches. Region IDs are
    explicit and must include tile/frame identity when necessary. No CLIP and
    decoder feature cosine is computed. Empty groups have zero moments.
    """
    x, p = finite(features, 2), finite(proj, 2)
    g, m = np.asarray(region_ids), np.asarray(selected)
    if groups < 1 or g.shape != (len(x),) or m.shape != (len(x),):
        raise ValueError('invalid groups, region IDs or selection shape')
    if m.dtype != np.bool_ or not np.issubdtype(g.dtype, np.integer):
        raise ValueError('selected must be bool; region IDs must be integers')
    if np.any(g < 0) or np.any(g >= groups) or p.shape[0] != x.shape[1]:
        raise ValueError('region or projection mismatch')
    unit = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    z = unit @ p
    total = np.bincount(g, minlength=groups)
    count = np.bincount(g[m], minlength=groups)
    sums = np.zeros((groups, p.shape[1]))
    second = np.zeros_like(sums)
    np.add.at(sums, g[m], z[m])
    np.add.at(second, g[m], z[m] ** 2)
    denom = np.maximum(count[:, None], 1)
    return np.concatenate([
        (count / np.maximum(total, 1))[:, None], sums / denom, second / denom,
    ], axis=1).ravel()


def decision_features(query_hidden, current_visual, visual_attention, q_proj,
                      v_proj, *, target_k, completed_blocks, total_blocks,
                      original_n):
    """One prompt-only query vector, current visual state and existing Q-ATTN.

    q_proj and v_proj are independent persisted projections, NOT aligned spaces.
    visual_attention is a nonnegative vector from the already-fixed selector.
    Its sum is a statistic only, not assumed to be a full attention denominator.
    """
    q, v = finite(query_hidden, 1), finite(current_visual, 2)
    a, pq, pv = finite(visual_attention, 1), finite(q_proj, 2), finite(v_proj, 2)
    n = len(v)
    if not n or a.shape != (n,) or np.any(a < 0):
        raise ValueError('need current visual tokens and nonnegative aligned attention')
    if pq.shape[0] != q.shape[0] or pv.shape[0] != v.shape[1]:
        raise ValueError('projection dimensions do not match hidden states')
    if not isinstance(target_k, int) or not 0 < target_k <= n:
        raise ValueError('invalid target_k')
    if not 0 <= completed_blocks <= total_blocks or total_blocks < 1 or original_n < n:
        raise ValueError('invalid layer or token counts')
    qz = (q / max(np.linalg.norm(q), 1e-12)) @ pq
    vz = (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)) @ pv
    weights = a / max(a.sum(), 1e-12)
    entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum() / max(np.log(n), 1)
    extras = [entropy, weights.max(), np.sort(weights)[-target_k:].sum(),
              n / original_n, completed_blocks / total_blocks, target_k / n,
              a.sum()]
    return np.concatenate([qz, vz.mean(axis=0), (vz ** 2).mean(axis=0), extras])


def inputs(base, history, retained, mode, stride=17):
    b, h, r = finite(base, 2), finite(history, 2), finite(retained, 2)
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError('stride must be a positive integer')
    if h.shape != r.shape or len(b) != len(h) or h.shape[1] % stride:
        raise ValueError('summary shapes or stride do not match')
    if mode == 'history':
        extra = h
    elif mode == 'retained':
        extra = r.copy()
        # All controls see identical dropped-location/count metadata.
        extra[:, ::stride] = h[:, ::stride]
    elif mode == 'geometry':
        extra = np.zeros_like(h)
        extra[:, ::stride] = h[:, ::stride]
    elif mode == 'zero':
        extra = np.zeros_like(h)
    else:
        raise ValueError('mode must be history/retained/geometry/zero')
    return np.concatenate([b, extra], axis=1)


def group_splits(image_ids, salt='E018-frozen-v1'):
    """Canonical image/content identity shared across tasks, questions and arms."""
    values = np.array([int(hashlib.sha256((salt + str(i)).encode()).hexdigest()[:12], 16)
                       / 16**12 for i in image_ids])
    return values < .6, (values >= .6) & (values < .8), values >= .8


def token_layers(total_blocks, initial, events):
    """Events are (number of blocks already completed, count AFTER deletion)."""
    if not isinstance(total_blocks, int) or total_blocks < 1 or initial < 1:
        raise ValueError('invalid model depth or initial count')
    last, count, work = 0, initial, 0
    for completed, remaining in events:
        if not isinstance(completed, int) or not last <= completed <= total_blocks:
            raise ValueError('unordered or invalid completed-block boundary')
        if not isinstance(remaining, int) or not 0 < remaining <= count:
            raise ValueError('counts must be positive and nonincreasing')
        work += (completed - last) * count
        last, count = completed, remaining
    return work + (total_blocks - last) * count


@dataclass
class Model:
    mean: np.ndarray
    std: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    ymean: float
    ystd: float

    def predict(self, x):
        x = finite(x, 2)
        if x.shape[1] != len(self.mean):
            raise ValueError('controller input dimension mismatch')
        z = np.clip((x - self.mean) / self.std, -10, 10)
        return ((np.tanh(z @ self.w1 + self.b1) @ self.w2 + self.b2).ravel()
                * self.ystd + self.ymean)

    def save(self, path, metadata):
        np.savez_compressed(path, **self.__dict__, metadata=json.dumps(metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in cls.__dataclass_fields__}
            metadata = json.loads(str(z['metadata']))
        return cls(**data), metadata


def _loss_and_grad(z, y, params, weight_decay):
    w1, b1, w2, b2 = params
    h = np.tanh(z @ w1 + b1)
    residual = (h @ w2 + b2) - y[:, None]
    loss = np.mean(residual ** 2) + weight_decay * (np.sum(w1*w1) + np.sum(w2*w2))
    dy = 2 * residual / len(z)
    dh = (dy @ w2.T) * (1 - h*h)
    grad = [z.T @ dh + 2*weight_decay*w1, dh.sum(axis=0),
            h.T @ dy + 2*weight_decay*w2, dy.sum(axis=0)]
    return float(loss), grad


def fit(x, y, train, val, *, hidden=32, steps=300, lr=.003,
        weight_decay=1e-5, seed=0):
    """Small full-batch tanh regressor; backbone stays frozen; signed target."""
    x, y = finite(x, 2), finite(y, 1)
    train, val = np.asarray(train), np.asarray(val)
    if len(y) != len(x) or train.shape != y.shape or val.shape != y.shape:
        raise ValueError('row/mask mismatch')
    if train.dtype != np.bool_ or val.dtype != np.bool_ or np.any(train & val):
        raise ValueError('train/validation masks must be disjoint boolean arrays')
    if train.sum() < 4 or val.sum() < 2 or hidden < 1 or steps < 1:
        raise ValueError('insufficient train/validation data or invalid fit settings')
    mean, raw_std = x[train].mean(0), x[train].std(0)
    # Do not amplify rounding noise in almost-constant count/budget fields.
    std = np.where(raw_std < 1e-5, 1., raw_std)
    ymean, ystd = y[train].mean(), max(y[train].std(), 1e-6)
    z = np.clip((x - mean) / std, -10, 10)
    target = (y - ymean) / ystd
    rng = np.random.default_rng(seed)
    params = [rng.normal(size=(x.shape[1], hidden)) / np.sqrt(x.shape[1]),
              np.zeros(hidden), rng.normal(size=(hidden, 1)) / np.sqrt(hidden), np.zeros(1)]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    best, best_loss = None, float('inf')
    for step in range(1, steps + 1):
        _, gradients = _loss_and_grad(z[train], target[train], params, weight_decay)
        for j, (p, g) in enumerate(zip(params, gradients)):
            first[j] = .9 * first[j] + .1 * g
            second[j] = .999 * second[j] + .001 * g*g
            p -= lr * (first[j] / (1 - .9**step)) / (np.sqrt(second[j] / (1 - .999**step)) + 1e-8)
        h = np.tanh(z[val] @ params[0] + params[1])
        loss = np.mean(((h @ params[2] + params[3]).ravel() - target[val]) ** 2)
        if loss < best_loss:
            best_loss, best = loss, [p.copy() for p in params]
    return Model(mean, std, *best, float(ymean), float(ystd))


def choose_defer(predicted_gain, extra_ms, lam):
    gain, cost = finite(predicted_gain, 1), finite(extra_ms, 1)
    if gain.shape != cost.shape or np.any(cost <= 0) or not np.isfinite(lam) or lam < 0:
        raise ValueError('need positive additional cost and nonnegative finite lambda')
    return gain > lam * cost


def calibrate_lambda(gain, cost, fraction):
    gain, cost = finite(gain, 1), finite(cost, 1)
    if gain.shape != cost.shape or not len(gain) or np.any(cost <= 0) or not 0 <= fraction <= 1:
        raise ValueError('invalid calibration inputs')
    # Nonnegative lambda never deliberately spends on predicted negative gain.
    return float(max(0, np.quantile(gain / cost, 1 - fraction)))


def load_dataset(path, *, allow_synthetic=False):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    required = ['schema', 'kind', 'base', 'history', 'retained', 'early_nll',
                'late_nll', 'extra_ms', 'image_ids', 'sample_ids', 'stride', 'manifest']
    missing = set(required) - set(data)
    if missing or str(data.get('schema')) != SCHEMA:
        raise ValueError('invalid dataset schema or missing fields: ' + str(missing))
    if str(data['kind']) != 'real' and not allow_synthetic:
        raise ValueError('synthetic data require --allow-synthetic; never report as real')
    n = len(data['base'])
    for key in ['base', 'history', 'retained']:
        if len(finite(data[key], 2)) != n:
            raise ValueError('feature row mismatch')
    for key in ['early_nll', 'late_nll', 'extra_ms']:
        if finite(data[key], 1).shape != (n,):
            raise ValueError('label/cost row mismatch')
    if np.any(data['early_nll'] < 0) or np.any(data['late_nll'] < 0) or np.any(data['extra_ms'] <= 0):
        raise ValueError('NLL must be nonnegative and extra cost positive')
    if len(data['image_ids']) != n or len(data['sample_ids']) != n:
        raise ValueError('ID row mismatch')
    if len(set(map(str, data['sample_ids']))) != n:
        raise ValueError('sample IDs must be unique, including stage1 arm identity')
    manifest = json.loads(str(data['manifest']))
    for key in ['checkpoint_hash', 'stage1_config_hash', 'selector_hash', 'schedule_hash',
                'prompt_hash', 'projection_hash', 'cache_policy', 'query_rule', 'cost_source', 'data_split']:
        if not manifest.get(key):
            raise ValueError('missing manifest field: ' + key)
    return data
