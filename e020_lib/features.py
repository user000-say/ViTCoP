"""One prompt-only feature schema shared by fitting and deployed inference.

No answer text or loss enters these inputs. Hidden projection is a fixed random
projection of decoder patches, not a CLIP-to-decoder similarity. Group rows are
existing question-to-visual responses, not semantic ground truth.
"""
import numpy as np

TOKEN_DIM = 22
CONTEXT_DIM = 28
INPUT_DIM = TOKEN_DIM + CONTEXT_DIM
HISTORY_COLUMNS = (6, 7, 8, 9, 10)
FEATURE_SCHEMA = 'e020_prompt_features_v2'


def ranks(values):
    x = np.asarray(values)
    order = np.argsort(x, kind='stable')
    out = np.empty(len(x), dtype=float)
    out[order] = np.arange(len(x)) / max(len(x)-1, 1)
    return out


def response_rows(values, n):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] != n or not 1 <= a.shape[0] <= 4:
        raise ValueError('supply 1..4 original-question response groups, aligned patches')
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('finite nonnegative response rows required')
    a = a / np.maximum(a.sum(1, keepdims=True), 1e-12)
    return np.pad(a, ((0, 4-len(a)), (0, 0)))


def token_features(scores, response, *, previous=None, hidden=None, projection=None,
                   encoder_rank=None, patch_xy=None):
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if s.ndim != 1 or n < 2 or not np.isfinite(s).all():
        raise ValueError('finite selector scores aligned with current visual patches required')
    a = response_rows(response, n)
    p = response_rows(previous, n) if previous is not None else np.zeros_like(a)
    now_rank = ranks(a.mean(0))
    old_rank = ranks(p.mean(0)) if previous is not None else np.zeros(n)
    x = np.zeros((n, TOKEN_DIM), dtype=float)
    x[:, 0] = ranks(s)
    x[:, 1:5] = (a * n).T
    x[:, 5] = now_rank
    x[:, 6:10] = (p * n).T
    x[:, 10] = now_rank-old_rank if previous is not None else 0
    if hidden is not None:
        h, proj = np.asarray(hidden, dtype=float), np.asarray(projection, dtype=float)
        if h.ndim != 2 or len(h) != n or proj.shape != (h.shape[1], 8):
            raise ValueError('persisted decoder random projection must be [D,8]')
        x[:, 11:19] = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12) @ proj
    if encoder_rank is not None:
        er = np.asarray(encoder_rank, dtype=float)
        if er.shape != (n,):
            raise ValueError('encoder rank must preserve original patch identity')
        x[:, 19] = er
    if patch_xy is not None:
        xy = np.asarray(patch_xy, dtype=float)
        if xy.shape != (n, 2):
            raise ValueError('patch/tile spatial coordinates must be explicitly supplied')
        x[:, 20:22] = xy
    if not np.isfinite(x).all():
        raise ValueError('nonfinite token features')
    return x


def pair_inputs(x, selected, donor_groups, incoming_groups, *, after, depth,
                next_after, final_k):
    """Each pair shares C = S minus donor. Output [pairs,2,50], incoming first.

    Context is updated after every accepted online exchange. A group value is
    conditional on that remaining set, not an intrinsic token importance.
    next_after denotes the planned FINAL compression boundary (the opportunity
    horizon), including when THREE has another intermediate event before it.
    """
    x = np.asarray(x, dtype=float)
    selected = np.asarray(selected, dtype=int)
    donors, incoming = np.asarray(donor_groups, dtype=int), np.asarray(incoming_groups, dtype=int)
    if x.ndim != 2 or x.shape[1] != TOKEN_DIM or donors.ndim != 2 or incoming.shape != donors.shape:
        raise ValueError('invalid feature/group shapes')
    if not len(donors) or not donors.shape[1] or not 0 <= after < next_after <= depth:
        raise ValueError('nonempty action groups and a future cutoff required')
    if len(np.unique(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= len(x)):
        raise ValueError('invalid selected indices')
    mask = np.zeros(len(x), dtype=bool); mask[selected] = True
    if np.any(donors < 0) or np.any(donors >= len(x)) or np.any(incoming < 0) or np.any(incoming >= len(x)):
        raise ValueError('action outside current patches')
    if not mask[donors].all() or mask[incoming].any():
        raise ValueError('donors must be inside and incoming patches outside')
    if any(len(np.unique(row)) != len(row) for row in donors) or any(len(np.unique(row)) != len(row) for row in incoming):
        raise ValueError('duplicate patch within action')
    g = donors.shape[1]
    means = (x[selected].sum(0)[None, :]-x[donors].sum(1)) / max(len(selected)-g, 1)
    scalars = np.tile([len(selected)/len(x), g/len(x), after/depth,
                       (next_after-after)/depth, final_k/len(x), len(x)/576], (len(donors), 1))
    context = np.concatenate([means, scalars], axis=1)
    return np.stack([np.concatenate([x[incoming].mean(1), context], axis=1),
                     np.concatenate([x[donors].mean(1), context], axis=1)], axis=1)


def feature_mask(mode):
    mask = np.ones(INPUT_DIM)
    if mode in ('current', 'no_context', 'shuffled'):
        mask[list(HISTORY_COLUMNS)] = 0
        mask[[TOKEN_DIM+i for i in HISTORY_COLUMNS]] = 0
    if mode == 'no_context':
        mask[TOKEN_DIM:TOKEN_DIM+TOKEN_DIM] = 0
    elif mode not in ('full', 'current', 'shuffled'):
        raise ValueError('unknown capacity-matched feature control')
    return mask
