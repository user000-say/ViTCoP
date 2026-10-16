"""Query-group response coverage exchange. Actual patches only; no model hook.

Reference and offline diagnostic implementation. No claim that attention mass
is semantic evidence or that increasing this objective improves task accuracy.
"""
import numpy as np


def normalize_rows(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('response must be a finite nonnegative [G,N] array')
    return a / np.maximum(a.sum(1, keepdims=True), 1e-12)


def group_queries(attention, group_ids, groups):
    """Average existing head-reduced question-to-vision rows into fixed groups.

    group_ids use ORIGINAL question token identity; exclude system/answer rows.
    The same mapping must be reused at both checkpoints. No linguistic parser
    is implied by contiguous grouping.
    """
    a = np.asarray(attention, dtype=np.float64)
    ids = np.asarray(group_ids)
    if a.ndim != 2 or ids.shape != (len(a),) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('invalid query rows or group IDs')
    if groups < 1 or np.any(ids < 0) or np.any(ids >= groups):
        raise ValueError('invalid query group range')
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('attention must be finite and nonnegative')
    out = np.zeros((groups, a.shape[1]))
    np.add.at(out, ids, a)
    return out / np.maximum(np.bincount(ids, minlength=groups)[:, None], 1)


def align_previous(previous, previous_patch_ids, current_patch_ids):
    """Restrict prior rows to current survivors, never reinsert removed patches."""
    p = np.asarray(previous)
    old, new = np.asarray(previous_patch_ids), np.asarray(current_patch_ids)
    if p.ndim != 2 or old.shape != (p.shape[1],) or new.ndim != 1:
        raise ValueError('patch identity shape mismatch')
    if len(set(old.tolist())) != len(old) or len(set(new.tolist())) != len(new):
        raise ValueError('patch IDs must be unique')
    lookup = {v:i for i,v in enumerate(old.tolist())}
    if any(v not in lookup for v in new.tolist()):
        raise ValueError('current patches are not a subset of prior patches')
    return p[:, [lookup[v] for v in new.tolist()]]


def prepare(scores, current, previous=None, alpha=.5):
    s = np.asarray(scores, dtype=np.float64)
    c = normalize_rows(current)
    if s.shape != (c.shape[1],) or not np.isfinite(s).all() or not 0 <= alpha <= 1:
        raise ValueError('score shape or alpha invalid')
    if alpha:
        if previous is None:
            raise ValueError('temporal modes require aligned prior response rows')
        p = normalize_rows(previous)
        if p.shape != c.shape:
            raise ValueError('query groups/patch columns differ across checkpoints')
        u = normalize_rows((1-alpha)*c + alpha*p)
    else:
        u = c
    active = u.sum(1) > 1e-12
    weights = active.astype(float) / max(int(active.sum()),1)
    # Preserve ranking while fixing modular term scale across candidate sets.
    s = s - min(float(s.min()),0.)
    s = s / s.sum() if s.sum() > 1e-12 else np.full_like(s,1/max(len(s),1))
    return s,u,weights


def objective(indices, scores, response, weights, lam=.15, eps=.01):
    coverage = response[:,indices].sum(1)
    return float(scores[indices].sum() + lam*np.dot(weights,np.log(eps+coverage)))


def select(scores, current, baseline_indices, *, previous=None, alpha=.5,
           lam=.15, swaps=4, mode='exchange', eps=.01, seed=0):
    """Repair a supplied selector result through <=swaps positive-gain swaps.

    The initial selector is external, so QED and project tie semantics remain
    intact. Return indices in CURRENT sequence order, plus objective diagnostics.
    With zero swaps or lambda, exchange is exact passthrough (no score rerank).
    """
    s,u,w = prepare(scores,current,previous,alpha)
    ids = np.asarray(baseline_indices)
    n = len(s)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype,np.integer) or not 0 < len(ids) <= n:
        raise ValueError('invalid baseline indices')
    if len(np.unique(ids)) != len(ids) or np.any(ids<0) or np.any(ids>=n):
        raise ValueError('baseline must contain distinct eligible indices')
    if swaps < 0 or lam < 0 or eps <= 0:
        raise ValueError('invalid exchange hyperparameter')
    mask = np.zeros(n,dtype=bool); mask[ids] = True
    before = objective(np.flatnonzero(mask),s,u,w,lam,eps)
    gains=[]
    if mode == 'score':
        mask[:]=False;mask[np.argsort(-s,kind='stable')[:len(ids)]]=True
    elif mode == 'mean':
        score = w @ u
        mask[:] = False
        mask[np.argsort(-score,kind='stable')[:len(ids)]] = True
    elif mode == 'random':
        rng = np.random.default_rng(seed)
        inside,outside=np.flatnonzero(mask),np.flatnonzero(~mask)
        b=min(swaps,len(inside),len(outside))
        if b:
            mask[rng.choice(inside,b,replace=False)]=False
            mask[rng.choice(outside,b,replace=False)]=True
    elif mode == 'exchange':
        if lam and swaps and len(ids)<n:
            for _ in range(swaps):
                inside,outside=np.flatnonzero(mask),np.flatnonzero(~mask)
                coverage=u[:,inside].sum(1)
                after=coverage[:,None,None]-u[:,inside][:,:,None]+u[:,outside][:,None,:]
                gain=(s[outside][None,:]-s[inside][:,None]
                      +lam*(w[:,None,None]*(np.log(np.maximum(eps+after,eps))
                                               -np.log(eps+coverage)[:,None,None])).sum(0))
                i,j=np.unravel_index(int(gain.argmax()),gain.shape)
                value=float(gain[i,j])
                if value <= 1e-12:
                    break
                mask[inside[i]]=False; mask[outside[j]]=True; gains.append(value)
    else:
        raise ValueError('mode must be exchange/mean/score/random')
    chosen=np.flatnonzero(mask)
    return chosen,{'objective_before':before,'objective_after':objective(chosen,s,u,w,lam,eps),
                   'accepted_swaps':len(gains),'gains':gains,
                   'coverage_per_group':(u[:,chosen].sum(1)).tolist(),
                   'jaccard_with_baseline':float(len(set(chosen)&set(ids))/len(set(chosen)|set(ids)))}
