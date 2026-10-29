#!/usr/bin/env python3
"""Small, auditable one-pass query+coverage comparator (NOT CoIn/SCoRe code).

This is only a controlled reference optimizer using *the same* affinity and
query-score inputs as QA-COVER. Parameter tuning must use development images.
"""
import numpy as np


def joint_greedy(sim, q, k, weight, native_ids=None, max_new=None):
    sim = np.asarray(sim,dtype=np.float64)
    q = np.asarray(q,dtype=np.float64)
    n=len(q)
    if sim.shape!=(n,n) or not np.all(np.isfinite(sim)) or not np.all(np.isfinite(q)):
        raise ValueError("finite sim[N,N] and query score[N] required")
    if np.any(sim<0) or np.any(sim>1) or not (0<=k<=n) or weight<0:
        raise ValueError("sim must be [0,1], 0<=K<=N, and weight>=0")
    if max_new is not None:
        if native_ids is None or not (0 <= max_new <= k):
            raise ValueError("replacement cap needs native IDs and 0<=cap<=K")
        native=np.asarray(native_ids,dtype=np.int64)
        if native.shape!=(k,) or len(set(native.tolist()))!=k or np.any(native<0) or np.any(native>=n):
            raise ValueError("native_ids needs K distinct IDs in [0,N)")
        is_native=np.zeros(n,dtype=bool)
        is_native[native]=True
    else:
        is_native=np.ones(n,dtype=bool)
    selected=[]
    running=np.zeros(n, dtype=np.float64)
    new_count=0
    for _ in range(k):
        candidate = np.ones(n,dtype=bool)
        candidate[selected]=False
        if max_new is not None and new_count>=max_new:
            candidate &= is_native
        # Fractional query term has a fixed K denominator, independent of
        # how many tokens have already been chosen.
        gain_cov=np.maximum(running[:,None],sim).mean(axis=0)-running.mean()
        score=gain_cov + weight*q/max(k,1)
        score[~candidate]=-np.inf
        chosen=int(np.argmax(score))
        if not np.isfinite(score[chosen]):
            raise AssertionError("not enough admissible candidates")
        selected.append(chosen)
        running=np.maximum(running,sim[:,chosen])
        new_count += int(not is_native[chosen])
    return np.array(sorted(selected), dtype=np.int64)


def representation_coverage(sim, selected):
    sim=np.asarray(sim)
    ids=np.asarray(selected,dtype=int)
    return float(sim[:,ids].max(axis=1).mean()) if len(ids) else 0.0
