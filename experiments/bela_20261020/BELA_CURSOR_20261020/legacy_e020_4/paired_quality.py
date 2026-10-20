"""Image-cluster paired bootstrap with an EXTERNAL official scorer callback.

score_a(indices)/score_b(indices) must score the resampled predictions and
preserve repeated index multiplicity. Never average per-sample F1 or MME.
"""
import numpy as np


def paired_quality(group_ids, strata, score_a, score_b, repeats=2000, seed=17):
    groups, layers = np.asarray(group_ids), np.asarray(strata)
    if groups.ndim != 1 or layers.shape != groups.shape or not len(groups):
        raise ValueError('aligned nonempty group IDs/strata required')
    if repeats < 1 or not isinstance(repeats, int):
        raise ValueError('positive bootstrap count required')
    # A cluster cannot straddle strata: use task/category + image as ID.
    grouped = {}
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        labels = np.unique(layers[indices])
        if len(labels) != 1:
            raise ValueError('image cluster straddles strata')
        grouped.setdefault(labels[0], []).append(indices)
    all_rows = np.arange(len(groups))
    point_a, point_b = score_a(all_rows), score_b(all_rows)
    names = sorted(point_a)
    if set(names) != set(point_b) or not names:
        raise ValueError('scorers must return matching nonempty metric dictionaries')
    def values(result):
        if set(result) != set(names):
            raise ValueError('scorer metric names changed')
        a = np.array([result[k] for k in names], dtype=float)
        if not np.isfinite(a).all():
            raise ValueError('nonfinite scorer result')
        return a
    delta = values(point_b)-values(point_a)
    rng, draws = np.random.default_rng(seed), []
    for _ in range(repeats):
        sample = np.concatenate([np.concatenate([clusters[j] for j in
            rng.integers(0, len(clusters), len(clusters))]) for clusters in grouped.values()])
        draws.append(values(score_b(sample))-values(score_a(sample)))
    ci = np.quantile(draws, [.025,.975], axis=0)
    return {name:dict(delta=float(delta[j]), ci95=ci[:,j].tolist()) for j,name in enumerate(names)}
