"""Budget-neutral exchanges: learn which patches deserve the SAME future work."""
import numpy as np
from features import pair_inputs
from policy_protocol import catalogue_indices


def proposals(scores, selected, *, group_size=1, shortlist=16, protected=None, candidate_limit=None):
    scores, selected = np.asarray(scores), np.asarray(selected, dtype=int)
    n = len(scores); mask = np.zeros(n, dtype=bool); mask[selected] = True
    protected = np.zeros(n, dtype=bool) if protected is None else np.asarray(protected, dtype=bool)
    if protected.shape != (n,) or group_size < 1 or shortlist < group_size:
        raise ValueError('invalid protection/group/shortlist')
    donors = np.flatnonzero(mask & ~protected)
    incoming = np.flatnonzero(~mask)
    donors = donors[np.argsort(scores[donors], kind='stable')[:shortlist]]
    incoming = incoming[np.argsort(-scores[incoming], kind='stable')[:shortlist]]
    a, b = len(donors)//group_size, len(incoming)//group_size
    if not a or not b:
        return np.empty((0,group_size),dtype=int), np.empty((0,group_size),dtype=int)
    donors, incoming = donors[:a*group_size].reshape(a,group_size), incoming[:b*group_size].reshape(b,group_size)
    take = catalogue_indices(a*b, candidate_limit)
    return np.repeat(donors,b,axis=0)[take], np.tile(incoming,(a,1))[take]


def select(x, scores, selected, model, *, after, depth, next_after, final_k,
           rounds=1, group_size=1, shortlist=16, threshold=0., protected=None,
           mode='learned', seed=20, candidate_limit=None):
    selected = np.asarray(selected, dtype=int).copy()
    if len(np.unique(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= len(x)):
        raise ValueError('selected indices must be distinct current visual patches')
    if rounds < 0:
        raise ValueError('rounds must be nonnegative')
    rng = np.random.default_rng(seed); diagnostics = []
    for _ in range(rounds):
        donors, incoming = proposals(scores,selected,group_size=group_size,shortlist=shortlist,
                                     protected=protected,candidate_limit=candidate_limit)
        if not len(donors): break
        pairs = pair_inputs(x,selected,donors,incoming,after=after,depth=depth,next_after=next_after,final_k=final_k)
        if mode == 'learned':
            pred = model.predict(pairs); i = int(np.argmax(pred)); accept = pred[i] > threshold
        elif mode == 'random':
            pred = np.zeros(len(pairs)); i = int(rng.integers(len(pairs))); accept = True
        else: raise ValueError('mode must be learned/random')
        if not accept: break
        mask = np.zeros(len(x),dtype=bool); mask[selected] = True
        mask[donors[i]] = False; mask[incoming[i]] = True
        selected = np.flatnonzero(mask)
        diagnostics.append({'donors':donors[i].tolist(),'incoming':incoming[i].tolist(),
                            'predicted_nll_gain':float(pred[i])})
    return np.sort(selected), diagnostics
