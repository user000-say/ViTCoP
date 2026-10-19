"""Single-swap lifetime policy; offline and online share one finite catalog.

Selected groups live only to end, then BOTH possible groups are discarded.
This is distinct from old final-identity replacement. NumPy reference only.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .counterfactual import Swap, run_swap
from .evidence import finite, ids_array, indices, topk
from .learning import pair_features, choose_swap


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Catalog:
    baseline: tuple               # active-local baseline indices
    final: tuple                  # active-local final indices, fixed before choice
    swaps: tuple                  # (common, inserted, removed), active-local
    rule_hash: str                 # shared across images, used by model manifest
    action_hash: str               # particular identities in this sample
    window: tuple
    active_ids: tuple


def propose(scores, active_ids, baseline_keep, final_k, *, group_size=4, limit=16,
            begin=10, end=20, depth=32, proposal_rule="BIND_NATIVE_RULE"):
    """At most limit TOTAL actions incl. no-op; no resurrection or answer use.

    scores is a proposal ranking (native Q_ATTN or RDRIFT), not a utility
    certificate. Final anchors are chosen within baseline NOW, never from a
    future answer-dependent reference. Labels must cover this exact catalog.
    """
    s, pid = finite(scores, 1), ids_array(active_ids)
    if len(s) != len(pid) or type(final_k) is not int or type(group_size) is not int or type(limit) is not int:
        raise ValueError("aligned scores/IDs and integer sizes required")
    base = indices(baseline_keep, len(pid))
    if not 0 < final_k <= len(base) or group_size < 1 or limit < 1:
        raise ValueError("invalid final/group/catalog sizes")
    if not all(type(v) is int for v in (begin, end, depth)) or not 0 <= begin < end <= depth:
        raise ValueError("invalid survival window")
    final = np.sort(base[topk(s[base], final_k)])
    removable = np.setdiff1d(base, final)
    outside = np.setdiff1d(np.arange(len(s)), base)
    # Rank tie-breaking is stable in active sequence order.
    removable = removable[np.argsort(s[removable], kind="stable")]
    outside = outside[np.argsort(-s[outside], kind="stable")]
    chunks = lambda a: [a[i:i+group_size] for i in range(0, len(a)-group_size+1, group_size)]
    candidates = []
    for d in chunks(removable):
        for inserted in chunks(outside):
            common = np.setdiff1d(base, d)
            if len(common) == 0:
                continue
            proxy = float(s[inserted].sum()-s[d].sum())
            candidates.append((proxy, tuple(common), tuple(inserted), tuple(d)))
    candidates.sort(key=lambda a: (-a[0], a[2], a[3]))
    swaps = tuple(row[1:] for row in candidates[:limit-1])
    rule = dict(version="single_transient_swap_catalog_v1", group_size=group_size,
                limit=limit, final_k=final_k, middle_k=len(base),
                begin=begin, end=end, depth=depth, proposal_rule=proposal_rule,
                grouping="stable-score-disjoint-chunks", final="current-baseline-topk")
    base, final = tuple(sorted(map(int, base))), tuple(map(int, final))
    identity = dict(ids=pid.tolist(), baseline=base, final=final, swaps=swaps)
    # Convert NumPy index scalars before canonical JSON hashing.
    swaps = tuple(tuple(tuple(map(int, part)) for part in action) for action in swaps)
    identity["swaps"] = swaps
    return Catalog(base, final, swaps, hash_value(rule), hash_value(identity),
                   (begin, end, depth), tuple(map(int, pid)))


def feature_catalog(catalog, token_features, *, begin, end, depth):
    if (begin, end, depth) != catalog.window:
        raise ValueError("survival window differs from trained catalog")
    x = finite(token_features, 2)
    zero = np.zeros(x.shape[1]*4)
    return np.array([zero]+[pair_features(x, c, inserted, removed,
                                         begin=begin, end=end, depth=depth)
                            for c, inserted, removed in catalog.swaps])


def decide(catalog, token_features, model, *, begin, end, depth, threshold):
    if (model.catalog_hash != catalog.rule_hash or model.stage != "decoder_transient_swap"
            or model.cache_mode != "native"):
        raise ValueError("model/catalog mismatch")
    features = feature_catalog(catalog, token_features, begin=begin, end=end, depth=depth)
    predicted = np.asarray(model.predict(features), float)
    action = choose_swap(predicted, threshold=threshold)
    if action == 0:
        keep = catalog.baseline
    else:
        common, inserted, _ = catalog.swaps[action-1]
        keep = tuple(sorted(common+inserted))
    # Caller must force catalog.final at end. Native re-selection there would
    # change the estimand and invalidate this head's deployment contract.
    return dict(action=action, keep_local=keep,
                keep_ids=tuple(catalog.active_ids[i] for i in keep),
                final_ids=tuple(catalog.active_ids[i] for i in catalog.final),
                predicted_gain=float(predicted[action]), action_hash=catalog.action_hash)


def collect_catalog(adapter, prefix, catalog, active_ids, token_features, *,
                    begin, end, depth, image_key, question_key, feature_names,
                    cache_mode="native"):
    pid = ids_array(active_ids)
    if (tuple(map(int, pid)) != catalog.active_ids
            or tuple(map(int, pid)) != tuple(adapter.inspect(prefix)["active_ids"])
            or len(token_features) != len(pid)):
        raise ValueError("prefix, feature rows and proposal original IDs must have identical order")
    features = feature_catalog(catalog, token_features, begin=begin, end=end, depth=depth)
    if len(feature_names) != features.shape[1]:
        raise ValueError("feature names do not match policy input")
    rows = []
    state_key = hash_value(dict(prefix=adapter.inspect(prefix)["prefix_fingerprint"],
                                image=image_key, question=question_key, begin=begin, end=end,
                                depth=depth, final=[int(pid[i]) for i in catalog.final],
                                action_hash=catalog.action_hash, cache_mode=cache_mode))
    for i, (common, inserted, removed) in enumerate(catalog.swaps, 1):
        original = lambda local: tuple(int(pid[j]) for j in local)
        swap = Swap(begin, end, original(common), original(inserted), original(removed), original(catalog.final))
        row = run_swap(adapter, prefix, swap, image_key=image_key, question_key=question_key, mode=cache_mode)
        row.update(state_key=state_key, catalog_hash=catalog.rule_hash,
                   action_hash=catalog.action_hash, catalog_size=len(catalog.swaps)+1,
                   action_index=i, x=features[i].tolist(), feature_names=list(feature_names),
                   feature_cutoff=begin, fixture=bool(getattr(adapter, "is_fixture", True)))
        rows.append(row)
    return rows
