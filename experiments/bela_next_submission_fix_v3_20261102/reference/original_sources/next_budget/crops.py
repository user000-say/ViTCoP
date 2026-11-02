"""Safe crop quotas for VALID patch tokens; never select image_newline tokens.

Only assigns per-crop K. Real token *choice* must be BELA Dual COVER + QA,
with original crop_id/patch_id mapping retained.
"""
def allocate_crop_quota(capacities, total, *, weights=None, min_per_crop=1):
    if not isinstance(total,int) or total<0:
        raise ValueError("total must be a nonnegative integer")
    if not capacities or any(type(v) is not int or v<0 for v in capacities):
        raise ValueError("capacities must contain nonnegative integers")
    n=len(capacities)
    active=[i for i,cap in enumerate(capacities) if cap>0]
    if not active:
        if total==0:return [0]*n
        raise ValueError("No patch capacity")
    if weights is None: weights=[1.0]*n
    if len(weights)!=n or any(not (0<=w<float("inf")) for w in weights):
        raise ValueError("Invalid group weights")
    if any(weights[i]==0 for i in active):
        raise ValueError("All real crop groups must have positive weights")
    if min_per_crop<0:
        raise ValueError("min_per_crop must be nonnegative")
    baseline=[min(c,min_per_crop) for c in capacities]
    if total<sum(baseline) or total>sum(capacities):
        raise ValueError("Crop quotas infeasible: cap or minimum")
    left=total-sum(baseline)
    quota=baseline[:]
    # Deterministic weighted divisor apportionment with hard capacities.
    # At most 1280 iterations, 5 crop groups for canonical paper setup.
    while left:
        elig=[i for i in active if quota[i]<capacities[i]]
        if not elig: raise AssertionError("Unexpected unfilled quota")
        chosen=min(elig,key=lambda i:(-(weights[i]/(quota[i]-baseline[i]+1)),i))
        quota[chosen]+=1
        left-=1
    assert sum(quota)==total
    return quota

def crop_quota_from_ids(crop_sizes, total, *,
                        policy="uniform", thumbnail_crop=0, global_weight=1.5,
                        min_per_crop=1):
    if policy not in ("uniform","thumbnail_boost"):
        raise ValueError("Unknown crop policy")
    if thumbnail_crop<0 or thumbnail_crop>=len(crop_sizes):
        raise ValueError("Invalid global crop id")
    weights=[1.]*len(crop_sizes)
    if policy=="thumbnail_boost": weights[thumbnail_crop]=global_weight
    return allocate_crop_quota(crop_sizes,total,weights=weights,
                               min_per_crop=min_per_crop)
