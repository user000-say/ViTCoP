"""Torch reference to reuse an *existing* Stage1 COVER greedy-addition order.

This does NOT invent the missing current BELA model wiring. Before enabling,
Cursor must expose `greedy_order` from the real production COVER selector.
The E014 historical FIXED_COVER comparator currently returns sorted tokens;
sorted IDs are NOT valid greedy-order input to this adapter.
"""
from __future__ import annotations

def torch_coverage_difficulty(patch_features, greedy_order,
                              probe_ks=(16,32,64,96),
                              probe_weights=(.15,.35,.35,.15)):
    import torch
    import torch.nn.functional as F
    if patch_features.ndim != 2:
        raise ValueError("vision features expected [N,D] for a SINGLE image")
    n = patch_features.shape[0]
    ids = torch.as_tensor(greedy_order, dtype=torch.long,
                          device=patch_features.device)
    if ids.ndim != 1 or ids.numel() < max(probe_ks):
        raise ValueError("missing full COVER greedy prefix")
    # The same original patch ID can never be added twice.
    if ids.unique().numel() != ids.numel() or bool((ids<0).any().item()) or bool((ids>=n).any().item()):
        raise ValueError("duplicate / invalid greedy patch ID")
    f = F.normalize(patch_features.float(), p=2, dim=-1, eps=1e-12)
    selected = f.index_select(0, ids[:max(probe_ks)])
    pairwise = (f @ selected.T).clamp(min=0, max=1)
    cumulative = torch.cummax(pairwise, dim=1).values
    r = torch.stack([
        (1.0 - cumulative[:, k-1]).clamp(min=0).mean()
        for k in probe_ks
    ])
    wt = torch.tensor(probe_weights, dtype=r.dtype, device=r.device)
    if wt.numel() != r.numel() or not bool((wt>=0).all().item()) or wt.sum().item()<=0:
        raise ValueError("invalid probe weights")
    # REFERENCE ONLY: validation guards above also call `.item()` and `.unique()`;
    # GPU performance is NOT optimized. In production move static invariants to
    # the preflight phase and profile any remaining GPU synchronization.
    return (r * wt).sum().div(wt.sum()).item()

def dynamic_budget_stage1(existing_cover_selector, patch_features,
                          greedy_order, profile_name):
    """Require the existing selector to return real patch IDs for requested K.

    No interpolation of hidden states and no remapping of original patch IDs.
    The actual LLaVA splice-in is NOT performed by this helper.
    """
    from .budget import PROFILES
    import torch
    if profile_name not in PROFILES:
        raise ValueError("unknown budget policy profile")
    k = PROFILES[profile_name].k0
    ids = existing_cover_selector(k)
    ids = torch.as_tensor(ids, dtype=torch.long, device=patch_features.device)
    if (ids.ndim != 1 or ids.numel()!=k or ids.unique().numel()!=k
        or (ids<0).any() or (ids>=patch_features.shape[0]).any()):
        raise ValueError("invalid actual-COVER token selection")
    greedy = torch.as_tensor(greedy_order, device=ids.device, dtype=torch.long)
    if greedy.ndim != 1 or greedy.numel() < k or greedy[:k].unique().numel()!=k:
        raise ValueError("True COVER greedy prefix required")
    if not torch.equal(ids.sort().values, greedy[:k].sort().values):
        raise ValueError("Production selector disagrees with claimed greedy prefix")
    # Stable original spatial order expected by ordinary ViT->LLM projector.
    return ids.sort().values
