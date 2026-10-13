"""E015: experimental region-aware COVER and AEE, under E014's exact interface.

Diagnostics, not a novel theoretical optimization. No feature merging:
selected outputs remain 172 ORIGINAL CLIP patch features in spatial order.
Fixed and dynamic variants differ only in b (exchange count), never input K.
"""
import torch
import torch.nn.functional as F
from .e014_stage1_support import _eviction_risk

K = 172
BUDGETS = (0, 8, 16, 24, 32)

def _check(tokens, attn):
    if tokens.ndim != 2 or tokens.size(0) != 576:
        raise ValueError("[E015] CLIP patches must have shape [576,D]")
    if attn.shape != (576,) or attn.device != tokens.device:
        raise ValueError("[E015] CLS attention must have shape [576] on same device")
    # Non-finite checks occur only in E014_AUDIT, not latency runs.

def _groups(n, cells, device):
    patch_grid = 24
    cell_size = patch_grid // cells
    ids = torch.arange(n, device=device)
    row = (ids // patch_grid) // cell_size
    col = (ids % patch_grid) // cell_size
    return row * cells + col

def _regional_novelty(tokens, attn, reference_ids, query_ids, gid, cells,
                      ref_mode="MEAN", dual_weight=0.25):
    """[0,1] novelty for query IDs against ACTUALLY retained reference IDs.

    MEAN: cosine with raw-feature region mean (direction normalized).
    MEDOID: cosine with nearest real reference to region mean.
    SALIENT: cosine with highest-CLS real reference per region.
    DUAL: convex combination of cosine similarities to MEAN and MEDOID.
    The representative is ONLY a scorer, never a new LLM input token.
    Empty region: novelty=1 for its queries.
    """
    x = tokens.float()
    xn = F.normalize(x, dim=-1, eps=1e-8)
    a = attn.float().clamp_min(0)
    group_n = cells * cells
    ref_groups = gid.index_select(0, reference_ids)
    query_groups = gid.index_select(0, query_ids)

    sums = x.new_zeros((group_n, x.size(-1)))
    sums.index_add_(0, ref_groups, x.index_select(0, reference_ids))
    counts = x.new_zeros((group_n,))
    counts.index_add_(0, ref_groups, x.new_ones((reference_ids.numel(),)))
    means = F.normalize(sums, dim=-1, eps=1e-8)
    cos_mean = (
        xn.index_select(0, query_ids) * means.index_select(0, query_groups)
    ).sum(-1).clamp(0, 1)

    if ref_mode == "MEAN":
        sim = cos_mean
    else:
        # All 16 region representatives computed using tensor argmax; no .item().
        # Empty groups are later masked; their argmax is a harmless placeholder.
        rvec = xn.index_select(0, reference_ids)
        if ref_mode == "SALIENT":
            rep_score = a.index_select(0, reference_ids)
        else:
            rep_score = (
                rvec * means.index_select(0, ref_groups)
            ).sum(-1)
        reps = []
        for g in range(group_n):
            valid_score = rep_score.masked_fill(ref_groups != g, -float("inf"))
            reps.append(reference_ids[torch.argmax(valid_score)])
        rep_ids = torch.stack(reps)
        cos_anchor = (
            xn.index_select(0, query_ids) *
            xn.index_select(0, rep_ids.index_select(0, query_groups))
        ).sum(-1).clamp(0, 1)
        if ref_mode == "DUAL":
            sim = ((1. - dual_weight) * cos_mean +
                   dual_weight * cos_anchor)
        else:
            sim = cos_anchor

    sim = torch.where(counts.index_select(0, query_groups) > 0,
                      sim, torch.zeros_like(sim))
    return (1. - sim).clamp(0,1)

def _core_gap(tokens, attn, core, residual, gid, cfg):
    novelty = _regional_novelty(
        tokens, attn, core, residual, gid, cfg.cells,
        cfg.ref_mode, cfg.dual_weight)
    weights = attn.float().index_select(0, residual).clamp_min(0.)
    return (weights * novelty).sum() / weights.sum().clamp_min(1e-8)

def _score_greedy_reference_alpha(tokens, attn, alpha):
    x = F.normalize(tokens.float(), p=2, dim=-1, eps=1e-8)
    weight = attn.float().clamp_min(0).pow(alpha)
    first = torch.argmax(attn)
    chosen = [first]
    alive = torch.ones(tokens.size(0), dtype=torch.bool, device=tokens.device)
    alive[first] = False
    max_sim = x @ x[first]
    for _ in range(K - 1):
        score = weight * (1. - max_sim)
        score = score.masked_fill(~alive, -float("inf"))
        j = torch.argmax(score)
        chosen.append(j)
        alive[j] = False
        max_sim = torch.maximum(max_sim, x @ x[j])
    return torch.stack(chosen).sort().values

def e015_choose(tokens, attn, cfg, core_ids=None):
    """Match E014 e014_choose return contract, including audit-only device stats."""
    _check(tokens, attn)
    core = (torch.topk(attn, K, largest=True).indices.sort().values
            if core_ids is None else core_ids)
    zero = attn.float().new_zeros(())
    empty = core[:0]
    variant = cfg.variant

    if variant == "E015_SCORE_RECON":
        selected = _score_greedy_reference_alpha(tokens, attn, cfg.alpha)
        new_mask = ~torch.isin(selected, core)
        old_mask = ~torch.isin(core, selected)
        added, removed = selected[new_mask], core[old_mask]
        return selected, core, added, removed, dict(
            budget=added.numel(), utility=zero, gain_sum=zero,
            risk_sum=zero, coverage_gap=zero,
            score_type="SCORE_PAPER_RECON_NOT_OFFICIAL")

    ids = torch.arange(tokens.shape[0], device=tokens.device)
    mask = torch.ones(tokens.shape[0],dtype=torch.bool,device=tokens.device)
    mask[core] = False
    residual = ids[mask]
    gid = _groups(tokens.shape[0], cfg.cells, tokens.device)

    if variant == "E015_PROBE":
        h = _core_gap(tokens, attn, core, residual, gid, cfg)
        return core, core, empty, empty, dict(
            budget=0, utility=zero, gain_sum=zero, risk_sum=zero,
            coverage_gap=h, score_type="PROBE_CORE172")

    if variant.endswith("_DYNAMIC"):
        h = _core_gap(tokens, attn, core, residual, gid, cfg)
        boundaries = attn.float().new_tensor(cfg.thresholds)
        bucket = int(torch.bucketize(h.detach(), boundaries).item())
        b = BUDGETS[bucket]
    else:
        h = zero
        b = cfg.b

    if b == 0:
        return core, core, empty, empty, dict(
            budget=0, utility=zero, gain_sum=zero, risk_sum=zero,
            coverage_gap=h, score_type="REGIONAL_ZERO_SWAP")

    if variant.startswith(("E015_AEE_", "E015_GLOBAL_AEE_")):
        # EXACT E014 individual eviction risk; does NOT solve group eviction.
        risk = _eviction_risk(tokens, attn, core)
        evict_score = risk
    else:
        risk = None
        evict_score = attn.float().index_select(0,core)
    evict_rank = torch.argsort(evict_score, descending=False, stable=True)
    removed = core.index_select(0, evict_rank[:b])
    retained = core.index_select(0, evict_rank[b:])

    if variant.startswith("E015_GLOBAL_"):
        # Direct control: exact original E014 global-576 mean Gain,
        # using the SAME frozen H-based b router as the regional variants.
        from .e014_stage1_support import _cover_gain
        gain = _cover_gain(tokens, attn, residual)
    else:
        novelty = _regional_novelty(tokens, attn, retained, residual, gid,
                                   cfg.cells, cfg.ref_mode, cfg.dual_weight)
        gain = (attn.float().index_select(0,residual).clamp_min(0.) * novelty)
    take = torch.argsort(gain, descending=True, stable=True)[:b]
    added = residual.index_select(0, take)
    selected = torch.cat((retained, added),0).sort().values

    if selected.numel() != K:
        raise RuntimeError("[E015] violated constant-K=172")
    return selected, core, added, removed, dict(
        budget=b, utility=zero, gain_sum=gain.index_select(0,take).sum(),
        risk_sum=(risk.index_select(0, evict_rank[:b]).sum()
                  if risk is not None else zero),
        coverage_gap=h,
        score_type=(("GLOBAL_" if variant.startswith("E015_GLOBAL_") else
                     "REGIONAL_") +
                    ("RISK_" if risk is not None else "COVER_") +
                    cfg.ref_mode))
