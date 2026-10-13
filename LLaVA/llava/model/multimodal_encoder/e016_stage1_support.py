"""E016 48h Stage1 selector study.

The selector swaps exactly b CLIP patch IDs within original CLS Top172.
It never merges features, changes sequence length, touches Stage2/QED, or
requires question annotations. Every method holds visual work constant.

New contribution here is a controlled diagnostic/implementation to evaluate
ALREADY PUBLISHED coverage ideas, not a claim that weighted facility
location, spatial affinity or hierarchical CLIP attention is novel.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

T = 576
K = 172
GRID = 24

def _validate(tokens, attention, middle_attention=None):
    if tokens.ndim != 2 or tokens.shape[0] != T or attention.shape != (T,):
        raise ValueError("E016 expects [576,D] tokens and [576] attention")
    if tokens.device != attention.device:
        raise ValueError("E016 tokens and attention must be on same device")
    if middle_attention is not None and (
        middle_attention.shape != (T,) or
        middle_attention.device != tokens.device
    ):
        raise ValueError("E016 middle_attention must be [576], same device")

def _base_similarity(tokens, centered=False, spatial_weight=0.0, sigma=4.0):
    """[576,576] nonnegative source x representative kernel; FP32.

    Keep source index on rows; target candidate on columns.
    After spatial mixing the diagonal remains 1 for valid nonzero tokens.
    """
    x = F.normalize(tokens.float(), p=2, dim=-1, eps=1e-8)
    affinity = x @ x.transpose(0, 1)
    if centered:
        # CoverPruner-inspired source-row centering. Applied in CLIP space;
        # do not mislabel it as the paper's projector-space implementation.
        affinity = (affinity - affinity.mean(dim=1,keepdim=True)).clamp_min(0.)
    else:
        affinity = affinity.clamp(0.,1.)
    if spatial_weight:
        i = torch.arange(T,device=tokens.device)
        row = (i // GRID).float()
        col = (i % GRID).float()
        dsq = (row[:,None]-row[None,:]).square() + (
               col[:,None]-col[None,:]).square()
        locality = torch.exp(-dsq/(2.0*sigma*sigma))
        affinity = affinity * (
            (1. - spatial_weight) + spatial_weight*locality
        )
    return affinity

def _demand(attn, mid_attn=None, mid_mix=0.3):
    """Source demand, NOT candidate salience score."""
    a=attn.float().clamp_min(0.)
    if mid_attn is not None:
        m=mid_attn.float().clamp_min(0.)
        a=(1. - mid_mix)*(a/a.sum().clamp_min(1e-9)) + (
             mid_mix*m/m.sum().clamp_min(1e-9))
    return a/a.sum().clamp_min(1e-9)

def _core_keep(core, attn, affinity, b, variant, protect=128):
    """Only E016_PROTECTED changes E014's evictions; others delete CLS tail."""
    rank=torch.argsort(attn.float().index_select(0,core),
                       descending=True,stable=True)
    ranked=core.index_select(0,rank)
    if variant=="E016_PROTECTED":
        anchors=ranked[:protect]
        candidates=ranked[protect:]
        if b>candidates.numel():
            raise ValueError("protected pool smaller than exchange budget")
        # Risk relative to anchors that are GUARANTEED to survive.
        # Two disposable patches cannot make each other appear "safe".
        ca=affinity.index_select(0,candidates).index_select(1,anchors)
        near=ca.amax(dim=1)
        risk=attn.float().index_select(0,candidates).clamp_min(0.)*(1.-near)
        order=torch.argsort(risk,stable=True)
        removed=candidates.index_select(0,order[:b])
        risk_sum=risk.index_select(0,order[:b]).sum()
    else:
        removed=ranked[-b:] if b else ranked[:0]
        risk_sum=attn.float().new_zeros(())
    # Core remains the original Top172; additions never select evicted IDs.
    keep_mask=~torch.isin(core,removed)
    kept=core[keep_mask]
    return kept,removed,risk_sum

def _block_marginal(affinity, demand, retained, residual, b, block):
    """Batched greedy marginal source coverage; no host/GPU sync inside loop.

    One block computes gains against current *real selected set* and selects
    its best <=block additions in parallel. This is NOT full greedy facility
    location unless block=1. Default block=8 reduces GPU dispatch overhead.
    """
    c=affinity.index_select(1,retained).amax(dim=1)
    a=affinity.index_select(1,residual)
    alive=torch.ones(residual.numel(),dtype=torch.bool,device=residual.device)
    picked=[]
    gains_sum=demand.new_zeros(())
    for _ in range((b+block-1)//block):
        n_pick=min(block,b-len(picked)*block)
        gain=(torch.relu(a-c[:,None])*demand[:,None]).sum(dim=0)
        masked=gain.masked_fill(~alive,-float("inf"))
        local=torch.topk(masked,k=n_pick,largest=True).indices
        # Attribute gain to candidate selection BEFORE coverage update.
        gains_sum=gains_sum+gain.index_select(0,local).sum()
        alive[local]=False
        picked.append(residual.index_select(0,local))
        c=torch.maximum(c,a.index_select(1,local).amax(dim=1))
    if not picked:
        return residual[:0],gains_sum
    return torch.cat(picked,dim=0),gains_sum

def e016_choose(tokens, attention, cfg, core_ids=None, middle_attention=None):
    """Contract: (selected,original_core,added,removed,stats)."""
    _validate(tokens,attention,middle_attention)
    if cfg.variant=="E016_HIER" and middle_attention is None:
        raise RuntimeError("E016_HIER needs middle-layer CLIP CLS attention")
    original=(torch.topk(attention,K).indices.sort().values
              if core_ids is None else core_ids)
    if original.shape != (K,):
        raise ValueError("expected 172 original CLS Core IDs")
    z=attention.float().new_zeros(())
    b=int(cfg.b)
    if b==0:
        return original,original,original[:0],original[:0],dict(
            budget=0,utility=z,gain_sum=z,risk_sum=z,score_type="E016_NO_SWAP")
    variant=cfg.variant
    if variant not in ("E016_MARGINAL","E016_PROTECTED","E016_SPATIAL",
                       "E016_HIER","E016_CENTERED"):
        raise ValueError("unknown E016 variant "+variant)
    affinity=_base_similarity(
        tokens,centered=(variant=="E016_CENTERED"),
        spatial_weight=(cfg.e016_spatial_weight if variant=="E016_SPATIAL" else 0.),
        sigma=cfg.e016_sigma)
    kept,removed,risk_sum=_core_keep(
        original,attention,affinity,b,variant,cfg.e016_protect)
    residual_mask=torch.ones(T,dtype=torch.bool,device=tokens.device)
    residual_mask[original]=False
    residual=torch.arange(T,device=tokens.device)[residual_mask]
    demand=_demand(
        attention, middle_attention if variant=="E016_HIER" else None,
        cfg.e016_mid_mix)
    added,gain_sum=_block_marginal(
        affinity,demand,kept,residual,b,cfg.e016_block)
    selected=torch.cat((kept,added)).sort().values
    if selected.numel()!=K:
        raise RuntimeError("E016 stage1 output must retain exactly 172")
    return selected,original,added,removed,dict(
        budget=b,utility=z,gain_sum=gain_sum,risk_sum=risk_sum,
        score_type=variant+"-CLIP-APPROX-BLOCK"+str(cfg.e016_block))

def coverage_diagnostics(tokens,attention,selected,mid=None):
    """Auditable proxies; should be used ONLY when E014_AUDIT=True.

    These are not task-grounded factual evidence or accuracy guarantees.
    """
    _validate(tokens,attention,mid)
    if selected.numel()!=K:
        raise ValueError("selected set must have 172 IDs")
    a=attention.float().clamp_min(0.)
    raw=_base_similarity(tokens)
    cov=raw.index_select(1,selected).amax(dim=1)
    occupied=((selected//GRID)//6)*4+((selected%GRID)//6)
    top64=torch.topk(a,k=64).indices
    top128=torch.topk(a,k=128).indices
    return {
        "salience_mass": a.index_select(0,selected).sum()/
                         a.sum().clamp_min(1e-9),
        "weighted_representational_coverage":
           (a*cov).sum()/a.sum().clamp_min(1e-9),
        "unweighted_representational_coverage":cov.mean(),
        "worst_10pct_coverage":torch.quantile(cov,.1),
        "top64_retained_count":torch.isin(top64,selected).sum(),
        "top128_retained_count":torch.isin(top128,selected).sum(),
        "spatial_region_occupancy_16":occupied.unique().numel(),
    }
