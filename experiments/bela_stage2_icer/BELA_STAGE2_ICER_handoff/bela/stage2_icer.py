"""BELA ICER: Interaction-Conditioned Evidence Refinement (research prototype).

This is a standalone selector core, NOT a production LLaVA hook. It preserves
the existing Q-Attn/RDRIFT scorer as the baseline and adds a bounded,
budget-neutral set of swaps for coverage debt. Positive drift can only help
shortlist donors at early pruning boundaries; it is disabled at the final
boundary. All IDs are original Stage-1 patch IDs.

Inputs:
  q_attention: [N], text-query -> visual-key probability mass, normalized over
               ALL causally visible keys before visual positions are sliced.
  descriptors: [N,D], fixed Stage-1 feature descriptors aligned to active IDs.
  original_ids: [N], unique original patch IDs.
  k: number to retain (must remain the already-planned K; this module never
     changes K, pruning boundaries, or W).
  ranking_score: optional exact baseline selector score (Q-Attn or the real,
                 source-verified historical RDRIFT implementation).
  positive_drift: optional source-verified positive trend, aligned with IDs.

The score fusion is deliberately not a new learned "future utility" claim.
This prototype performs local swaps only when coverage improves and the
retained Q-Attn mass remains above a guard. Validate on model-level accuracy
and latency before adopting.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ICERConfig:
    max_swaps: int = 3
    donor_pool: int = 24
    drop_pool: int = 16
    min_qmass_ratio: float = 0.97
    drift_weight: float = 0.20
    query_weight: float = 0.25
    min_coverage_gain: float = 1e-5
    normalize_descriptors: bool = True
    center_descriptors: bool = False
    validate_finite: bool = True

    def __post_init__(self):
        if min(self.max_swaps, self.donor_pool, self.drop_pool) < 0:
            raise ValueError("search sizes must be nonnegative")
        if not 0 < self.min_qmass_ratio <= 1:
            raise ValueError("min_qmass_ratio must be in (0,1]")
        if not 0 <= self.drift_weight <= 1 or not 0 <= self.query_weight <= 1:
            raise ValueError("weights must be in [0,1]")
        if self.min_coverage_gain < 0:
            raise ValueError("min_coverage_gain must be nonnegative")


@dataclass
class ICERResult:
    keep_slots: torch.Tensor       # slots in current active-token array, sorted
    base_slots: torch.Tensor       # baseline top-k slots, sorted
    keep_original_ids: torch.Tensor
    swaps: int
    diagnostics: Optional[list] = None


def _stable_desc(score: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    # Deterministic tie-break: descending score, then ascending original ID.
    by_id = torch.argsort(ids, stable=True)
    by_score = torch.argsort(score[by_id], descending=True, stable=True)
    return by_id[by_score]


def _validate(q, desc, ids, k, rank, drift, cfg):
    if q.ndim != 1 or desc.ndim != 2 or ids.ndim != 1:
        raise ValueError("expected q[N], descriptors[N,D], ids[N]")
    n = q.numel()
    if n == 0 or desc.shape[0] != n or ids.numel() != n or not (0 < k <= n):
        raise ValueError("misaligned input or invalid k")
    if desc.shape[1] == 0:
        raise ValueError("empty descriptor dimension")
    for name, x in (("rank", rank), ("drift", drift)):
        if x is not None and x.shape != q.shape:
            raise ValueError(f"{name} must have shape [N]")
    xs = [desc, ids] + ([rank] if rank is not None else []) + ([drift] if drift is not None else [])
    if any(x.device != q.device for x in xs):
        raise ValueError("all tensors must share one device")
    if cfg.validate_finite:
        for x in [q, desc, rank, drift]:
            if x is not None and not bool(torch.isfinite(x).all()):
                raise ValueError("nonfinite input")
        if bool((q < 0).any()):
            raise ValueError("Q-Attn mass must be nonnegative")
        if torch.unique(ids).numel() != n:
            raise ValueError("original IDs must be unique")


@torch.no_grad()
def select_icer(
    q_attention: torch.Tensor,
    descriptors: torch.Tensor,
    original_ids: torch.Tensor,
    k: int,
    *,
    ranking_score: Optional[torch.Tensor] = None,
    positive_drift: Optional[torch.Tensor] = None,
    final_boundary: bool = False,
    config: ICERConfig = ICERConfig(),
    record_diagnostics: bool = False,
) -> ICERResult:
    _validate(q_attention, descriptors, original_ids, k,
              ranking_score, positive_drift, config)
    q = q_attention.float().clamp_min(0)
    rank = q if ranking_score is None else ranking_score.float()
    ids = original_ids.long()
    base = _stable_desc(rank, ids)[:k].sort().values
    keep = base.clone()
    if k == q.numel() or config.max_swaps == 0 or not config.donor_pool or not config.drop_pool:
        return ICERResult(keep, base, ids[keep], 0, [] if record_diagnostics else None)

    f = descriptors.float()
    if config.center_descriptors:
        f = f - f.mean(0, keepdim=True)
    if config.normalize_descriptors:
        f = F.normalize(f, dim=-1, eps=1e-8)
    # Nonnegative cosine/feature similarity used as a cheap coverage proxy.
    sim = (f @ f.T).clamp(min=0, max=1)
    qtotal = q.sum().clamp_min(1e-12)
    weights = (1-config.query_weight)/q.numel() + config.query_weight*q/qtotal
    weights = weights / weights.sum().clamp_min(1e-12)
    base_qmass = q[base].sum().clamp_min(1e-12)

    if positive_drift is None or final_boundary:
        drift = torch.zeros_like(q)
    else:
        drift = positive_drift.float().clamp_min(0)
        drift = drift / drift.max().clamp_min(1e-12)

    history = [] if record_diagnostics else None
    swaps = 0
    for _ in range(config.max_swaps):
        active = torch.zeros(q.numel(), dtype=torch.bool, device=q.device)
        active[keep] = True
        outside = torch.arange(q.numel(), device=q.device)[~active]
        if outside.numel() == 0:
            break

        selected_sim = sim[:, keep]  # [N,K]
        top1, top1_pos = selected_sim.max(dim=1)
        if k > 1:
            masked = selected_sim.clone()
            masked.scatter_(1, top1_pos[:, None], -1)
            top2 = masked.max(dim=1).values.clamp_min(0)
        else:
            top2 = torch.zeros_like(top1)
        current_cov = (weights * top1).sum()
        # Coverage after removing each current selected slot, for all reference tokens.
        ar_k = torch.arange(k, device=q.device)
        without = torch.where(top1_pos[None, :] == ar_k[:, None],
                              top2[None, :], top1[None, :])  # [K,N]
        cov_without = (without * weights[None, :]).sum(-1)
        loss_if_drop = current_cov - cov_without

        sim_out = sim[:, outside].T  # [N_out,N_ref]
        add_gain = (torch.relu(sim_out - top1[None, :]) * weights[None, :]).sum(-1)
        donor_priority = add_gain + (0.0 if final_boundary else config.drift_weight / max(k,1))*drift[outside]
        donor_order = _stable_desc(donor_priority, ids[outside])
        donors = outside[donor_order[:min(config.donor_pool, outside.numel())]]

        # Candidate drop set = low relevance and low marginal coverage loss.
        drop_priority = 0.5*q[keep]/base_qmass + loss_if_drop
        drop_positions = torch.argsort(drop_priority, stable=True)[:min(config.drop_pool, k)]
        old_slots = keep[drop_positions]

        # Exact weighted facility-coverage delta for one-for-one swaps.
        minus = without[drop_positions]  # [D,N_ref]
        donor_sims = sim[:, donors].T  # [N_donor,N_ref]
        replacement_cov = (torch.maximum(minus[:, None, :], donor_sims[None, :, :]) *
                           weights[None, None, :]).sum(-1)
        delta_cov = replacement_cov - current_cov
        q_new = q[keep].sum() - q[old_slots][:, None] + q[donors][None, :]
        q_guard = q_new >= config.min_qmass_ratio*base_qmass - 1e-8
        delta_drift = (drift[donors][None, :] - drift[old_slots][:, None]) / max(k,1)
        utility = delta_cov + (0.0 if final_boundary else config.drift_weight)*delta_drift
        valid = q_guard & (delta_cov >= -1e-8) & (utility > config.min_coverage_gain)
        utility = utility.masked_fill(~valid, -torch.inf)
        best_flat = utility.flatten().argmax()
        best = utility.flatten()[best_flat]
        if not bool(torch.isfinite(best)):
            break
        di = best_flat // donors.numel()
        dj = best_flat % donors.numel()
        outgoing, incoming = old_slots[di], donors[dj]
        keep = torch.where(keep == outgoing, incoming, keep).sort().values
        swaps += 1
        if history is not None:
            history.append({
                "drop_original_id": int(ids[outgoing].item()),
                "add_original_id": int(ids[incoming].item()),
                "delta_coverage": float(delta_cov[di, dj].item()),
                "delta_drift": float(delta_drift[di, dj].item()),
                "objective": float(best.item()),
                "qmass_ratio": float((q_new[di,dj]/base_qmass).item()),
            })

    return ICERResult(keep, base, ids[keep], swaps, history)


@torch.no_grad()
def singleton_perturbation(attention: torch.Tensor, values: torch.Tensor,
                           visual_key_positions: torch.Tensor, eps: float = 1e-8):
    """CoRePrune-style frozen-QKV singleton distortion diagnostic.

    attention [H,Q,T] must be normalized over ALL causally visible keys;
    values [H,T,D] include both text and visual keys; positions are visual keys.
    Returns [N_visual]. This is a diagnostic baseline, not a future-loss oracle.
    """
    if attention.ndim != 3 or values.ndim != 3:
        raise ValueError("attention [H,Q,T], values [H,T,D] expected")
    if attention.shape[0] != values.shape[0] or attention.shape[-1] != values.shape[1]:
        raise ValueError("head/key dimensions mismatch")
    a, v = attention.float(), values.float()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(v).all()):
        raise ValueError("nonfinite attention/value")
    if bool((a < 0).any()) or not torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=2e-3, rtol=2e-3):
        raise ValueError("attention must be normalized over all visible keys")
    ix = torch.as_tensor(visual_key_positions, dtype=torch.long, device=a.device)
    if ix.ndim != 1 or ix.numel() == 0 or bool((ix < 0).any()) or bool((ix >= a.shape[-1]).any()):
        raise ValueError("invalid visual positions")
    if torch.unique(ix).numel() != ix.numel():
        raise ValueError("duplicate visual key positions")
    av = a.index_select(-1, ix)
    if bool((av >= 1-eps).any()):
        raise ValueError("single visual key has all attention mass")
    y = torch.einsum("hqt,htd->hqd", a, v)
    vv = v.index_select(1, ix)
    delta = (vv[:, None, :, :] - y[:, :, None, :]).norm(dim=-1)
    return (av/(1-av).clamp_min(eps)*delta).mean((0,1))


@torch.no_grad()
def set_perturbation(attention: torch.Tensor, values: torch.Tensor,
                     removed_visual_positions: torch.Tensor, eps: float = 1e-8):
    """CoRePrune-style set-conditioned distortion for a fixed attention layer."""
    if attention.ndim != 3 or values.ndim != 3:
        raise ValueError("attention [H,Q,T], values [H,T,D] expected")
    a, v = attention.float(), values.float()
    if attention.shape[0] != values.shape[0] or attention.shape[-1] != values.shape[1]:
        raise ValueError("head/key dimensions mismatch")
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(v).all()):
        raise ValueError("nonfinite attention/value")
    if bool((a < 0).any()) or not torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=2e-3, rtol=2e-3):
        raise ValueError("attention must be normalized over all visible keys")
    ix = torch.as_tensor(removed_visual_positions, dtype=torch.long, device=a.device)
    if ix.ndim != 1 or ix.numel() == 0 or bool((ix < 0).any()) or bool((ix >= a.shape[-1]).any()):
        raise ValueError("invalid removed positions")
    if torch.unique(ix).numel() != ix.numel():
        raise ValueError("duplicate removed positions")
    ai = a.index_select(-1, ix)
    mass = ai.sum(-1)
    if bool((mass >= 1-eps).any()):
        raise ValueError("removed set exhausts attention mass")
    y = torch.einsum("hqt,htd->hqd", a, v)
    removed_value = torch.einsum("hqs,hsd->hqd", ai, v.index_select(1, ix))
    delta = (mass[..., None]*y - removed_value).norm(dim=-1)/(1-mass).clamp_min(eps)
    return delta.mean()
