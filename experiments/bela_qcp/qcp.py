"""Query-Aware Coverage Preservation (QCP) — first-cut boundary swaps only.

After native RDRIFT/Q_ATTN produces keep set A of size K, optionally swap a
few boundary members with same-capacity donors from alive set S so that
query-weighted Stage1 coverage of entry set S0 is better preserved:

  C(A) = sum_{i in S0} q_i * max_{j in A} s_ij

Swap u∈A for v∈S\\A only when:
  ΔC > λ * (r(u) - r(v)) + τ

r = minmax native ranking score; q_i from same-cut question→vision mass;
s_ij = nonnegative cosine on Stage1 Dual COVER features (full 576 table).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QCPConfig:
    max_swaps: int = 2
    donor_pool: int = 16
    drop_pool: int = 12
    lambda_rank: float = 0.5
    tau: float = 1e-4
    min_delta_c: float = 0.0


@dataclass
class QCPResult:
    keep_slots: torch.Tensor
    base_slots: torch.Tensor
    swaps: int
    diagnostics: Optional[list]


def _minmax01(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    if float((hi - lo).clamp_min(0).item()) < 1e-12:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


@torch.no_grad()
def select_qcp(
    *,
    keep_slots: torch.Tensor,
    ranking_score: torch.Tensor,
    q_attn: torch.Tensor,
    alive_ids: torch.Tensor,
    entry_ids: torch.Tensor,
    entry_features: torch.Tensor,
    config: QCPConfig = QCPConfig(),
    record_diagnostics: bool = False,
) -> QCPResult:
    if keep_slots.ndim != 1 or ranking_score.ndim != 1 or q_attn.ndim != 1:
        raise ValueError("keep/rank/q must be 1-D")
    n = int(alive_ids.numel())
    k = int(keep_slots.numel())
    if ranking_score.numel() != n or q_attn.numel() != n:
        raise ValueError("alive tensors misaligned")
    if entry_features.ndim != 2 or entry_features.shape[0] < 1:
        raise ValueError("entry_features must be [Nfeat, D]")
    if k == 0 or k == n or config.max_swaps <= 0:
        base = keep_slots.long().sort().values
        return QCPResult(base, base, 0, [] if record_diagnostics else None)

    device = alive_ids.device
    base = keep_slots.long().sort().values
    keep = base.clone()
    r = _minmax01(ranking_score.float())
    q = q_attn.float().clamp_min(0)
    entry_ids = entry_ids.long().to(device)
    alive_ids = alive_ids.long()

    id_to_slot = {int(pid.item()): i for i, pid in enumerate(alive_ids)}
    full_table = (
        int(entry_features.shape[0]) >= 576
        and int(entry_ids.max().item()) < int(entry_features.shape[0])
    )
    if full_table:
        full = entry_features.float().to(device)
        alive_feat = full.index_select(0, alive_ids)
        s0_slots = [id_to_slot[int(pid)] for pid in entry_ids.tolist() if int(pid) in id_to_slot]
        if not s0_slots:
            return QCPResult(base, base, 0, [] if record_diagnostics else None)
        s0_slots_t = torch.tensor(s0_slots, device=device, dtype=torch.long)
        s0_feat = alive_feat.index_select(0, s0_slots_t)
    else:
        if entry_ids.numel() != entry_features.shape[0]:
            raise ValueError("entry_ids / entry_features misaligned")
        s0_slots, s0_feat_idx = [], []
        for ei, pid in enumerate(entry_ids.tolist()):
            if pid in id_to_slot:
                s0_slots.append(id_to_slot[pid])
                s0_feat_idx.append(ei)
        if not s0_slots:
            return QCPResult(base, base, 0, [] if record_diagnostics else None)
        s0_slots_t = torch.tensor(s0_slots, device=device, dtype=torch.long)
        s0_feat = entry_features.float().index_select(
            0, torch.tensor(s0_feat_idx, device=entry_features.device, dtype=torch.long)
        ).to(device)
        alive_feat = torch.zeros(n, s0_feat.shape[-1], device=device, dtype=torch.float32)
        alive_feat[s0_slots_t] = s0_feat

    ref_f = F.normalize(s0_feat, dim=-1, eps=1e-12)
    all_f = F.normalize(alive_feat, dim=-1, eps=1e-12)
    sim = (ref_f @ all_f.T).clamp(min=0.0, max=1.0)  # [n_s0, N]

    q_s0 = q.index_select(0, s0_slots_t).clamp_min(0)
    if float(q_s0.sum().item()) <= 0:
        q_s0 = torch.ones_like(q_s0)
    q_s0 = q_s0 / q_s0.sum().clamp_min(1e-12)

    def coverage(keep_local: torch.Tensor) -> torch.Tensor:
        return (q_s0 * sim.index_select(1, keep_local).max(dim=1).values).sum()

    history = [] if record_diagnostics else None
    swaps = 0
    for _ in range(config.max_swaps):
        c_now = coverage(keep)
        active = torch.zeros(n, dtype=torch.bool, device=device)
        active[keep] = True
        outside = torch.where(~active)[0]
        if outside.numel() == 0:
            break

        r_keep = r.index_select(0, keep)
        drop_local = torch.argsort(r_keep, descending=False, stable=True)[
            : min(config.drop_pool, k)
        ]
        q_out = q.index_select(0, outside)
        donor_order = torch.argsort(q_out, descending=True, stable=True)[
            : min(config.donor_pool, int(outside.numel()))
        ]
        donors = outside.index_select(0, donor_order)
        drops = keep.index_select(0, drop_local)
        n_d, n_v = int(drops.numel()), int(donors.numel())
        if n_d == 0 or n_v == 0:
            break

        # Vectorized: for each (drop u, donor v), coverage of (keep\{u})∪{v}.
        # keep_wo[d, :] = keep without drops[d] (pad with first keep member then overwrite).
        # Use: max(sim[:, keep_wo], sim[:, v]) for each pair.
        sel = sim.index_select(1, keep)  # [n_s0, K]
        top1, top1_pos = sel.max(dim=1)  # [n_s0]
        if k > 1:
            masked = sel.clone()
            masked.scatter_(1, top1_pos[:, None], -1.0)
            top2 = masked.max(dim=1).values.clamp_min(0)
        else:
            top2 = torch.zeros_like(top1)

        # Map drop slot -> position in keep
        # drop_local already positions in keep
        # without_u[a, d] = coverage ref after removing keep[drop_local[d]]
        ar = torch.arange(n_d, device=device)
        # top1_pos is index into keep; compare to drop_local
        # without[d, a] = top2[a] if top1_pos[a]==drop_local[d] else top1[a]
        without = torch.where(
            top1_pos[None, :] == drop_local[:, None],
            top2[None, :],
            top1[None, :],
        )  # [n_d, n_s0]
        donor_sims = sim.index_select(1, donors).T  # [n_v, n_s0]
        new_cov = (torch.maximum(without[:, None, :], donor_sims[None, :, :]) * q_s0[None, None, :]).sum(-1)
        # [n_d, n_v]
        delta_c = new_cov - c_now
        rank_pen = config.lambda_rank * (
            r.index_select(0, drops)[:, None] - r.index_select(0, donors)[None, :]
        )
        ok = (delta_c > rank_pen + config.tau) & (delta_c >= config.min_delta_c)
        objective = (delta_c - rank_pen).masked_fill(~ok, float("-inf"))
        flat = objective.reshape(-1)
        best = flat.argmax()
        if not bool(torch.isfinite(flat[best])):
            break
        di = int((best // n_v).item())
        dj = int((best % n_v).item())
        u = int(drops[di].item())
        v = int(donors[dj].item())
        keep = torch.where(keep == u, keep.new_tensor(v), keep).sort().values
        swaps += 1
        if history is not None:
            history.append({
                "drop_original_id": int(alive_ids[u].item()),
                "add_original_id": int(alive_ids[v].item()),
                "delta_coverage": float(delta_c[di, dj].item()),
                "rank_penalty": float(rank_pen[di, dj].item()),
                "objective": float(objective[di, dj].item()),
            })

    if keep.numel() != k or int(keep.unique().numel()) != k:
        raise RuntimeError("QCP must preserve unique K")
    return QCPResult(keep.sort().values, base, swaps, history)
