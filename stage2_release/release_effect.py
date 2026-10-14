"""
Core module: Exact set-level release effect computation.

Key insight: A token's contribution is proportional to A_{t,j} * ||v_j - o_t||,
not just attention mass. This captures "information saturation".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ReleaseEffectOutput:
    """Output of release effect computation."""

    epsilon_per_k: Dict[int, float]
    token_contributions: torch.Tensor
    chosen_k: int
    keep_indices: torch.Tensor


class ReleaseEffect:
    """Computes exact layer-internal deletion effect with attention renormalization."""

    def __init__(
        self,
        k_grid: Tuple[int, ...] = (96, 64, 48, 32, 16, 0),
        eps: float = 1e-6,
        compute_dtype: torch.dtype = torch.float32,
    ):
        self.k_grid = k_grid
        self.eps = eps
        self.compute_dtype = compute_dtype

    @torch.no_grad()
    def compute(
        self,
        q_rows: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        vis_idx: torch.Tensor,
        o_proj_weight: torch.Tensor,
        h_rows: torch.Tensor,
        tau: float = 0.05,
        row_weight: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> ReleaseEffectOutput:
        h, t, d_h = q_rows.shape
        m = int(vis_idx.numel())
        device = q_rows.device

        q = q_rows.to(self.compute_dtype)
        k = k_all.to(self.compute_dtype)
        v = v_all.to(self.compute_dtype)

        scores = torch.einsum("htd,hnd->htn", q, k) / (d_h ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask.to(self.compute_dtype)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.einsum("htn,hnd->htd", attn_weights, v)

        attn_vis = attn_weights[:, :, vis_idx]
        v_vis = v[:, vis_idx]

        contrib = attn_vis.unsqueeze(-1) * (
            v_vis.unsqueeze(1) - attn_output.unsqueeze(2)
        )

        if row_weight is None:
            row_weight = torch.ones(t, device=device, dtype=self.compute_dtype) / t
        else:
            row_weight = row_weight.to(device=device, dtype=self.compute_dtype)
            row_weight = row_weight / row_weight.sum().clamp_min(self.eps)

        token_contrib = (contrib.norm(dim=-1) * row_weight.view(1, t, 1)).sum((0, 1))
        order = token_contrib.argsort()

        contrib_ordered = contrib[:, :, order]
        attn_ordered = attn_vis[:, :, order]
        contrib_cum = contrib_ordered.cumsum(dim=2)
        attn_cum = attn_ordered.cumsum(dim=2)

        h_norm = h_rows.to(self.compute_dtype).norm(dim=-1).clamp_min(self.eps)
        o_w = o_proj_weight.to(self.compute_dtype)

        epsilon_dict: Dict[int, float] = {}
        for k_keep in sorted({min(int(kk), m) for kk in self.k_grid}, reverse=True):
            r = m - k_keep
            if r == 0:
                epsilon_dict[k_keep] = 0.0
                continue

            denom = (1.0 - attn_cum[:, :, r - 1]).clamp_min(self.eps)
            delta_per_head = contrib_cum[:, :, r - 1] / denom.unsqueeze(-1)
            delta_concat = delta_per_head.permute(1, 0, 2).reshape(t, h * d_h)
            delta_out = delta_concat @ o_w.t()
            eps_val = ((delta_out.norm(dim=-1) / h_norm) * row_weight).sum()
            epsilon_dict[k_keep] = float(eps_val.item())

        # Largest K with ε(R) ≤ tau (most tokens kept among feasible budgets).
        sorted_ks = sorted(epsilon_dict.keys())
        epsilon_tensor = torch.tensor(
            [epsilon_dict[kk] for kk in sorted_ks], device=device
        )
        valid_mask = epsilon_tensor <= tau
        if valid_mask.any():
            chosen_idx = int(valid_mask.nonzero()[-1].item())
            chosen_k = sorted_ks[chosen_idx]
        else:
            # No budget meets τ: keep all tokens (ε = 0 by definition).
            chosen_k = m
            epsilon_dict[m] = 0.0

        keep_local = order[m - chosen_k :].sort().values

        return ReleaseEffectOutput(
            epsilon_per_k=epsilon_dict,
            token_contributions=token_contrib,
            chosen_k=chosen_k,
            keep_indices=keep_local,
        )


def compute_release_effect(
    q_rows: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    vis_idx: torch.Tensor,
    o_proj_weight: torch.Tensor,
    h_rows: torch.Tensor,
    tau: float = 0.05,
    k_grid: Tuple[int, ...] = (96, 64, 48, 32, 16, 0),
    **kwargs,
) -> ReleaseEffectOutput:
    """Functional API for release effect computation."""
    calculator = ReleaseEffect(k_grid=k_grid)
    return calculator.compute(
        q_rows, k_all, v_all, vis_idx, o_proj_weight, h_rows, tau, **kwargs
    )
