"""BELA Stage1: Question-Anchored Dual COVER.

Research candidate; no GPU accuracy improvement has been established.

Start from native Dual COVER patch IDs. Replace at most max_swaps
original patch IDs before LLM prefill, without changing K.

The query score is a proxy. LLaVA projected visual embeddings
and LLM text embeddings are NOT guaranteed to be semantically aligned.
Matched shuffled-question and random controls are mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

FeatureInput = Union[torch.Tensor, Sequence[torch.Tensor]]


@dataclass(frozen=True)
class AnchorConfig:
    max_swaps: int = 8
    donor_pool: int = 16
    drop_pool: int = 12
    min_raw_std: float = 0.008
    min_query_gain: float = 0.08
    min_novelty: float = 0.05
    max_coverage_loss: float = 0.005
    coverage_weight: float = 0.5
    saliency_guard_slots: int = 0


@dataclass
class AnchorResult:
    keep_ids: torch.Tensor
    native_ids: torch.Tensor
    swaps: int
    reason: str
    records: Optional[list]


def _check_ids(
    ids: torch.Tensor, n: int, label: str
) -> torch.Tensor:
    if ids.ndim != 1 or ids.dtype not in (
        torch.int32, torch.int64
    ):
        raise ValueError(f"{label}: expected 1-D integer IDs")

    ids = ids.long()
    if ids.numel() and (
        (ids < 0).any() or (ids >= n).any()
    ):
        raise ValueError(f"{label}: patch ID out of range")

    if ids.unique().numel() != ids.numel():
        raise ValueError(f"{label}: duplicate patch IDs")

    return ids


def _features(
    x: FeatureInput, n: int, device: torch.device
) -> list:
    views = [x] if isinstance(x, torch.Tensor) else list(x)
    if not views:
        raise ValueError("visual features required")

    out = []
    for v in views:
        if (
            v.ndim != 2
            or v.shape[0] != n
            or not torch.is_floating_point(v)
        ):
            raise ValueError("features must be [N,D]")

        v = v.detach().to(
            device=device, dtype=torch.float32
        )
        if not torch.isfinite(v).all():
            raise ValueError("NaN/Inf visual feature")

        out.append(F.normalize(v, dim=-1, eps=1e-12))
    return out


def _query_from_embeds(
    visual_llm_embeds: torch.Tensor,
    question_embeds: torch.Tensor,
) -> torch.Tensor:
    """Experimental LLaVA projector/text embedding proxy."""
    if (
        visual_llm_embeds.ndim != 2
        or question_embeds.ndim != 2
    ):
        raise ValueError("embeddings must be 2-D")

    if question_embeds.shape[0] == 0:
        return torch.empty(
            0, device=visual_llm_embeds.device
        )

    if (
        visual_llm_embeds.shape[1]
        != question_embeds.shape[1]
    ):
        raise ValueError("embedding dimension mismatch")

    device = visual_llm_embeds.device
    v = visual_llm_embeds.detach().float()
    t = question_embeds.detach().to(
        device=device, dtype=torch.float32
    )

    if not torch.isfinite(v).all() or not torch.isfinite(t).all():
        raise ValueError("NaN/Inf embedding")

    text = F.normalize(
        t.mean(dim=0, keepdim=True), dim=-1, eps=1e-12
    )
    return (
        F.normalize(v, dim=-1, eps=1e-12) @ text.T
    ).squeeze(1)


@torch.no_grad()
def select_question_anchored(
    *,
    native_keep_ids: torch.Tensor,
    visual_metric: FeatureInput,
    visual_llm_embeds: Optional[torch.Tensor] = None,
    question_embeds: Optional[torch.Tensor] = None,
    raw_query_scores: Optional[torch.Tensor] = None,
    native_saliency: Optional[torch.Tensor] = None,
    config: AnchorConfig = AnchorConfig(),
    record: bool = False,
) -> AnchorResult:
    """Return same-K ORIGINAL patch IDs.

    No future Decoder states, labels, or second LLM forward.
    """
    if config.saliency_guard_slots < 0:
        raise ValueError("invalid saliency guard")

    if (
        config.max_swaps < 0
        or config.donor_pool < 1
        or config.drop_pool < 1
    ):
        raise ValueError("invalid swap/pool limits")

    if (
        config.min_query_gain < 0
        or config.max_coverage_loss < 0
    ):
        raise ValueError("invalid thresholds")

    if isinstance(visual_metric, torch.Tensor):
        n = int(visual_metric.shape[0])
        device = visual_metric.device
    else:
        if not len(visual_metric):
            raise ValueError("missing visual features")
        n = int(visual_metric[0].shape[0])
        device = visual_metric[0].device

    native = _check_ids(
        native_keep_ids.to(device), n, "native_keep_ids"
    ).sort().values

    k = native.numel()
    records = [] if record else None

    def untouched(reason: str) -> AnchorResult:
        return AnchorResult(
            native.clone(), native.clone(),
            0, reason, records
        )

    if k == 0 or k == n or config.max_swaps == 0:
        return untouched("no_capacity_to_change")

    aligned = raw_query_scores is not None
    embedded = (
        visual_llm_embeds is not None
        or question_embeds is not None
    )
    if aligned and embedded:
        raise ValueError("choose query scores OR embeddings")

    if aligned:
        raw = raw_query_scores.detach().to(
            device=device, dtype=torch.float32
        )
    elif (
        visual_llm_embeds is not None
        and question_embeds is not None
    ):
        raw = _query_from_embeds(
            visual_llm_embeds, question_embeds
        )
    else:
        return untouched("missing_question_signal")

    if raw.numel() == 0:
        return untouched("empty_question")

    if (
        raw.ndim != 1
        or raw.numel() != n
        or not torch.isfinite(raw).all()
    ):
        raise ValueError("query score must be finite [N]")

    std = raw.std(unbiased=False)
    if float(std) < config.min_raw_std:
        return untouched("unreliable_flat_query")

    q = torch.sigmoid(
        (
            (raw - raw.median())
            / (2.0 * std.clamp_min(1e-12))
        ).clamp(-8, 8)
    )

    # Optional saliency guard, OFF by default.
    protected_ids = torch.empty(
        0, dtype=torch.long, device=device
    )
    if config.saliency_guard_slots:
        if native_saliency is None:
            raise ValueError("saliency guard requires scores")
        sal = native_saliency.detach().to(
            device=device, dtype=torch.float32
        )
        if (
            sal.ndim != 1
            or sal.numel() != n
            or not torch.isfinite(sal).all()
        ):
            raise ValueError("saliency must be finite [N]")

        order = torch.argsort(
            sal.index_select(0, native),
            descending=True, stable=True
        )
        protected_ids = native.index_select(
            0, order[:min(config.saliency_guard_slots, k)]
        )

    views = _features(visual_metric, n, device)
    sim = torch.stack([
        (v @ v.T).clamp(0.0, 1.0)
        for v in views
    ]).mean(dim=0)

    keep = native.clone()
    max_steps = min(config.max_swaps, k, n-k)
    swaps = 0

    for _ in range(max_steps):
        active = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        active[keep] = True
        outside = torch.nonzero(
            ~active, as_tuple=False
        ).flatten()
        if outside.numel() == 0:
            break

        novelty = 1 - sim[outside][:, keep].amax(dim=1)
        priority = q[outside] * novelty

        donors = outside[torch.argsort(
            priority, descending=True, stable=True
        )[:config.donor_pool]]

        inside_sim = sim[keep][:, keep].clone()
        inside_sim.fill_diagonal_(0)
        redundancy = (
            inside_sim.amax(dim=1)
            if k > 1
            else torch.zeros_like(q[keep])
        )

        drop_priority = (
            (1 - q[keep]) + 0.5 * redundancy
        )

        if protected_ids.numel():
            mask = torch.isin(keep, protected_ids)
            drop_priority = drop_priority.masked_fill(
                mask, float("-inf")
            )

        order = torch.argsort(
            drop_priority, descending=True, stable=True
        )
        order = order[
            torch.isfinite(drop_priority[order])
        ]
        drop_positions = order[:config.drop_pool]

        if not drop_positions.numel() or not donors.numel():
            break

        drops = keep[drop_positions]

        selected_sims = sim[:, keep]
        top1, top1_pos = selected_sims.max(dim=1)

        if k > 1:
            masked = selected_sims.clone()
            masked.scatter_(1, top1_pos[:, None], -1)
            top2 = masked.max(dim=1).values.clamp_min(0)
        else:
            top2 = torch.zeros_like(top1)

        without = torch.where(
            top1_pos.unsqueeze(0)
            == drop_positions.unsqueeze(1),
            top2.unsqueeze(0),
            top1.unsqueeze(0),
        )

        donor_sims = sim[:, donors].T
        old_cov = top1.mean()

        new_cov = torch.maximum(
            without[:, None, :],
            donor_sims[None, :, :]
        ).mean(dim=-1)

        delta_cov = new_cov - old_cov
        query_gain = (
            q[donors][None, :]
            - q[drops][:, None]
        )

        donor_novelty = (
            1 - sim[donors][:, keep].amax(dim=1)
        )

        accepted = (
            (query_gain >= config.min_query_gain)
            & (
                donor_novelty[None, :]
                >= config.min_novelty
            )
            & (
                delta_cov
                >= -config.max_coverage_loss
            )
        )

        objective = (
            query_gain * donor_novelty[None, :]
            + config.coverage_weight * delta_cov
        ).masked_fill(~accepted, float("-inf"))

        flat = objective.flatten()
        index = int(flat.argmax())

        if not torch.isfinite(flat[index]):
            break

        nv = donors.numel()
        di, dj = divmod(index, nv)

        removed = int(drops[di])
        added = int(donors[dj])

        if records is not None:
            records.append({
                "removed_patch": removed,
                "added_patch": added,
                "query_gain": float(
                    query_gain[di, dj]
                ),
                "coverage_delta": float(
                    delta_cov[di, dj]
                ),
                "donor_novelty": float(
                    donor_novelty[dj]
                ),
            })

        keep = torch.cat([
            keep[keep != removed],
            keep.new_tensor([added]),
        ]).sort().values
        swaps += 1

    if (
        keep.numel() != k
        or keep.unique().numel() != k
    ):
        raise AssertionError("fixed K violated")

    return AnchorResult(
        keep, native, swaps,
        "swapped" if swaps else "no_safe_query_anchor",
        records,
    )
