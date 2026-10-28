"""BELA Stage1 Question-Anchored Dual COVER: training-free, fixed-K entry swaps.

This is a proposed research arm, NOT a measured improvement.
The native BELA Dual COVER result is always computed first. This selector
can replace a limited number of its ORIGINAL patch IDs before LLM prefill.

Query evidence is a proxy computed without ground-truth answers or future
decoder states. In LLaVA the optional adapter uses the existing mm_projector
and LLM input embeddings; that cosine is NOT guaranteed to be grounded.
A matched shuffled-question control is mandatory.

The selector does not call the vision tower or language decoder, and is
compatible with exactly the features that the calling integration supplies.
No global variables, no modification of Stage2, no per-sample label access.
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
    # These are PREDECLARED pilot defaults, not established optimal values.
    min_raw_std: float = 0.008
    min_query_gain: float = 0.08
    min_novelty: float = 0.05
    max_coverage_loss: float = 0.005
    coverage_weight: float = 0.5
    saliency_guard_slots: int = 0  # 0=disabled; optional CLS-attention ablation only


@dataclass
class AnchorResult:
    keep_ids: torch.Tensor
    native_ids: torch.Tensor
    swaps: int
    reason: str
    records: Optional[list]


def _check_ids(ids: torch.Tensor, n: int, label: str) -> torch.Tensor:
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{label}: expected 1-D integer patch IDs")
    ids = ids.long()
    if ids.numel() and ((ids < 0).any() or (ids >= n).any()):
        raise ValueError(f"{label}: ID outside [0, {n})")
    if ids.unique().numel() != ids.numel():
        raise ValueError(f"{label}: duplicate IDs")
    return ids


def _features(x: FeatureInput, n: int, device: torch.device) -> list:
    views = [x] if isinstance(x, torch.Tensor) else list(x)
    if not views:
        raise ValueError("at least one Dual COVER visual descriptor required")
    out = []
    for v in views:
        if v.ndim != 2 or v.shape[0] != n or not torch.is_floating_point(v):
            raise ValueError("visual descriptors must have shape [N,D]")
        v = v.detach().to(device=device, dtype=torch.float32)
        if not torch.isfinite(v).all():
            raise ValueError("visual descriptor has NaN/Inf")
        out.append(F.normalize(v, dim=-1, eps=1e-12))
    return out


def _query_from_embeds(
    visual_llm_embeds: torch.Tensor,
    question_embeds: torch.Tensor,
) -> torch.Tensor:
    """Use existing LLaVA projector outputs and question-only LLM embeddings.

    NOTE: cosine alignment is a hypothesis; compare to shuffled questions.
    Do not pass conversation-template tokens, labels, or answer tokens.
    """
    if visual_llm_embeds.ndim != 2 or question_embeds.ndim != 2:
        raise ValueError("visual_llm_embeds and question_embeds must be 2-D")
    if question_embeds.shape[0] == 0:
        return torch.empty(0, device=visual_llm_embeds.device)
    if visual_llm_embeds.shape[1] != question_embeds.shape[1]:
        raise ValueError("LLM visual/text embedding dimensions mismatch")
    device = visual_llm_embeds.device
    v = visual_llm_embeds.detach().to(dtype=torch.float32)
    t = question_embeds.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(v).all() or not torch.isfinite(t).all():
        raise ValueError("query embeddings contain NaN/Inf")
    text = F.normalize(t.mean(dim=0, keepdim=True), dim=-1, eps=1e-12)
    return (F.normalize(v, dim=-1, eps=1e-12) @ text.T).squeeze(1)


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
    """Select no more than max_swaps outside-Stage1 patches, never change K.

    visual_metric: native Stage1 Dual feature descriptor [N,D] or a pair.
    visual_llm_embeds: full-grid output of EXISTING mm_projector, [N,H].
    question_embeds: QUESTION-ONLY LLM embedding vectors, [T,H].
    raw_query_scores: an independently validated aligned score for [N];
                      mutually exclusive with embedding inputs.
    Returns sorted ORIGINAL patch IDs. Existing Stage2 remains unchanged.
    """
    if config.saliency_guard_slots < 0:
        raise ValueError("saliency_guard_slots must be >= 0")
    if config.max_swaps < 0 or config.donor_pool < 1 or config.drop_pool < 1:
        raise ValueError("invalid swap/pool limits")
    if config.min_query_gain < 0 or config.max_coverage_loss < 0:
        raise ValueError("invalid constraint threshold")
    if isinstance(visual_metric, torch.Tensor):
        n, device = int(visual_metric.shape[0]), visual_metric.device
    else:
        if len(visual_metric) == 0:
            raise ValueError("no vision features")
        n, device = int(visual_metric[0].shape[0]), visual_metric[0].device
    native = _check_ids(native_keep_ids.to(device), n, "native_keep_ids")
    native = native.sort().values
    k = native.numel()
    records = [] if record else None

    def untouched(reason: str) -> AnchorResult:
        return AnchorResult(native.clone(), native.clone(), 0, reason, records)

    if k == 0 or k == n or config.max_swaps == 0:
        return untouched("no_capacity_to_change")

    aligned = raw_query_scores is not None
    embedded = visual_llm_embeds is not None or question_embeds is not None
    if aligned and embedded:
        raise ValueError("choose precomputed raw_query_scores OR embeddings")
    if aligned:
        raw = raw_query_scores.detach().to(device=device, dtype=torch.float32)
    elif visual_llm_embeds is not None and question_embeds is not None:
        raw = _query_from_embeds(visual_llm_embeds, question_embeds)
    else:
        return untouched("missing_question_signal")

    if raw.numel() == 0:
        return untouched("empty_question")
    if raw.ndim != 1 or raw.numel() != n or not torch.isfinite(raw).all():
        raise ValueError("query signal must be finite and aligned with N patches")
    std = raw.std(unbiased=False)
    if float(std) < config.min_raw_std:
        return untouched("unreliable_flat_query")

    # Do NOT min-max amplify near-constant similarities.
    q = torch.sigmoid(((raw - raw.median()) / (2.0 * std.clamp_min(1e-12))).clamp(-8, 8))
    # Saliency is a separate, explicitly controlled ablation; never silently infer it.
    protected_ids = torch.empty(0, dtype=torch.long, device=device)
    if config.saliency_guard_slots:
        if native_saliency is None:
            raise ValueError("saliency guard enabled but no native_saliency supplied")
        sal = native_saliency.detach().to(device=device, dtype=torch.float32)
        if sal.ndim != 1 or sal.numel() != n or not torch.isfinite(sal).all():
            raise ValueError("native_saliency must be finite and [N]")
        order = torch.argsort(sal.index_select(0, native), descending=True, stable=True)
        protected_ids = native.index_select(
            0, order[: min(config.saliency_guard_slots, k)]
        )
    views = _features(visual_metric, n, device)
    sim = torch.stack(
        [(view @ view.T).clamp(0.0, 1.0) for view in views]
    ).mean(dim=0)     # [N,N]; no future LLM states
    keep = native.clone()
    max_steps = min(config.max_swaps, k, n - k)
    swaps = 0
    for _ in range(max_steps):
        active = torch.zeros(n, device=device, dtype=torch.bool)
        active[keep] = True
        outside = torch.nonzero(~active, as_tuple=False).flatten()
        if outside.numel() == 0:
            break

        # Query-relevant and visually novel outside candidates.
        novelty = 1 - sim[outside][:, keep].amax(dim=1)
        candidate_priority = q[outside] * novelty
        donors = outside[torch.argsort(
            candidate_priority, descending=True, stable=True
        )[:config.donor_pool]]

        # Drop candidates with weak question support and visual redundancy.
        in_sim = sim[keep][:, keep].clone()
        in_sim.fill_diagonal_(0)
        redundant = in_sim.amax(dim=1) if k > 1 else torch.zeros_like(q[keep])
        drop_priority = (1 - q[keep]) + 0.5 * redundant
        if protected_ids.numel():
            protected_mask = torch.isin(keep, protected_ids)
            drop_priority = drop_priority.masked_fill(protected_mask, float("-inf"))
        drop_order = torch.argsort(
            drop_priority, descending=True, stable=True
        )
        drop_order = drop_order[torch.isfinite(drop_priority[drop_order])]
        drop_positions = drop_order[:config.drop_pool]
        if not drop_positions.numel():
            break
        drops = keep[drop_positions]
        if not donors.numel() or not drops.numel():
            break

        # Coverage of all visual patches by the candidate set; this is
        # a SAFEGUARD, not a claim of question-conditioned submodular novelty.
        current_sims = sim[:, keep]          # [N,K]
        top1, argtop = current_sims.max(dim=1)
        if k > 1:
            masked = current_sims.clone()
            masked.scatter_(1, argtop[:, None], -1)
            top2 = masked.max(dim=1).values.clamp_min(0)
        else:
            top2 = torch.zeros_like(top1)
        without = torch.where(
            argtop.unsqueeze(0) == drop_positions.unsqueeze(1),
            top2.unsqueeze(0), top1.unsqueeze(0)
        )                                  # [drop,N]
        donor_sim = sim[:, donors].T        # [donor,N]
        current_cov = top1.mean()
        new_cov = torch.maximum(
            without[:, None, :], donor_sim[None, :, :]
        ).mean(dim=-1)                     # [drop,donor]
        delta_cov = new_cov - current_cov
        query_gain = q[donors][None, :] - q[drops][:, None]
        donor_novelty = 1 - sim[donors][:, keep].amax(dim=1)
        ok = (
            (query_gain >= config.min_query_gain) &
            (donor_novelty[None, :] >= config.min_novelty) &
            (delta_cov >= -config.max_coverage_loss)
        )
        objective = (
            query_gain * donor_novelty[None, :] +
            config.coverage_weight * delta_cov
        ).masked_fill(~ok, float("-inf"))
        index = int(objective.flatten().argmax())
        if not torch.isfinite(objective.flatten()[index]):
            break
        nv = donors.numel()
        di, dj = divmod(index, nv)
        u, v = int(drops[di]), int(donors[dj])
        if records is not None:
            records.append({
                "removed_patch": u, "added_patch": v,
                "query_gain": float(query_gain[di, dj]),
                "coverage_delta": float(delta_cov[di, dj]),
                "donor_novelty": float(donor_novelty[dj]),
            })
        keep = torch.cat([keep[keep != u], keep.new_tensor([v])]).sort().values
        swaps += 1
    if keep.numel() != k or keep.unique().numel() != k:
        raise AssertionError("fixed-K invariant violated")
    return AnchorResult(
        keep, native, swaps,
        "swapped" if swaps else "no_safe_query_anchor",
        records
    )


def matched_random_entry_control(
    native_keep_ids: torch.Tensor,
    candidate_keep_ids: torch.Tensor,
    n_patches: int,
    seed: int = 17,
) -> torch.Tensor:
    """Match EXACT removed IDs and net swap count; randomize only additions."""
    base = _check_ids(native_keep_ids, n_patches, "native_keep_ids")
    cand = _check_ids(candidate_keep_ids, n_patches, "candidate_keep_ids")
    if cand.numel() != base.numel():
        raise ValueError("matched random control requires identical K")
    base_set, cand_set = set(base.tolist()), set(cand.tolist())
    removed = sorted(base_set - cand_set)
    pool = sorted(set(range(n_patches)) - base_set)
    if len(removed) > len(pool):
        raise ValueError("insufficient replacement pool")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(len(pool), generator=gen)[:len(removed)].tolist()
    added = [pool[i] for i in perm]
    out = sorted((base_set - set(removed)) | set(added))
    return base.new_tensor(out)
