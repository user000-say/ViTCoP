"""Model-agnostic Stage1 QA-COVER bridge for ALREADY ALIGNED post-merge tokens.

This is a CPU/GPU testable *selector adapter*, not a Hugging Face decoder patch.
It never edits model position IDs, caches, image placeholders or DeepStack injection.
Those MUST be implemented and regression-tested at the real model callsite.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import torch.nn.functional as F
from qa_cover_original.experiments.bela_query_anchor.query_anchor import (
    AnchorConfig, select_question_anchored
)
from tools.budget_planner import allocate_group_quotas


@dataclass
class EntryBridgeOutput:
    keep_ids: torch.Tensor
    native_keep_ids: torch.Tensor
    swaps: int
    reason: str
    # Data to use only after the real model's position/DeepStack integration is fixed.
    selected_mrope_ids: Optional[torch.Tensor] = None
    selected_deepstack: Optional[list[torch.Tensor]] = None


def _require_unique_ids(ids: torch.Tensor, n: int, name: str):
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be integer [K]")
    if ids.numel() and (ids.min() < 0 or ids.max() >= n):
        raise ValueError(f"{name} not in 0..N-1")
    if ids.unique().numel() != ids.numel():
        raise ValueError(f"{name} contains duplicate IDs")


def _native_noop(native, name):
    return EntryBridgeOutput(native.clone(), native.clone(), 0, name)


@torch.no_grad()
def choose_postmerge_qa(
    *,
    model_name: str,
    native_keep_ids: torch.Tensor,
    dual_metric: torch.Tensor | Sequence[torch.Tensor],
    projected_full_vision: Optional[torch.Tensor],
    question_content_embeds: Optional[torch.Tensor],
    group_ids: Optional[torch.Tensor] = None,
    mrope_ids: Optional[torch.Tensor] = None,
    deepstack_features: Optional[list[torch.Tensor]] = None,
    max_swaps_total: int = 8,
    max_group_n: int = 2048,
    enabled: bool = True,
) -> EntryBridgeOutput:
    """Identities only; returns ORIGINAL post-merger visual IDs, sorted.

    `dual_metric` views MUST ALREADY be aligned to this post-merger order.
    `projected_full_vision` MUST ALREADY be in the language input feature space.
    `question_content_embeds` MUST exclude assistant output, image placeholders,
    and answer-bearing metadata; alignment is a HYPOTHESIS, not guaranteed.
    """
    if isinstance(dual_metric,torch.Tensor):
        views=[dual_metric]
    else:
        views=list(dual_metric)
    if not views:
        raise ValueError("missing dual metric")
    n=int(views[0].shape[0])
    if n == 0:
        raise ValueError("empty image")
    dev=views[0].device
    for view in views:
        if view.ndim != 2 or view.shape[0] != n or view.device != dev:
            raise ValueError("all visual views must be [N,D] on same device")
        if not torch.isfinite(view).all():
            raise ValueError("non-finite dual metric")
    keep=native_keep_ids.long().to(dev).sort().values
    _require_unique_ids(keep,n,"native_keep_ids")
    if mrope_ids is not None:
        if mrope_ids.ndim != 2 or mrope_ids.shape != (n,3):
            raise ValueError("mrope_ids must be [N,3] original coords, in original patch order")
    if deepstack_features is not None:
        for ds in deepstack_features:
            if ds.ndim != 2 or ds.shape[0] != n:
                raise ValueError("DeepStack views must have [N,D] and original shared ID order")
    if group_ids is None:
        group_ids=torch.zeros(n,dtype=torch.long,device=dev)
    else:
        if group_ids.shape != (n,) or group_ids.dtype not in (torch.int32,torch.int64):
            raise ValueError("group IDs must be integer [N]")
        group_ids=group_ids.to(dev)
        if group_ids.min()<0:
            raise ValueError("negative group ID")
    if model_name.startswith("llava-next") is False and group_ids.unique().numel()>1:
        raise ValueError("group-wise crop selection only implemented for LLaVA-NeXT")
    if not enabled or max_swaps_total==0:
        out=_native_noop(keep,"off")
    elif projected_full_vision is None or question_content_embeds is None:
        out=_native_noop(keep,"missing_query_signal")
    else:
        proj=projected_full_vision.detach()
        txt=question_content_embeds.detach()
        if (proj.ndim!=2 or proj.shape[0]!=n or txt.ndim!=2
                or proj.shape[-1]!=txt.shape[-1]):
            raise ValueError("projected visual [N,H] and question text [T,H] misaligned")
        if not torch.isfinite(proj).all() or not torch.isfinite(txt).all():
            raise ValueError("non-finite projected embeddings")
        if txt.shape[0]==0:
            out=_native_noop(keep,"empty_question")
        else:
            q=F.normalize(txt.float().mean(0,keepdim=True).to(dev),dim=-1)
            raw=(F.normalize(proj.float().to(dev),dim=-1) @ q.T).squeeze(-1)
            unique_g=sorted(int(x) for x in group_ids.unique().tolist())
            groups=[torch.nonzero(group_ids==g,as_tuple=False).flatten() for g in unique_g]
            if any(len(g)>max_group_n for g in groups):
                raise ValueError("group too large for dense QA-COVER similarity; use bounded exact tiled implementation")
            native_per_group=[int(torch.isin(keep,g).sum().item()) for g in groups]
            quotas=allocate_group_quotas(min(max_swaps_total,len(keep)),native_per_group)
            selected=[]
            total_swaps=0
            for indices,quota in zip(groups,quotas):
                in_grp=torch.isin(keep,indices)
                current=keep[in_grp]
                # Group positions need not be consecutive; map original -> local IDs
                reverse=torch.full((n,),-1,device=dev,dtype=torch.long)
                reverse[indices]=torch.arange(len(indices),device=dev)
                local_ids=reverse[current]
                if local_ids.numel()==0:
                    continue
                features=[v.index_select(0,indices) for v in views]
                local=select_question_anchored(
                    native_keep_ids=local_ids,
                    visual_metric=features,
                    raw_query_scores=raw.index_select(0,indices),
                    config=AnchorConfig(max_swaps=quota),
                    record=False,
                )
                chosen=indices.index_select(0,local.keep_ids)
                selected.append(chosen)
                total_swaps+=local.swaps
            final=torch.cat(selected).sort().values if selected else keep.new_empty(0)
            _require_unique_ids(final,n,"output keep")
            if final.numel()!=keep.numel():
                raise RuntimeError("group-wise QA-COVER changed total K")
            out=EntryBridgeOutput(final,keep,total_swaps,"selected")
    if mrope_ids is not None:
        out.selected_mrope_ids=mrope_ids.to(dev).index_select(0,out.keep_ids)
    if deepstack_features is not None:
        out.selected_deepstack=[v.to(dev).index_select(0,out.keep_ids) for v in deepstack_features]
    return out
