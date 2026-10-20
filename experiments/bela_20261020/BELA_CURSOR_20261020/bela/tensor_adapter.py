"""Tensor-level prefill adapter for *explicit* visual positions.

Uses positions in the CURRENT sequence, maintains original text order, and
returns the complete kept sequence positions for repository-managed KV/masks.
It cannot know model-specific RoPE/cache conventions: callers must handle them.
"""
from __future__ import annotations
from typing import Sequence

def compose_sequence_keep(seq_len:int,visual_positions:Sequence[int],
                          visual_keep_slots:Sequence[int])->list[int]:
    vp=list(map(int,visual_positions))
    ks=list(map(int,visual_keep_slots))
    if not (0<=len(vp)<=seq_len) or len(set(vp))!=len(vp) or any(v<0 or v>=seq_len for v in vp):
        raise ValueError("invalid visual positions")
    if vp!=sorted(vp): raise ValueError("visual positions must be sorted in sequence order")
    if len(set(ks))!=len(ks) or any(k<0 or k>=len(vp) for k in ks):
        raise ValueError("invalid visual slots")
    selected={vp[i] for i in ks}
    visual_set=set(vp)
    return [i for i in range(seq_len) if i not in visual_set or i in selected]

def gather_prefill_tensor(hidden,visual_positions,visual_keep_slots,sequence_dim=-2):
    """Index-select along sequence axis, returns (new_hidden, kept_original_seq_positions).

    Sequence order remains ascending (even if selector score ranking differs).
    If batch items have different token lengths, model must use per-sample
    execution or explicit padding/mask logic; don't use this blindly for B>1.
    """
    import torch
    if not isinstance(hidden,torch.Tensor) or hidden.ndim<2:
        raise ValueError("expected torch tensor")
    dim=sequence_dim%hidden.ndim
    if dim==hidden.ndim-1: raise ValueError("cannot treat hidden feature dimension as sequence")
    ids=compose_sequence_keep(hidden.shape[dim],visual_positions,visual_keep_slots)
    idx=torch.tensor(ids,device=hidden.device,dtype=torch.long)
    return hidden.index_select(dim,idx),ids
