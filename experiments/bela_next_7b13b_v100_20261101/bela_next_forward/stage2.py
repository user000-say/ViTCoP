"""Stage2 Q_ATTN physical prune for TF 5.20 Llama inside LlavaNext."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class Stage2State:
    patch_pos: torch.Tensor  # local indices of surviving visual patches in current seq
    k_targets: Tuple[int, int]
    cut_layers: Tuple[int, int]  # 1-based "after this layer"
    # Per-position kind aligned with current hidden length: -2=text, -1=newline, 0=patch
    source: Optional[torch.Tensor] = None
    events: list = field(default_factory=list)


def q_attn_scores_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    patch_pos: torch.Tensor,
    text_start: int,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """q,k: [B,H,S,Dh] after RoPE. Mean text→patch attention over heads/queries.

    Prefer suffix-text queries (abs RoPE pos > max patch abs pos) when position_ids
    carry Dense-absolute positions — matches fidelity kit Q_ATTN policy.
    """
    assert q.shape[0] == 1
    qq, kk = q[0].float(), k[0].float()
    scale = kk.shape[-1] ** -0.5
    logits = torch.matmul(qq, kk.transpose(-1, -2)) * scale
    s = logits.shape[-1]
    causal = torch.ones(s, s, device=logits.device, dtype=torch.bool).triu(1)
    logits = logits.masked_fill(causal.unsqueeze(0), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    cols = patch_pos.to(device=probs.device, dtype=torch.long)

    text_idx = None
    if position_ids is not None and patch_pos.numel() > 0:
        patch_abs = position_ids[0].index_select(0, cols)
        text_idx = torch.where(position_ids[0] > patch_abs.max())[0]
    if text_idx is None or text_idx.numel() == 0:
        if text_start >= s:
            return torch.ones(patch_pos.numel(), device=logits.device)
        text = probs[:, int(text_start) :, :]
        patch_p = text.index_select(-1, cols)
    else:
        text = probs.index_select(1, text_idx)
        patch_p = text.index_select(-1, cols)
    return patch_p.mean(dim=(0, 1))


def _layer_qk(decoder_layer, hs, position_embeddings):
    attn = decoder_layer.self_attn
    cfg = attn.config
    normed = decoder_layer.input_layernorm(hs)
    bsz, q_len, _ = normed.shape
    head_dim = attn.head_dim
    num_heads = int(cfg.num_attention_heads)
    num_kv = int(cfg.num_key_value_heads)
    q = attn.q_proj(normed).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    k = attn.k_proj(normed).view(bsz, q_len, num_kv, head_dim).transpose(1, 2)
    if num_kv != num_heads:
        k = k.repeat_interleave(num_heads // num_kv, dim=1)
    cos, sin = position_embeddings
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k


def gather_seq(x: torch.Tensor, keep_abs: torch.Tensor, dim: int) -> torch.Tensor:
    return x.index_select(dim, keep_abs.to(device=x.device))


def prune_cache(past_key_values, keep_abs: torch.Tensor, layers_filled: int):
    """Physically index-select seq dim on already-filled TF5.20 DynamicCache layers."""
    if past_key_values is None or layers_filled <= 0:
        return past_key_values
    from transformers.cache_utils import DynamicCache

    if isinstance(past_key_values, DynamicCache):
        for i in range(min(layers_filled, len(past_key_values.layers))):
            layer = past_key_values.layers[i]
            if not getattr(layer, "is_initialized", False):
                continue
            if layer.get_seq_length() == 0:
                continue
            idx = keep_abs.to(device=layer.keys.device)
            if int(keep_abs.max()) >= int(layer.keys.shape[-2]):
                raise RuntimeError(
                    f"prune_cache keep max {int(keep_abs.max())} >= cache S {layer.keys.shape[-2]}"
                )
            layer.keys = layer.keys.index_select(-2, idx)
            layer.values = layer.values.index_select(-2, idx)
        return past_key_values
    out = []
    for i, layer in enumerate(past_key_values):
        if i >= layers_filled:
            out.append(layer)
            continue
        kk, vv = layer[0], layer[1]
        idx = keep_abs.to(device=kk.device)
        out.append((kk.index_select(2, idx), vv.index_select(2, idx)))
    return tuple(out)


def _drop_orphan_newlines(keep: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Drop image_newline markers whose preceding HR row lost all patches."""
    keep = keep.clone()
    nl = torch.where(source == -1)[0]
    if nl.numel() == 0:
        return keep
    highres = bool((source == 0).any())  # patches present
    prev = 0
    for newline in nl.tolist():
        row = torch.arange(prev, newline, device=source.device)
        has_patch = bool((keep[row] & (source[row] == 0)).any()) if row.numel() else False
        if not has_patch and highres:
            keep[newline] = False
        prev = newline + 1
    return keep


@torch.no_grad()
def select_and_prune(
    *,
    state: Stage2State,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    past_key_values,
    decoder_layer,
    position_embeddings,
    layer_1based: int,
    k_target: int,
    layers_filled: int,
    text_start: int,
):
    device = hidden_states.device
    patch_pos = state.patch_pos.to(device=device, dtype=torch.long)
    q, k = _layer_qk(decoder_layer, hidden_states, position_embeddings)
    scores = q_attn_scores_from_qk(
        q, k, patch_pos, text_start=int(text_start), position_ids=position_ids
    )
    if scores.numel() != patch_pos.numel():
        raise RuntimeError("score/patch size mismatch")
    if k_target > patch_pos.numel():
        raise RuntimeError(f"k_target {k_target} > alive patches {patch_pos.numel()}")
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        raise RuntimeError("NaN/Inf in Stage2 scores")
    top_local = scores.topk(k_target, largest=True).indices
    keep_patch = patch_pos.index_select(0, top_local).sort().values
    seq = hidden_states.shape[1]
    is_patch = torch.zeros(seq, dtype=torch.bool, device=device)
    is_patch[patch_pos] = True
    keep = ~is_patch
    keep[keep_patch] = True
    if state.source is not None and state.source.numel() == seq:
        src = state.source.to(device=device)
        keep = _drop_orphan_newlines(keep, src)
    keep_abs = torch.where(keep)[0]

    hs = gather_seq(hidden_states, keep_abs, dim=1)
    # Always index_select current position_ids by keep_abs so Dense-absolute
    # RoPE ids survive both cuts (never replace with local arange).
    if position_ids is not None:
        position_ids = position_ids.index_select(
            -1, keep_abs.to(device=position_ids.device)
        )
    past_key_values = prune_cache(past_key_values, keep_abs, layers_filled)

    old_to_new = torch.full((seq,), -1, dtype=torch.long, device=device)
    old_to_new[keep_abs] = torch.arange(keep_abs.numel(), device=device)
    new_patch = old_to_new[keep_patch]
    state.patch_pos = new_patch.detach().cpu()
    if state.source is not None and state.source.numel() == seq:
        state.source = state.source.to(device=device)[keep_abs].detach().cpu()
    state.events.append(
        {
            "layer_1based": layer_1based,
            "k_before": int(patch_pos.numel()),
            "k_after": int(k_target),
            "seq_before": int(seq),
            "seq_after": int(hs.shape[1]),
            "unique_patch": int(new_patch.unique().numel()),
        }
    )
    return hs, position_ids, past_key_values, state
