"""Stage2 Q_ATTN physical prune for TF 5.20 Llama inside LlavaNext."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class Stage2State:
    patch_pos: torch.Tensor  # absolute indices of surviving visual patches
    k_targets: Tuple[int, int]
    cut_layers: Tuple[int, int]  # 1-based "after this layer"
    events: list = field(default_factory=list)


def q_attn_scores_from_qk(
    q: torch.Tensor, k: torch.Tensor, patch_pos: torch.Tensor, text_start: int
) -> torch.Tensor:
    """q,k: [B,H,S,Dh] after RoPE. Mean text→patch attention over heads/queries."""
    assert q.shape[0] == 1
    qq, kk = q[0].float(), k[0].float()
    scale = kk.shape[-1] ** -0.5
    logits = torch.matmul(qq, kk.transpose(-1, -2)) * scale
    s = logits.shape[-1]
    causal = torch.ones(s, s, device=logits.device, dtype=torch.bool).triu(1)
    logits = logits.masked_fill(causal.unsqueeze(0), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if text_start >= s:
        return torch.ones(patch_pos.numel(), device=logits.device)
    text = probs[:, text_start:, :]
    cols = patch_pos.to(device=probs.device, dtype=torch.long)
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
            # keys/values: [B, H, S, Dh] — select along S
            if layer.keys.shape[-2] != keep_abs.numel() and layer.keys.shape[-2] != int(
                # allow when keep_abs indexes into current S
                layer.keys.shape[-2]
            ):
                pass
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
    scores = q_attn_scores_from_qk(q, k, patch_pos, text_start=int(text_start))
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
    non_patch = torch.where(~is_patch)[0]
    keep_abs = torch.cat([non_patch, keep_patch], dim=0).sort().values

    hs = gather_seq(hidden_states, keep_abs, dim=1)
    # Always index_select current position_ids by keep_abs.
    # Cut1: position_ids are 0..S-1 so this equals keep_abs (absolute).
    # Cut2+: position_ids already hold absolute RoPE ids; must NOT replace with
    # local keep_abs (that desyncs RoPE vs cached K/V).
    if position_ids is not None:
        position_ids = position_ids.index_select(
            -1, keep_abs.to(device=position_ids.device)
        )
    past_key_values = prune_cache(past_key_values, keep_abs, layers_filled)

    old_to_new = torch.full((seq,), -1, dtype=torch.long, device=device)
    old_to_new[keep_abs] = torch.arange(keep_abs.numel(), device=device)
    new_patch = old_to_new[keep_patch]
    state.patch_pos = new_patch.detach().cpu()
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
