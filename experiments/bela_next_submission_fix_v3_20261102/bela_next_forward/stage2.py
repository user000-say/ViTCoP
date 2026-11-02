"""BELA NeXT7 Q_ATTN: score input-Q/K, prune layer output + already-filled KV.

Index namespace: `patch_pos` are positions in *current physical* decoder
sequence, while `patch_uids` preserve identities in the native AnyRes stream.
RoPE position_ids refer to original Stage1 positions and must never be reset.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch
import os


@dataclass
class Stage2State:
    patch_pos: torch.Tensor
    k_targets: Tuple[int, int]
    cut_layers: Tuple[int, int]
    patch_uids: Optional[torch.Tensor] = None
    events: list = field(default_factory=list)
    last_keep_abs: Optional[torch.Tensor] = None


def q_attn_scores_from_qk(q: torch.Tensor, k: torch.Tensor, patch_pos: torch.Tensor,
                          text_start: int, valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Correct causal text->visual score (head-avg), respecting GQA expanded K."""
    if q.ndim != 4 or k.ndim != 4 or q.shape[:3] != k.shape[:3] or q.shape[-1] != k.shape[-1] or q.shape[0] != 1:
        raise ValueError('q and expanded k must be [1,H,S,D] and identical shapes')
    seq = q.shape[-2]
    patch_pos = patch_pos.to(device=q.device, dtype=torch.long)
    if patch_pos.numel() == 0 or bool((patch_pos < 0).any()) or bool((patch_pos >= seq).any()) or patch_pos.unique().numel()!=patch_pos.numel():
        raise ValueError('invalid Stage2 physical patch positions')
    if not (0 <= text_start < seq):
        raise RuntimeError('Q_ATTN_NO_TEXT: no text queries after image')
    # Physical sequence coordinates only. RoPE coordinates may be sparse after
    # pruning; causality is the physical order of keys actually stored in cache.
    # Exclude padded query rows and padded key columns, matching Dense attention.
    if valid_token_mask is None:
        valid = torch.ones(seq, dtype=torch.bool, device=q.device)
    else:
        if valid_token_mask.shape != (1, seq):
            raise RuntimeError('Q_ATTN_MASK_SHAPE: expected [1,physical_seq_len]')
        valid = valid_token_mask[0].to(device=q.device).bool()
    text_rows = torch.where(valid & (torch.arange(seq, device=q.device) >= text_start))[0]
    if text_rows.numel() == 0:
        raise RuntimeError('Q_ATTN_NO_VALID_TEXT: all suffix query positions padded')
    if not bool(valid.index_select(0, patch_pos).all()):
        raise RuntimeError('Q_ATTN_PADDED_PATCH: selected patch cannot be padding')
    qq = q[0].index_select(-2, text_rows).float()
    kk = k[0].float()
    logits = torch.matmul(qq, kk.transpose(-1,-2)) * (kk.shape[-1] ** -0.5)
    causal = (torch.arange(seq, device=q.device).view(1, 1, seq)
              <= text_rows.view(1, -1, 1))
    permitted = causal & valid.view(1,1,-1)
    logits = logits.masked_fill(~permitted, -torch.inf)
    probs = torch.softmax(logits, dim=-1)
    scores = probs.index_select(-1, patch_pos).mean(dim=(0,1))
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError('Q_ATTN_NAN_INF: corrupt input Q/K or attention mask')
    return scores


def _layer_qk(decoder_layer, hidden_states, position_embeddings):
    """Replicate TF5.20.2 LlamaAttention input LN, projection, RoPE and GQA."""
    attn = decoder_layer.self_attn
    cfg = attn.config
    h = decoder_layer.input_layernorm(hidden_states)
    b, s, _ = h.shape
    hdim = attn.head_dim
    heads = int(cfg.num_attention_heads)
    kvheads = int(cfg.num_key_value_heads)
    if heads % kvheads:
        raise RuntimeError('invalid GQA head ratio')
    q = attn.q_proj(h).view(b, s, heads, hdim).transpose(1, 2)
    k = attn.k_proj(h).view(b, s, kvheads, hdim).transpose(1, 2)
    if kvheads != heads:
        k = k.repeat_interleave(heads//kvheads, dim=1)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    return q, k


def _checked_keep(seq: int, patch_pos: torch.Tensor, keep_patch: torch.Tensor, device):
    if seq <= 0 or patch_pos.ndim != 1 or keep_patch.ndim != 1:
        raise RuntimeError('invalid keep/index rank')
    if bool((patch_pos < 0).any()) or bool((patch_pos >= seq).any()) or patch_pos.unique().numel()!=patch_pos.numel():
        raise RuntimeError('Stage2 stale/out-of-range patch coordinate')
    if not bool(torch.isin(keep_patch, patch_pos).all()) or keep_patch.unique().numel()!=keep_patch.numel():
        raise RuntimeError('Stage2 non-nested or duplicated patch keep')
    is_patch = torch.zeros(seq, dtype=torch.bool, device=device)
    is_patch[patch_pos] = True
    is_patch[keep_patch] = False
    keep = torch.where(~is_patch)[0]
    if keep.numel() != seq - (patch_pos.numel() - keep_patch.numel()):
        raise RuntimeError('physical prune length mismatch')
    return keep


def prune_cache(past_key_values, keep_abs: torch.Tensor, layers_filled: int, expected_seq: int):
    """Require all filled cache layers to have exactly the pre-prune length."""
    if past_key_values is None:
        raise RuntimeError('CACHE_REQUIRED: need real DynamicCache for generation')
    from transformers.cache_utils import DynamicCache
    if not isinstance(past_key_values, DynamicCache):
        raise RuntimeError(f'CACHE_API_UNSUPPORTED: {type(past_key_values)}; only DynamicCache verified')
    if len(past_key_values.layers) < layers_filled:
        raise RuntimeError('cache contains fewer layers than executed decoder blocks')
    idx = keep_abs.long()
    if idx.ndim != 1 or idx.unique().numel() != idx.numel() or (idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= expected_seq)):
        raise RuntimeError('cache gather out-of-bounds or duplicate indices')
    for i in range(layers_filled):
        layer = past_key_values.layers[i]
        if not getattr(layer, 'is_initialized', False):
            raise RuntimeError(f'CACHE_MISSING: layer {i} was not initialized')
        kk, vv = layer.keys, layer.values
        if kk.ndim != 4 or vv.ndim != 4 or kk.shape[-2] != expected_seq or vv.shape[-2] != expected_seq:
            raise RuntimeError(f'CACHE_LENGTH: layer {i} K/V must both have exactly {expected_seq} positions')
        layer.keys = kk.index_select(-2, idx.to(kk.device))
        layer.values = vv.index_select(-2, idx.to(vv.device))
    return past_key_values


@torch.no_grad()
def select_and_prune(*, state: Stage2State, hidden_states: torch.Tensor,
                     position_ids: Optional[torch.Tensor], past_key_values, decoder_layer,
                     position_embeddings, layer_1based: int, k_target: int,
                     layers_filled: int, text_start: int,
                     score_hidden_states: Optional[torch.Tensor] = None,
                     q: Optional[torch.Tensor] = None, k: Optional[torch.Tensor] = None,
                     valid_token_mask: Optional[torch.Tensor] = None):
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise RuntimeError('BELA Stage2 only supports batch1')
    dev = hidden_states.device
    patch_pos = state.patch_pos.to(device=dev, dtype=torch.long)
    seq = hidden_states.shape[1]
    if k_target <= 0 or k_target > patch_pos.numel():
        raise RuntimeError('invalid Stage2 k_target')
    if (q is None) != (k is None):
        raise RuntimeError('must supply both Q and K from same layer input')
    if q is None:
        if score_hidden_states is None:
            raise RuntimeError('INPUT_QK_REQUIRED: post-layer Q/K fallback forbidden')
        q, k = _layer_qk(decoder_layer, score_hidden_states, position_embeddings)
    scores = q_attn_scores_from_qk(q, k, patch_pos, text_start=int(text_start), valid_token_mask=valid_token_mask)
    order = torch.argsort(scores, descending=True, stable=True)[:k_target]
    keep_patch = patch_pos.index_select(0, order).sort().values
    keep_abs = _checked_keep(seq, patch_pos, keep_patch, dev)
    if position_ids is None or position_ids.shape != (1,seq):
        raise RuntimeError('POSITION_IDS_REQUIRED: preserve original Stage1 RoPE coordinates')
    kept_positions = position_ids.index_select(-1, keep_abs.to(position_ids.device))
    if kept_positions.numel() > 1 and not bool((kept_positions[:,1:] > kept_positions[:,:-1]).all()):
        raise RuntimeError('RoPE original position order not strictly increasing after prune')
    cache = prune_cache(past_key_values, keep_abs, layers_filled, seq)
    new_h = hidden_states.index_select(1, keep_abs)
    mapping = torch.full((seq,), -1, dtype=torch.long, device=dev)
    mapping[keep_abs] = torch.arange(keep_abs.numel(), device=dev)
    new_patch = mapping[keep_patch]
    if bool((new_patch < 0).any()):
        raise RuntimeError('Stage2 patch identity was lost during gather')
    if state.patch_uids is not None:
        if state.patch_uids.numel() != patch_pos.numel():
            raise RuntimeError('Stage2 patch UID/position length drift')
        old_ids = state.patch_uids.to(device=dev, dtype=torch.long)
        retained = torch.isin(patch_pos, keep_patch)
        new_uids = old_ids[retained]
        if new_uids.unique().numel() != k_target:
            raise RuntimeError('Stage2 UID duplicated or resurrected')
        state.patch_uids = new_uids.detach().cpu()
    state.patch_pos = new_patch.detach().cpu()
    state.last_keep_abs = keep_abs.detach().cpu()
    event = dict(layer_1based=int(layer_1based),
                 k_before=int(patch_pos.numel()), k_after=k_target,
                 seq_before=seq, seq_after=int(new_h.shape[1]),
                 unique_patch=int(new_patch.unique().numel()),
                 kept_uid_sample=[] if state.patch_uids is None else state.patch_uids[:16].tolist(),
                 kept_rope_max=int(kept_positions.max()))
    # Full UID and keep-position proof on a FEW selected smoke examples only.
    if os.environ.get('BELA_TRACE_FULL_UID','0') == '1':
        event['full_keep_abs'] = keep_abs.detach().cpu().tolist()
        event['full_patch_uids'] = [] if state.patch_uids is None else state.patch_uids.tolist()
        event['full_position_ids'] = kept_positions.detach().cpu()[0].tolist()
    state.events.append(event)
    return new_h, kept_positions, cache, state
