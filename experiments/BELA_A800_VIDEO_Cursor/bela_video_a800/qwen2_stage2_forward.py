"""Monkey-patch Qwen2Model.forward for BELA video Stage2 physical cuts.

Install via apply_patch(). Safe no-op when BELA_VIDEO_STAGE2=OFF.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from .live_stage2 import Stage2State, cut_plan, select_and_prune, stage2_active


_ORIG = {}


def _prepare_mask(self, attention_mask, inputs_embeds, batch_size, seq_length, past_key_values_length, output_attentions):
    if self._attn_implementation == "flash_attention_2":
        return attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    if self._attn_implementation == "sdpa" and not output_attentions:
        return _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
    return _prepare_4d_causal_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )


def bela_qwen2_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    # Decode / inactive → original forward
    is_prefill = (inputs_embeds is not None and inputs_embeds.shape[1] > 1) or (
        input_ids is not None and input_ids.shape[1] > 1
    )
    ctx = getattr(self, "_bela_stage2_ctx", None)
    if (not stage2_active()) or (not is_prefill) or ctx is None:
        return _ORIG["forward"](
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Cannot specify both input_ids and inputs_embeds")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    if batch_size != 1:
        raise RuntimeError("BELA video Stage2 supports batch=1 only")

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        ).unsqueeze(0)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    attention_mask = _prepare_mask(
        self, attention_mask, inputs_embeds, batch_size, seq_length, past_key_values_length, False
    )

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    from .live_stage2 import (
        q_attn_scores_from_hidden,
        q_attn_scores_from_qk,
        ranks_from_scores,
        stage2_mode,
    )

    tier = ctx["tier"]
    plan = {layer: (k, sel) for layer, k, sel in cut_plan(tier)}
    state = Stage2State(
        frames=int(ctx["frames"]),
        entry_k=int(ctx["keep_per_frame"]),
        patch_ids=torch.tensor(ctx["ids"], dtype=torch.long),
        visual_start=int(ctx["visual_start"]),
    )
    observed = []
    first_cut = min(plan.keys())
    probe_layer = first_cut - 1 if stage2_mode() == "RDRIFT_FIRST" else -1
    # P3: same-layer Q/K (default on). Set BELA_VIDEO_STAGE2_SCORE=hidden to revert.
    score_mode = os.environ.get("BELA_VIDEO_STAGE2_SCORE", "qk").strip().lower()
    if score_mode not in ("qk", "hidden"):
        score_mode = "qk"

    def _layer_qk(decoder_layer, hs, pos_ids):
        """Build RoPE'd Q/K for this layer from pre-attn hidden (batch=1)."""
        attn = decoder_layer.self_attn
        normed = decoder_layer.input_layernorm(hs)
        bsz, q_len, _ = normed.shape
        q = attn.q_proj(normed)
        k = attn.k_proj(normed)
        head_dim = attn.head_dim
        num_heads = attn.num_heads
        num_kv = attn.num_key_value_heads
        q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, num_kv, head_dim).transpose(1, 2)
        if num_kv != num_heads:
            k = k.repeat_interleave(num_heads // num_kv, dim=1)
        try:
            cos, sin = attn.rotary_emb(k, seq_len=q_len)
        except TypeError:
            cos, sin = attn.rotary_emb(k, position_ids=pos_ids)
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        try:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
        except TypeError:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)
        return q, k

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_1based = layer_idx + 1
        observed.append(state.frames * state.current_k())

        need_cut = layer_1based in plan
        cur_seq = hidden_states.shape[1]
        if attention_mask is None or (
            attention_mask.dim() == 4 and attention_mask.shape[-1] != cur_seq
        ):
            attention_mask = _prepare_mask(
                self, None, hidden_states, batch_size, cur_seq, past_key_values_length, False
            )

        # P3: capture Q/K *before* this layer's attention when a cut follows the layer.
        # Cut is applied after the layer forward (post-attn hidden), but scores use
        # this layer's Q/K built from pre-attn residual (STAR-compatible instrumentation).
        pre_qk = None
        if need_cut and score_mode == "qk" and state.current_k() > plan[layer_1based][0]:
            pre_qk = _layer_qk(decoder_layer, hidden_states, position_ids)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=use_cache,
        )
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[1]
            past_key_values = next_decoder_cache

        if layer_1based == probe_layer:
            k = state.current_k()
            f = state.frames
            vs, ve = state.visual_start, state.visual_start + f * (k + 1)
            if score_mode == "qk":
                q, kk = _layer_qk(decoder_layer, hidden_states, position_ids)
                flat = q_attn_scores_from_qk(q, kk, vs, ve).view(f, k + 1)[:, :k]
            else:
                flat = q_attn_scores_from_hidden(hidden_states, vs, ve).view(f, k + 1)[:, :k]
            state.ranks_history = ranks_from_scores(flat)

        if need_cut:
            k_tgt, selector = plan[layer_1based]
            if state.current_k() > k_tgt:
                if hidden_states.is_cuda:
                    torch.cuda.synchronize()
                print(
                    f"[BELA_S2] cut_enter layer={layer_1based} seq={tuple(hidden_states.shape)} "
                    f"K={state.current_k()} -> {k_tgt} sel={selector} vs={state.visual_start} score={score_mode}",
                    flush=True,
                )
                hidden_states, position_ids, attention_mask, past_key_values, state = select_and_prune(
                    state=state,
                    hidden_states=hidden_states,
                    attn_avg=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    layer_1based=layer_1based,
                    k_target=k_tgt,
                    selector=selector,
                    layers_filled=layer_idx + 1,
                    qk=pre_qk if score_mode == "qk" else None,
                )
                next_decoder_cache = past_key_values
                attention_mask = None
                print(f"[BELA_S2] cut_done layer={layer_1based} seq={tuple(hidden_states.shape)}", flush=True)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # publish audit
    self._bela_last_stage2 = {
        "mode": os.environ.get("BELA_VIDEO_STAGE2"),
        "tier": tier,
        "score_mode": score_mode,
        "events": state.events,
        "physical_layers": state.physical_layers,
        "observed_visual_k_patches_by_layer": observed,
        "final_patch_ids": state.patch_ids.tolist(),
        "final_seq_len": int(hidden_states.shape[1]),
    }
    # also mirror onto parent Llava if present
    parent = getattr(self, "_bela_parent_ref", None)
    if parent is not None:
        parent._bela_last_stage2 = self._bela_last_stage2

    # Prefill pruning is one-shot. Clear ctx so decode steps use the stock
    # Qwen2 forward. Stash pruned length so generate()'s stale attention_mask
    # (still full pre-prune length) can be rewritten in prepare_inputs_for_generation.
    prefill_len = int(hidden_states.shape[1])
    self._bela_stage2_ctx = None
    self._bela_prefill_seq_len = prefill_len
    if parent is not None:
        parent._bela_stage2_ctx = None
        parent._bela_pending_visual_start = None
        parent._bela_prefill_seq_len = prefill_len

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def apply_patch():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    if getattr(Qwen2Model, "_bela_stage2_patched", False):
        return False
    _ORIG["forward"] = Qwen2Model.forward
    Qwen2Model.forward = bela_qwen2_forward
    Qwen2Model._bela_stage2_patched = True
    return True
