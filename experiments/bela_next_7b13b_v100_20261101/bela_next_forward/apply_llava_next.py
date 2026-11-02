"""Wire native NeXT BELA (FIX KIT 20261102) into HF LlavaNext TF5.20.

BELA_NEXT_V2_ENABLE=0 → stock Dense forward/generate untouched.
ENABLE=1 → Stage1: stock spatial_unpad+newline + Dual COVER (CLIP hs[10]=L9+pen) +
global QA B8 + structural newline keep; Stage2 Q_ATTN physical cuts;
RoPE uses Dense-absolute positions (gather, never compact-renumber);
decode clock = dense_seq_len + step.

Do NOT force image_grid_pinpoints=[[672,672]] here — match Dense processor.
Do NOT prune inside get_image_features (no question there); Stage1 runs in
prefill forward with input_ids so QA sees the real question.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import torch

from .native_stage1 import run_native_stage1
from .stage2 import Stage2State, select_and_prune

_ARMS = None
_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
if str(_PKG / "BELA_NEXT_7B13B_V100_CURSOR_20261101/src") not in sys.path:
    sys.path.insert(0, str(_PKG / "BELA_NEXT_7B13B_V100_CURSOR_20261101/src"))


def _arms():
    global _ARMS
    if _ARMS is None:
        p = _PKG / "BELA_NEXT_7B13B_V100_CURSOR_20261101/configs/arms.json"
        _ARMS = {a["arm_id"]: a for a in json.loads(p.read_text())["arm_matrix"]}
    return _ARMS


def enabled() -> bool:
    return os.environ.get("BELA_NEXT_V2_ENABLE", "0").strip() in ("1", "true", "TRUE", "yes")


def current_arm():
    aid = os.environ.get("BELA_NEXT_ARM", "").strip()
    if not aid:
        raise RuntimeError("BELA_NEXT_ARM must be set when BELA_NEXT_V2_ENABLE=1")
    arms = _arms()
    if aid not in arms:
        raise RuntimeError(f"unknown BELA_NEXT_ARM={aid}")
    return arms[aid]


def _rewrite_image_placeholders(input_ids, attention_mask, image_token_id: int, n_keep: int):
    assert input_ids.shape[0] == 1
    ids = input_ids[0]
    pos = (ids == image_token_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        raise RuntimeError("PRUNING_NOT_ACTIVE: no image placeholders in input_ids")
    start, end = int(pos[0]), int(pos[-1]) + 1
    if int(pos[-1] - pos[0] + 1) != int(pos.numel()):
        raise RuntimeError("PRUNING_NOT_ACTIVE: image placeholders not contiguous")
    prefix = ids[:start]
    suffix = ids[end:]
    mid = ids.new_full((n_keep,), image_token_id)
    new_ids = torch.cat([prefix, mid, suffix], dim=0).unsqueeze(0)
    if attention_mask is None:
        new_mask = None
    else:
        am = attention_mask[0]
        new_mask = torch.cat([am[:start], am.new_ones(n_keep), am[end:]], dim=0).unsqueeze(0)
    return new_ids, new_mask, start


def _past_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            return 0
    return 0


def apply_patch(model) -> bool:
    if getattr(model, "_bela_next_v2_patched", False):
        return False

    from bela_next_compat import original_dual, original_qa

    llava = model.model
    lm = llava.language_model
    llama_model = lm.model if hasattr(lm, "model") else lm

    orig_llava_fwd = llava.forward
    orig_get_feats = llava.get_image_features

    def hooked_get_image_features(self_m, *args, **kwargs):
        # Never prune here — no question available. Stock Dense pack only.
        return orig_get_feats(*args, **kwargs)

    def _llama_forward_with_cuts(
        self_lm,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        state: Stage2State,
        parent,
    ):
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self_lm.config)

        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long
            ).unsqueeze(0)

        hidden_states = inputs_embeds
        text_start = int(getattr(parent, "_bela_text_start", int(state.patch_pos.max().item()) + 1))
        cut_map = {state.cut_layers[0]: state.k_targets[0], state.cut_layers[1]: state.k_targets[1]}

        for layer_idx, decoder_layer in enumerate(self_lm.layers[: self_lm.config.num_hidden_layers]):
            causal_mask = create_causal_mask(
                config=self_lm.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            position_embeddings = self_lm.rotary_emb(hidden_states, position_ids=position_ids)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            layer_1based = layer_idx + 1
            if layer_1based in cut_map:
                k_tgt = cut_map[layer_1based]
                if state.patch_pos.numel() > k_tgt:
                    hidden_states, position_ids, past_key_values, state = select_and_prune(
                        state=state,
                        hidden_states=hidden_states,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        decoder_layer=decoder_layer,
                        position_embeddings=position_embeddings,
                        layer_1based=layer_1based,
                        k_target=k_tgt,
                        layers_filled=layer_idx + 1,
                        text_start=text_start,
                    )
                    # Prefer suffix-text via absolute RoPE in q_attn; keep local fallback.
                    if position_ids is not None and state.patch_pos.numel() > 0:
                        pabs = position_ids[0].index_select(
                            0, state.patch_pos.to(device=position_ids.device)
                        )
                        suf = torch.where(position_ids[0] > pabs.max())[0]
                        text_start = (
                            int(suf[0].item())
                            if suf.numel()
                            else int(hidden_states.shape[1])
                        )
                    else:
                        text_start = int(state.patch_pos.max().item()) + 1
                    parent._bela_text_start = text_start
                    if attention_mask is not None:
                        attention_mask = torch.ones(
                            hidden_states.shape[0],
                            hidden_states.shape[1],
                            device=hidden_states.device,
                            dtype=attention_mask.dtype,
                        )

        hidden_states = self_lm.norm(hidden_states)
        parent._bela_position_ids = position_ids.detach()
        # Decode RoPE must continue the Dense multimodal clock, not max(kept)+1
        # on a compacted axis (fidelity kit / historical BELA).
        dense_len = int(getattr(parent, "_bela_dense_seq_len", -1))
        if dense_len > 0:
            parent._bela_rope_next_pos = dense_len
        else:
            parent._bela_rope_next_pos = int(position_ids.max().item()) + 1
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def hooked_llava_forward(
        self_m,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        use_cache=None,
        mm_encoder_outputs=None,
        **kwargs,
    ):
        if not enabled():
            # TF generate may leave both pixels and mm_encoder_outputs; stock forbids both.
            kw = dict(kwargs)
            mm = mm_encoder_outputs if mm_encoder_outputs is not None else kw.get("mm_encoder_outputs")
            has_mm = False
            if isinstance(mm, dict):
                has_mm = mm.get("image") is not None
            elif mm is not None:
                has_mm = True
            pv = pixel_values if pixel_values is not None else kw.get("pixel_values")
            if pv is not None and has_mm:
                pixel_values = None
                image_sizes = None
                kw.pop("pixel_values", None)
                kw.pop("image_sizes", None)
            return orig_llava_fwd(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                use_cache=use_cache,
                mm_encoder_outputs=mm_encoder_outputs,
                **kw,
            )

        past_len = _past_len(past_key_values)
        if past_len > 0:
            return orig_llava_fwd(
                input_ids=input_ids,
                pixel_values=None,
                image_sizes=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                mm_encoder_outputs=None,
                **kwargs,
            )

        arm = current_arm()
        if pixel_values is None or pixel_values.numel() == 0:
            raise RuntimeError(
                "PRUNING_NOT_ACTIVE: ENABLE=1 prefill requires pixel_values "
                "(QA needs joint vision+question; refuse mm_encoder-only pruned path)"
            )
        if input_ids is None:
            raise RuntimeError("BELA NeXT Stage1 requires input_ids for QA question span")

        s1 = run_native_stage1(
            self_m,
            pixel_values,
            image_sizes,
            input_ids,
            arm,
            dual_fn=original_dual,
            qa_fn=original_qa,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        native_packed = s1.native_packed
        native_keep = s1.native_keep.to(device=native_packed.device)
        native_is_patch = s1.native_is_patch.to(device=native_packed.device)
        n_native = int(native_packed.shape[0])
        model._bela_stage1 = {
            "arm": s1.arm_id,
            "k0": s1.k0,
            "k1": s1.k1,
            "k2": s1.k2,
            "packed_len": int(s1.packed_features.shape[0]),
            "packed_patches": s1.n_patches,
            "packed_newlines": s1.n_newlines,
            "native_patches": s1.native_patch_count,
            "native_newlines": s1.native_newlines,
            "native_len": n_native,
            "pack_mode": s1.pack_mode,
            "mid_layer": s1.mid_layer,
            "qa_log": s1.qa_log,
            "selector": "NATIVE_DUAL_COVER_L9_PLUS_QA_B8",
            "rope_mode": "dense_absolute",
        }
        model._bela_prefill_hits = int(getattr(model, "_bela_prefill_hits", 0)) + 1

        # 1) Build FULL Dense multimodal sequence (native AnyRes pack, unpruned).
        image_token_id = self_m.config.image_token_id
        dense_ids, dense_mask, img_start = _rewrite_image_placeholders(
            input_ids, attention_mask, image_token_id, n_native
        )
        dense_embeds = self_m.get_input_embeddings()(dense_ids)
        special = (dense_ids == image_token_id).unsqueeze(-1).expand_as(dense_embeds)
        dense_embeds = dense_embeds.masked_scatter(
            special, native_packed.to(dense_embeds.device, dense_embeds.dtype)
        )
        dense_len = int(dense_embeds.shape[1])
        dense_pos = torch.arange(dense_len, device=dense_embeds.device, dtype=torch.long)

        # 2) Stage1 gather: keep text + Dual/QA-selected patches/newlines; preserve
        #    Dense-absolute RoPE ids (fidelity kit contract).
        keep_mask = torch.ones(dense_len, dtype=torch.bool, device=dense_embeds.device)
        keep_mask[img_start : img_start + n_native] = native_keep
        seq_indices = torch.where(keep_mask)[0]
        inputs_embeds = dense_embeds.index_select(1, seq_indices)
        position_ids = dense_pos.index_select(0, seq_indices).unsqueeze(0)
        if dense_mask is not None:
            new_mask = dense_mask.index_select(1, seq_indices)
        else:
            new_mask = None

        # source kinds on Stage1-compressed seq: -2 text, -1 newline, 0 patch
        source = torch.full((seq_indices.numel(),), -2, dtype=torch.long, device=dense_embeds.device)
        for new_i, abs_i in enumerate(seq_indices.tolist()):
            if img_start <= abs_i < img_start + n_native:
                li = abs_i - img_start
                source[new_i] = 0 if bool(native_is_patch[li]) else -1
        patch_pos = torch.where(source == 0)[0]
        if int(patch_pos.numel()) != int(s1.k0):
            raise RuntimeError(
                f"Stage1 gather patches {patch_pos.numel()} != K0 {s1.k0}"
            )
        # local text_start fallback = first suffix text after image span
        text_start = int((seq_indices < img_start + n_native).sum().item())
        model._bela_patch_pos = patch_pos.detach().cpu()
        model._bela_text_start = text_start
        model._bela_dense_seq_len = dense_len
        model._bela_stage1["img_start"] = int(img_start)
        model._bela_stage1["dense_seq_len"] = dense_len
        model._bela_stage1["seq_after_stage1"] = int(inputs_embeds.shape[1])

        state = Stage2State(
            patch_pos=patch_pos.clone(),
            k_targets=(s1.k1, s1.k2),
            cut_layers=s1.cut_layers,
            source=source.detach().cpu(),
        )
        outputs = _llama_forward_with_cuts(
            llama_model,
            inputs_embeds=inputs_embeds,
            attention_mask=new_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True if use_cache is None else use_cache,
            state=state,
            parent=model,
        )
        model._bela_stage2 = {
            "events": list(state.events),
            "final_patches": int(state.patch_pos.numel()),
            "rope_next_pos": int(getattr(model, "_bela_rope_next_pos", -1)),
            "dense_seq_len": dense_len,
        }
        model._bela_patch_pos = state.patch_pos
        model._bela_last_prefill_seq = int(outputs.last_hidden_state.shape[1])

        from transformers.models.llava_next.modeling_llava_next import LlavaNextModelOutputWithPast

        return LlavaNextModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
            image_hidden_states=None,
        )

    llava.get_image_features = types.MethodType(hooked_get_image_features, llava)
    llava.forward = types.MethodType(hooked_llava_forward, llava)
    if hasattr(model, "get_image_features"):
        model.get_image_features = types.MethodType(
            lambda self, *a, **k: llava.get_image_features(*a, **k), model
        )

    orig_generate = model.generate

    def hooked_generate(self, *args, **kwargs):
        if not enabled():
            return orig_generate(*args, **kwargs)

        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if input_ids is None:
            raise RuntimeError("BELA generate requires input_ids")
        pixel_values = kwargs.get("pixel_values", None)
        attention_mask = kwargs.get("attention_mask", None)
        image_sizes = kwargs.get("image_sizes", None)
        # Match Dense protocol: do not silently shrink vs task yaml unless OOM env set.
        # Default cap 256 — enough for VQA; override with BELA_NEXT_MAX_NEW.
        max_new_tokens = kwargs.get("max_new_tokens", None)
        if max_new_tokens is None:
            gen_cfg = kwargs.get("generation_config", None) or getattr(self, "generation_config", None)
            max_new_tokens = getattr(gen_cfg, "max_new_tokens", None) if gen_cfg is not None else None
        max_new_tokens = int(max_new_tokens or 32)
        max_new_tokens = max(1, min(max_new_tokens, int(os.environ.get("BELA_NEXT_MAX_NEW", "256"))))
        eos_id = kwargs.get("eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(self.config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_ids = set(int(x) for x in eos_id)
        elif eos_id is None:
            eos_ids = set()
        else:
            eos_ids = {int(eos_id)}

        past = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            pre = self(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                use_cache=True,
                return_dict=True,
            )
            if getattr(self, "_bela_stage2", None) is None:
                raise RuntimeError("PRUNING_NOT_ACTIVE: generate prefill missing Stage2")
            s1 = getattr(self, "_bela_stage1", {}) or {}
            if not s1.get("qa_log", {}).get("legacy_dual_called"):
                raise RuntimeError("PRUNING_NOT_ACTIVE: Dual COVER not called")
            if int(s1.get("qa_log", {}).get("qa_swaps_cfg", 0)) > 0 and not s1.get("qa_log", {}).get(
                "legacy_qa_called"
            ):
                raise RuntimeError("PRUNING_NOT_ACTIVE: QA B8 not called")
            past = pre.past_key_values
            short_len = int(pre.logits.shape[1])
            next_token = pre.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            del pre
            new_list = [next_token]
            device = input_ids.device
            dtype = attention_mask.dtype if attention_mask is not None else input_ids.dtype
            attn = torch.ones(input_ids.shape[0], short_len, device=device, dtype=dtype)
            next_pos = int(getattr(self, "_bela_rope_next_pos", short_len))

            for _ in range(max(0, max_new_tokens - 1)):
                tok = int(next_token.item())
                if eos_ids and tok in eos_ids:
                    break
                attn = torch.cat(
                    [attn, torch.ones(attn.shape[0], 1, device=device, dtype=dtype)], dim=1
                )
                pos = torch.tensor([[next_pos]], device=device, dtype=torch.long)
                step = self(
                    input_ids=next_token,
                    attention_mask=attn,
                    past_key_values=past,
                    position_ids=pos,
                    use_cache=True,
                    return_dict=True,
                )
                past = step.past_key_values
                next_pos += 1
                self._bela_rope_next_pos = next_pos
                next_token = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del step
                new_list.append(next_token)

            return torch.cat([input_ids] + new_list, dim=1)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return input_ids
        finally:
            past = None
            if hasattr(self, "_bela_position_ids"):
                self._bela_position_ids = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model.generate = types.MethodType(hooked_generate, model)

    # ForConditionalGeneration.forward also receives both; strip before model.model
    if hasattr(model, "forward"):
        orig_cg_fwd = model.forward

        def hooked_cg_forward(self, *args, **kwargs):
            if not enabled():
                pv = kwargs.get("pixel_values", None)
                mm = kwargs.get("mm_encoder_outputs", None)
                has_mm = isinstance(mm, dict) and mm.get("image") is not None
                if pv is not None and has_mm:
                    kwargs = dict(kwargs)
                    kwargs["pixel_values"] = None
                    kwargs["image_sizes"] = None
            return orig_cg_fwd(*args, **kwargs)

        model.forward = types.MethodType(hooked_cg_forward, model)
        model._bela_orig_cg_fwd = orig_cg_fwd

    model._bela_next_v2_patched = True
    model._bela_llama = llama_model
    model._bela_orig_llava_fwd = orig_llava_fwd
    model._bela_orig_get_feats = orig_get_feats
    model._bela_orig_generate = orig_generate
    model._bela_prefill_hits = 0
    try:
        n_layers = int(model.config.text_config.num_hidden_layers)
    except Exception:
        n_layers = int(getattr(llama_model.config, "num_hidden_layers", -1))
    model._bela_num_layers = n_layers
    return True


def summarize_trace(model) -> dict:
    return {
        "enabled": enabled(),
        "arm": os.environ.get("BELA_NEXT_ARM"),
        "stage1": getattr(model, "_bela_stage1", None),
        "stage2": getattr(model, "_bela_stage2", None),
        "prefill_hits": getattr(model, "_bela_prefill_hits", None),
        "last_prefill_seq": getattr(model, "_bela_last_prefill_seq", None),
        "patch_pos_n": None
        if getattr(model, "_bela_patch_pos", None) is None
        else int(model._bela_patch_pos.numel()),
        "rope_next_pos": getattr(model, "_bela_rope_next_pos", None),
    }
