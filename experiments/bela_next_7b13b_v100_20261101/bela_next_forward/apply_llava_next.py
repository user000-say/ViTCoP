"""Wire BELA Dual COVER + Stage2 into HF LlavaNext (Transformers 5.20).

BELA_NEXT_V2_ENABLE=0 → completely untouched stock forward (Dense parity).
ENABLE=1 → Stage1 Dual COVER on crop grids, pack via stock NeXT
spatial_unpad + image_newline (newlines always kept), then Stage2 Q_ATTN
physical cuts on *patch* tokens only.

CRITICAL engineering fidelity (vs earlier simple_crop_concat distortion):
  - Vision features go through the same AnyRes unpad + image_newline layout
    as stock `LlavaNextModel.pack_image_features`.
  - Stage1 only drops patch tokens; every newline stock NeXT emits is kept.
  - Stage2 never drops newlines/text; RoPE position_ids stay original abs ids.
  - Prefill/decode continue RoPE from max kept position (not compressed length).

CRITICAL (TF 5.20 generate path):
  GenerationMixin pre-encodes via get_image_features then pops pixel_values.
  We patch get_image_features and run a faithful greedy generate that passes
  explicit position_ids after Stage2 physical prune.
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path

import torch

from .dual_cover import dual_cover_five_crops
from .next_pack import pack_spatial_unpad_selected, valid_tile_local_ids
from .stage2 import Stage2State, select_and_prune

_ARMS = None


def _arms():
    global _ARMS
    if _ARMS is None:
        p = Path(__file__).resolve().parents[1] / "BELA_NEXT_7B13B_V100_CURSOR_20261101/configs/arms.json"
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


def crop_quotas(k0: int, n_crops: int) -> list:
    from bela_next_budget.crops import crop_quota_from_ids

    return crop_quota_from_ids([576] * int(n_crops), total=int(k0), policy="uniform")


def _rewrite_image_placeholders(input_ids, attention_mask, image_token_id: int, n_keep: int):
    """Collapse contiguous image-token block to exactly n_keep placeholders."""
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


def _image_size_row(image_sizes, batch_idx: int = 0):
    if image_sizes is None:
        raise RuntimeError("PRUNING_NOT_ACTIVE: image_sizes required for NeXT unpad pack")
    if isinstance(image_sizes, torch.Tensor):
        if image_sizes.ndim == 1:
            return image_sizes.tolist()
        return image_sizes[batch_idx].tolist()
    row = image_sizes[batch_idx]
    if isinstance(row, torch.Tensor):
        return row.tolist()
    return list(row)


def apply_patch(model) -> bool:
    if getattr(model, "_bela_next_v2_patched", False):
        return False

    llava = model.model  # LlavaNextModel
    lm = llava.language_model
    llama_model = lm.model if hasattr(lm, "model") else lm

    orig_llava_fwd = llava.forward
    orig_get_feats = llava.get_image_features

    def stage1_features(
        self_m,
        pixel_values,
        image_sizes,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        **kwargs,
    ):
        """Vision → Dual COVER per crop → stock spatial_unpad+newline pack."""
        arm = current_arm()
        k0, k1, k2 = [int(x) for x in arm["stage_k"]]
        if pixel_values.dim() == 5:
            assert pixel_values.shape[0] == 1, "BELA NeXT Stage1 batch=1 only"
            crops = pixel_values[0]
        else:
            crops = pixel_values
        n_crops = int(crops.shape[0])
        if n_crops < 1:
            raise RuntimeError("PRUNING_NOT_ACTIVE_LAYOUT: empty crops")

        kwargs = dict(kwargs)
        kwargs.pop("return_dict", None)
        kwargs["output_hidden_states"] = True
        image_outputs = self_m.vision_tower(crops, return_dict=True, **kwargs)
        hs = image_outputs.hidden_states
        mid = hs[12][:, 1:, :]
        pen = hs[-2][:, 1:, :]

        vfl = vision_feature_layer if vision_feature_layer is not None else self_m.config.vision_feature_layer
        vfs = vision_feature_select_strategy or self_m.config.vision_feature_select_strategy
        if isinstance(vfl, int):
            selected = hs[vfl]
        else:
            selected = torch.cat([hs[i] for i in vfl], dim=-1)
        if vfs == "default":
            selected = selected[:, 1:]
        projected = self_m.multi_modal_projector(selected)  # [C,576,D]
        if projected.shape[1] != 576:
            raise RuntimeError(f"PRUNING_NOT_ACTIVE_LAYOUT: projected {tuple(projected.shape)}")

        vis_size = int(self_m.config.vision_config.image_size)
        patch = int(self_m.config.vision_config.patch_size)
        pinpoints = self_m.config.image_grid_pinpoints
        img_hw = _image_size_row(image_sizes, 0)

        # Valid post-unpad pools (base = all 576; tiles = unpad survivors)
        valid_pools = [None] * n_crops
        if n_crops > 1:
            tile_pools = valid_tile_local_ids(
                img_hw,
                image_grid_pinpoints=pinpoints,
                vision_image_size=vis_size,
                patch_size=patch,
                device=projected.device,
            )
            if len(tile_pools) != n_crops - 1:
                raise RuntimeError(
                    f"tile pool count {len(tile_pools)} != n_crops-1={n_crops - 1}"
                )
            valid_pools = [None] + list(tile_pools)

        quotas = crop_quotas(k0, n_crops)
        # Clamp quotas to valid pool sizes so emitted patch count can hit K0 on square paper mode
        clamped = []
        for ci, q in enumerate(quotas):
            pool = valid_pools[ci]
            if pool is None:
                clamped.append(int(q))
            else:
                clamped.append(min(int(q), int(pool.numel())))
        # Redistribute shortfall to crops that still have room (stable crop order)
        short = int(k0) - int(sum(clamped))
        if short > 0:
            for ci in range(n_crops):
                if short <= 0:
                    break
                pool = valid_pools[ci]
                cap = 576 if pool is None else int(pool.numel())
                room = cap - clamped[ci]
                take = min(room, short)
                clamped[ci] += take
                short -= take
        if int(sum(clamped)) != int(k0) and all(
            (valid_pools[i] is None) or (valid_pools[i].numel() == 576) for i in range(n_crops)
        ):
            raise RuntimeError(f"quota sum {sum(clamped)} != K0 {k0} on full-valid crops")

        keep_flat, per = dual_cover_five_crops(
            mid, pen, clamped, slots=32, valid_pools=valid_pools
        )

        packed = pack_spatial_unpad_selected(
            projected,
            img_hw,
            per,
            self_m.image_newline,
            image_grid_pinpoints=pinpoints,
            vision_image_size=vis_size,
            patch_size=patch,
        )
        if packed.n_patches <= 0:
            raise RuntimeError("PRUNING_NOT_ACTIVE: zero patches after NeXT pack")
        # Paper-mode square: expect exact K0. Non-square unpad may emit fewer if
        # Dual COVER selected pad-only ids — we selected inside valid pools so
        # n_patches should equal sum(clamped).
        if packed.n_patches != int(sum(clamped)):
            raise RuntimeError(
                f"PRUNING_NOT_ACTIVE: packed patches {packed.n_patches} != selected {sum(clamped)}"
            )

        patch_local = torch.where(packed.is_patch)[0].to(dtype=torch.long)
        model._bela_stage1 = {
            "arm": arm["arm_id"],
            "k0": k0,
            "k1": k1,
            "k2": k2,
            "quotas": clamped,
            "quotas_requested": quotas,
            "keep_flat": keep_flat.detach().cpu().tolist(),
            "per_crop": [x.detach().cpu().tolist() for x in per],
            "selector": "E013_DUAL_COVER_CROPWISE",
            "packed_len": int(packed.features.shape[0]),
            "packed_patches": int(packed.n_patches),
            "packed_newlines": int(packed.n_newlines),
            "pack_mode": "spatial_unpad_newline",
            "unpad_hw": list(packed.unpad_hw),
            "grid": [packed.grid_nh, packed.grid_nw],
            "image_size_hw": list(img_hw),
            "n_crops": n_crops,
            "base_patches": int(packed.base_patches),
            "tile_patches": int(packed.tile_patches),
        }
        model._bela_keep_flat = keep_flat.detach().cpu()
        model._bela_visual_is_patch = packed.is_patch.detach().cpu()
        model._bela_visual_patch_local = patch_local.detach().cpu()
        model._bela_prefill_hits = int(getattr(model, "_bela_prefill_hits", 0)) + 1
        # Drop vision tower hidden_states — otherwise all CLIP layers stay alive
        # through LLM prefill and V100 OOMs on long prompts (mmbench).
        try:
            image_outputs.hidden_states = None
        except Exception:
            pass
        del mid, pen, selected, projected, hs
        image_outputs.pooler_output = [packed.features]
        return image_outputs

    def hooked_get_image_features(
        self_m,
        pixel_values,
        image_sizes,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        **kwargs,
    ):
        if not enabled():
            return orig_get_feats(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                **kwargs,
            )
        return stage1_features(
            self_m,
            pixel_values,
            image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            **kwargs,
        )

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
                **kwargs,
            )

        past_len = _past_len(past_key_values)
        # Decode steps: cache already Stage2-pruned; honor explicit position_ids
        if past_len > 0:
            return orig_llava_fwd(
                input_ids=input_ids,
                pixel_values=None,
                image_sizes=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                use_cache=use_cache,
                mm_encoder_outputs=None,
                **kwargs,
            )

        arm = current_arm()
        k0, k1, k2 = [int(x) for x in arm["stage_k"]]
        c1, c2 = [int(x) for x in arm["cuts_after_layers"]]

        mm = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        image_features = None
        if mm.get("image") is not None:
            pooled = mm["image"].pooler_output
            if isinstance(pooled, (list, tuple)):
                image_features = torch.cat(pooled, dim=0)
            else:
                image_features = pooled
            if getattr(model, "_bela_stage1", None) is None:
                raise RuntimeError(
                    "PRUNING_NOT_ACTIVE: mm_encoder_outputs present but Stage1 metadata missing "
                    "(get_image_features was not BELA-patched)"
                )
            if int(image_features.shape[0]) != int(model._bela_stage1["packed_len"]):
                raise RuntimeError(
                    f"PRUNING_NOT_ACTIVE: feature len {image_features.shape[0]} != "
                    f"Stage1 packed_len {model._bela_stage1['packed_len']}"
                )
        elif pixel_values is not None and pixel_values.numel() > 0:
            img_out = stage1_features(
                self_m,
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_features = img_out.pooler_output[0]
        else:
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
                **kwargs,
            )

        n_img = int(image_features.shape[0])
        n_patches = int(model._bela_stage1["packed_patches"])
        if n_patches > k0:
            raise RuntimeError(f"PRUNING_NOT_ACTIVE: patches {n_patches} > arm K0={k0}")

        image_token_id = self_m.config.image_token_id
        if input_ids is None:
            raise RuntimeError("BELA NeXT requires input_ids for placeholder rewrite")
        new_ids, new_mask, img_start = _rewrite_image_placeholders(
            input_ids, attention_mask, image_token_id, n_img
        )
        inputs_embeds = self_m.get_input_embeddings()(new_ids)
        special = (new_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            special, image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        )

        patch_local = model._bela_visual_patch_local.to(dtype=torch.long)
        patch_pos = patch_local + int(img_start)
        model._bela_patch_pos = patch_pos
        model._bela_text_start = int(img_start + n_img)
        model._bela_stage1["img_start"] = int(img_start)
        model._bela_stage1["seq_after_stage1"] = int(new_ids.shape[1])
        model._bela_stage1["prefill_via"] = "mm_encoder" if mm.get("image") is not None else "pixel_values"

        state = Stage2State(
            patch_pos=patch_pos.clone(),
            k_targets=(k1, k2),
            cut_layers=(c1, c2),
        )
        outputs = _llama_forward_with_cuts(
            llama_model,
            inputs_embeds=inputs_embeds,
            attention_mask=new_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=True if use_cache is None else use_cache,
            state=state,
            parent=model,
        )
        model._bela_stage2 = {
            "events": list(state.events),
            "final_patches": int(state.patch_pos.numel()),
            "rope_next_pos": int(getattr(model, "_bela_rope_next_pos", -1)),
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
        # lmms defaults max_new_tokens=1024 when task yaml omits it — hard-cap for V100.
        max_new_tokens = kwargs.get("max_new_tokens", None)
        if max_new_tokens is None:
            gen_cfg = kwargs.get("generation_config", None) or getattr(self, "generation_config", None)
            max_new_tokens = getattr(gen_cfg, "max_new_tokens", None) if gen_cfg is not None else None
        max_new_tokens = int(max_new_tokens or 32)
        max_new_tokens = max(1, min(max_new_tokens, int(os.environ.get("BELA_NEXT_MAX_NEW", "64"))))
        eos_id = kwargs.get("eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(self.config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_ids = set(int(x) for x in eos_id)
        elif eos_id is None:
            eos_ids = set()
        else:
            eos_ids = {int(eos_id)}
        # Do NOT force-stop on newline — that distorted SciQA ("B." vs "B") vs Dense.

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
                raise RuntimeError("PRUNING_NOT_ACTIVE: generate prefill missing Stage2 trace")
            past = pre.past_key_values
            short_len = int(pre.logits.shape[1])
            next_token = pre.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            # Free prefill logits/hidden ASAP (only need last token + cache)
            del pre
            new_list = [next_token]
            device = input_ids.device
            dtype = attention_mask.dtype if attention_mask is not None else input_ids.dtype
            attn = torch.ones(input_ids.shape[0], short_len, device=device, dtype=dtype)
            # Continue RoPE from preserved absolute positions, not compressed cache length
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
            # Let lmms continue: return prompt-only so batch_decode gets a tensor
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
