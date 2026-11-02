"""BELA NeXT7 native packing, actual CLS Dual COVER, QA B8 and L12/L20 cuts.

Version-locked Transformers 5.20.2 implementation. Disabling BELA passes
through the original model without mutating its Dense input arguments.
This experimental generate path supports strict greedy decoding only.
"""
from __future__ import annotations
import hashlib
import json
import os
import types
from pathlib import Path
import torch
from .native_stage1 import run_native_stage1
from .stage2 import Stage2State, select_and_prune, _layer_qk
from .bela_next_compat import original_dual, original_qa

_ROOT = Path(__file__).resolve().parents[1]
_ARMS = None


def enabled():
    return os.environ.get('BELA_NEXT_V2_ENABLE', '0').lower() in ('1', 'true', 'yes')


def _arms():
    global _ARMS
    if _ARMS is None:
        data = json.loads((_ROOT/'next_configs'/'arms.json').read_text())
        arms = {}
        for arm in data['arm_matrix']:
            a,b = map(int, arm['cuts_after_layers'])
            k0,k1,k2 = map(int, arm['stage_k'])
            layers = int(arm['layers'])
            w = a*k0 + (b-a)*k1 + (layers-b)*k2
            if w != int(arm['visual_patch_token_layer_W']) or w != layers*int(arm['budget_T']):
                raise RuntimeError(f'BUDGET_MISMATCH: {arm["arm_id"]} {w}')
            if arm['arm_id'] in arms:
                raise RuntimeError('duplicate arm ID')
            arms[arm['arm_id']] = arm
        _ARMS = arms
    return _ARMS


def current_arm():
    arm_id = os.environ.get('BELA_NEXT_ARM', '').strip()
    if arm_id not in _arms():
        raise RuntimeError(f'BELA_NEXT_ARM invalid/missing: {arm_id!r}')
    arm = _arms()[arm_id]
    if arm['model'] != 'next7' or arm['layers'] != 32 or tuple(arm['cuts_after_layers']) != (12,20):
        raise RuntimeError('This fixed implementation is verified only for NeXT-7B 32-layer L12/L20')
    return arm


def _verify_runtime():
    if os.environ.get('BELA_REQUIRE_RUNTIME_HASH', '1') == '0':
        return
    import transformers
    if transformers.__version__ != '5.20.2':
        raise RuntimeError(f'Version mismatch: need transformers 5.20.2, found {transformers.__version__}')
    import transformers.models.llama.modeling_llama as llama
    import transformers.models.llava_next.modeling_llava_next as next_model
    meta = json.loads((_ROOT/'reference'/'original_runtime_tf520'/'IMPORT_PATHS.json').read_text())
    for key, mod in (('modeling_llama', llama), ('modeling_llava_next', next_model)):
        path = Path(mod.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta['files'][key]['sha256']:
            raise RuntimeError(f'RUNTIME_HASH_MISMATCH {key}: expected {meta["files"][key]["sha256"]}, got {digest}')


def _rewrite_image_placeholders(input_ids, attention_mask, image_token_id, replacement_count):
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError('BATCH1_ONLY: model prompt must have shape [1,L]')
    ids = input_ids[0]
    matches = torch.where(ids == image_token_id)[0]
    if matches.numel() == 0 or replacement_count <= 0:
        raise RuntimeError('IMAGE_PLACEHOLDER_MISSING')
    start, end = int(matches[0]), int(matches[-1])+1
    if matches.numel() != end-start:
        raise RuntimeError('NONCONTIGUOUS_IMAGE_PLACEHOLDERS: multi-image unsupported')
    output = torch.cat((ids[:start], ids.new_full((replacement_count,), image_token_id), ids[end:])).unsqueeze(0)
    if attention_mask is None:
        new_mask = torch.ones_like(output)
    else:
        if attention_mask.shape != input_ids.shape:
            raise RuntimeError('attention_mask does not match original prompt')
        am = attention_mask[0]
        new_mask = torch.cat((am[:start], am.new_ones(replacement_count), am[end:])).unsqueeze(0)
    return output, new_mask, start


def _past_len(past):
    return 0 if past is None else int(past.get_seq_length())


def _remap_text_start(keep_abs: torch.Tensor, old_text_start: int, old_seq_len: int) -> int:
    """Map the FIRST true text position through Stage2 physical gathers.

    Keeping `max(patch_pos)+1` is WRONG if the last kept patch is followed by
    one or more surviving AnyRes image_newline markers: that would turn those
    structural markers into question queries for the second prune.
    """
    if old_seq_len < 1 or not 0 < int(old_text_start) < old_seq_len:
        raise RuntimeError('TEXT_START_INVALID: must have a nonempty question suffix')
    keep = torch.as_tensor(keep_abs, dtype=torch.long).reshape(-1)
    if keep.numel() == 0 or keep.unique().numel() != keep.numel():
        raise RuntimeError('TEXT_REMAP_INVALID_KEEP: empty or duplicate gather')
    if int(keep.min()) < 0 or int(keep.max()) >= old_seq_len or not bool((keep[1:] > keep[:-1]).all()):
        raise RuntimeError('TEXT_REMAP_INVALID_KEEP: gather not sorted, unique or in range')
    if not bool((keep >= old_text_start).any()):
        raise RuntimeError('TEXT_REMAP_NO_TEXT: pruning attempted to remove all text')
    return int((keep < old_text_start).sum())


def _verify_native_placeholders(input_ids, attention_mask, image_token_id, original_packed_length):
    """Check native Dense image packing BEFORE any BELA placeholder rewriting."""
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError('BATCH1_ONLY: image input IDs required')
    patch_positions = torch.where(input_ids[0] == image_token_id)[0]
    if patch_positions.numel() != original_packed_length:
        raise RuntimeError(
            f'NATIVE_PLACEHOLDER_MISMATCH: processor={patch_positions.numel()} '
            f'native_packer={original_packed_length}; do not conceal a Dense AnyRes error'
        )
    if not patch_positions.numel() or (int(patch_positions[-1] - patch_positions[0] + 1) != patch_positions.numel()):
        raise RuntimeError('NONCONTIGUOUS_IMAGE_PLACEHOLDERS: image layout unsupported')
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape or not bool(attention_mask[0,patch_positions].bool().all()):
            raise RuntimeError('IMAGE_PLACEHOLDER_ATTENTION_MASK_MISMATCH')
    return int(patch_positions[0])


def _llama_forward_with_cuts(llama, *, inputs_embeds, attention_mask, position_ids,
                             past_key_values, use_cache, state: Stage2State, parent):
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast
    if inputs_embeds.shape[0] != 1 or not use_cache or past_key_values is not None:
        raise RuntimeError('BELA only supports batch1, empty cache, use_cache=True prefill')
    cache = DynamicCache(config=llama.config)
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
    if position_ids.shape != (1, inputs_embeds.shape[1]):
        raise RuntimeError('Invalid prefill position_ids')
    if attention_mask is None or attention_mask.shape != position_ids.shape:
        raise RuntimeError('Expected exact 2-D prefill attention mask')
    hs = inputs_embeds
    text_start = int(parent._bela_text_start)
    cuts = dict(zip(state.cut_layers, state.k_targets))
    for index, layer in enumerate(llama.layers[:llama.config.num_hidden_layers]):
        # create_causal_mask in HF5.20 uses *physical index* while RoPE receives
        # original Stage1 position_ids. Keep these namespaces independent.
        causal = create_causal_mask(config=llama.config, inputs_embeds=hs,
                                    attention_mask=attention_mask, past_key_values=None,
                                    position_ids=position_ids)
        rope = llama.rotary_emb(hs, position_ids=position_ids)
        layer_num = index+1
        q = k = None
        if layer_num in cuts and state.patch_pos.numel() > cuts[layer_num]:
            q, k = _layer_qk(layer, hs, rope)
        hs = layer(hs, attention_mask=causal, position_embeddings=rope,
                   position_ids=position_ids, past_key_values=cache, use_cache=True)
        if layer_num in cuts and state.patch_pos.numel() > cuts[layer_num]:
            before = int(hs.shape[1])
            hs, position_ids, cache, state = select_and_prune(
                state=state, hidden_states=hs, position_ids=position_ids,
                past_key_values=cache, decoder_layer=layer, position_embeddings=rope,
                layer_1based=layer_num, k_target=cuts[layer_num], layers_filled=index+1,
                text_start=text_start, q=q, k=k, valid_token_mask=attention_mask)
            idx = state.last_keep_abs.to(device=attention_mask.device)
            if idx.numel() >= before:
                raise RuntimeError('Layer pruning did not reduce physical sequence length')
            # DO NOT reset padding to ones; retain original mask semantics.
            attention_mask = attention_mask.index_select(-1, idx)
            if int(cache.get_seq_length()) != hs.shape[1]:
                raise RuntimeError('post-cut DynamicCache sequence length mismatch')
            # Preserve true text boundary through physical gather; trailing
            # image_newline is not a text Query in Q_ATTN.
            next_text_start = _remap_text_start(idx, text_start, before)
            state.events[-1]['true_text_start_before'] = int(text_start)
            state.events[-1]['true_text_start_after'] = int(next_text_start)
            state.events[-1]['padding_after'] = int((attention_mask == 0).sum())
            text_start = next_text_start
            if text_start <= int(state.patch_pos.max()):
                raise RuntimeError('TEXT_START_MAPPING: text boundary overlapped an image patch')
            parent._bela_text_start = text_start
    hs = llama.norm(hs)
    parent._bela_position_ids = position_ids.detach()
    parent._bela_rope_next_pos = int(position_ids.max())+1
    parent._bela_last_prefill_mask = attention_mask.detach()
    parent._bela_last_prefill_seq = hs.shape[1]
    return BaseModelOutputWithPast(last_hidden_state=hs, past_key_values=cache)


def _generation_parameters(model, input_ids, kwargs):
    """Fail-closed API: unsupported sampling/beams/extra processors are errors."""
    generation_cfg = kwargs.get('generation_config') or getattr(model, 'generation_config', None)
    def opt(name, fallback):
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]
        return getattr(generation_cfg, name, fallback) if generation_cfg is not None else fallback
    allow_keys = {'input_ids','pixel_values','image_sizes','attention_mask','max_new_tokens',
                  'generation_config','eos_token_id','pad_token_id','do_sample','num_beams',
                  'stopping_criteria','return_dict_in_generate','output_scores',
                  'temperature','top_p','use_cache','max_length', 'repetition_penalty'}
    extra = {k:v for k,v in kwargs.items() if k not in allow_keys and v is not None}
    if extra:
        raise RuntimeError(f'UNSUPPORTED_GENERATION_OPTIONS: {sorted(extra)}')
    if bool(opt('do_sample', False)) or int(opt('num_beams', 1)) != 1:
        raise RuntimeError('Only greedy non-sampling, single-beam generation is supported')
    if float(opt('repetition_penalty', 1.0)) != 1.0:
        raise RuntimeError('repetition_penalty alters logits; unsupported')
    t = opt('temperature', 1.0)
    if t is not None and float(t) not in (0., 1.):
        raise RuntimeError('Nontrivial temperature unsupported')
    p = opt('top_p', 1.0)
    if p is not None and float(p) != 1.0:
        raise RuntimeError('top_p unsupported for this strict greedy decoder')
    if not bool(opt('use_cache', True)):
        raise RuntimeError('BELA pruned generation requires use_cache=True')
    if bool(opt('return_dict_in_generate', False)) or bool(opt('output_scores', False)):
        raise RuntimeError('Generation scores/dict output unsupported: expected plain token IDs')
    if int(opt('num_return_sequences', 1)) != 1:
        raise RuntimeError('BELA only supports one generated candidate')
    # Non-default logits processors / output constraints must not be silently skipped.
    unsupported_active = {
        'min_length': (0, None), 'min_new_tokens': (0, None),
        'forced_bos_token_id': (None,), 'forced_eos_token_id': (None,),
        'bad_words_ids': (None, []), 'force_words_ids': (None, []),
        'no_repeat_ngram_size': (0, None), 'encoder_no_repeat_ngram_size': (0, None),
        'suppress_tokens': (None, []), 'begin_suppress_tokens': (None, []),
        'sequence_bias': (None, {}), 'stop_strings': (None, []),
        'constraints': (None, []), 'max_time': (None,),
        'exponential_decay_length_penalty': (None,),
        'watermarking_config': (None,), 'guidance_scale': (None, 1.0),
    }
    for key, inactive in unsupported_active.items():
        value = opt(key, None)
        if value not in inactive:
            raise RuntimeError(f'UNSUPPORTED_NONDEFAULT_GENERATION_CONFIG: {key}={value!r}')
    max_new = opt('max_new_tokens', None)
    max_len = opt('max_length', None)
    if max_new is None:
        if max_len is None:
            raise RuntimeError('max_new_tokens or max_length required for reproducible eval')
        max_new = int(max_len) - int(input_ids.shape[-1])
    max_new = int(max_new)
    if max_new <= 0:
        raise RuntimeError('Invalid max_new_tokens or max_length')
    if max_len is not None and 'max_new_tokens' not in kwargs and max_new + input_ids.shape[-1] > int(max_len):
        raise RuntimeError('max_length violated')
    eos = opt('eos_token_id', None)
    eos_ids = set() if eos is None else ({int(x) for x in eos} if isinstance(eos,(list,tuple)) else {int(eos)})
    return max_new, eos_ids


def apply_patch(model):
    """Patch the existing HF instance, never its global class or site-packages."""
    if getattr(model, '_bela_next_v2_patched', False):
        return False
    _verify_runtime()
    if model.training:
        raise RuntimeError('BELA requires .eval() model')
    llava = model.model
    llama = llava.language_model.model if hasattr(llava.language_model, 'model') else llava.language_model
    if int(llama.config.num_hidden_layers) != 32:
        raise RuntimeError('Expected NeXT7 32 decoder layers')
    orig_fwd, orig_gen = llava.forward, model.generate

    def hooked_llava_forward(self_m, input_ids=None, pixel_values=None, image_sizes=None,
                             attention_mask=None, position_ids=None, past_key_values=None,
                             inputs_embeds=None, vision_feature_layer=None,
                             vision_feature_select_strategy=None, use_cache=None,
                             mm_encoder_outputs=None, **kwargs):
        if not enabled():
            # PASS-THROUGH: do not silently change Dense inputs or failure semantics.
            return orig_fwd(input_ids=input_ids, pixel_values=pixel_values,
                           image_sizes=image_sizes, attention_mask=attention_mask,
                           position_ids=position_ids, past_key_values=past_key_values,
                           inputs_embeds=inputs_embeds, vision_feature_layer=vision_feature_layer,
                           vision_feature_select_strategy=vision_feature_select_strategy,
                           use_cache=use_cache, mm_encoder_outputs=mm_encoder_outputs, **kwargs)
        if _past_len(past_key_values) > 0:
            if pixel_values is not None or mm_encoder_outputs is not None:
                raise RuntimeError('DECODE_IMAGE_REUSE: multimodal inputs forbidden after prefill')
            return orig_fwd(input_ids=input_ids, pixel_values=None, image_sizes=None,
                           attention_mask=attention_mask, position_ids=position_ids,
                           past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                           vision_feature_layer=vision_feature_layer,
                           vision_feature_select_strategy=vision_feature_select_strategy,
                           use_cache=use_cache, mm_encoder_outputs=None, **kwargs)
        if mm_encoder_outputs is not None or pixel_values is None or image_sizes is None:
            raise RuntimeError('PRUNING_NOT_ACTIVE: pruned prefill requires real pixels + image_sizes')
        if inputs_embeds is not None or input_ids is None or use_cache is False:
            raise RuntimeError('BELA prefill requires input_ids and use_cache=True')
        if kwargs.get('labels') is not None:
            raise RuntimeError('BELA is inference-only; labels/training unsupported')
        arm = current_arm()
        stage = run_native_stage1(self_m, pixel_values, image_sizes, input_ids, arm,
                                  dual_fn=original_dual, qa_fn=original_qa,
                                  vision_feature_layer=vision_feature_layer,
                                  vision_feature_select_strategy=vision_feature_select_strategy)
        image_feats = stage.packed_features
        native_packed_length = stage.native_patch_count + stage.native_newlines
        _verify_native_placeholders(input_ids, attention_mask,
                                    int(self_m.config.image_token_id), native_packed_length)
        new_ids, new_mask, image_start = _rewrite_image_placeholders(
            input_ids, attention_mask, int(self_m.config.image_token_id), int(image_feats.shape[0]))
        embeds = self_m.get_input_embeddings()(new_ids)
        image_feats = image_feats.to(device=embeds.device, dtype=embeds.dtype)
        patch_mask = (new_ids == int(self_m.config.image_token_id)).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(patch_mask, image_feats)
        local_patch = torch.where(stage.is_patch)[0].to(device=embeds.device, dtype=torch.long)
        physical_patch = local_patch + image_start
        state = Stage2State(patch_pos=physical_patch,
                            k_targets=(stage.k1, stage.k2), cut_layers=stage.cut_layers,
                            patch_uids=stage.patch_uids)
        model._bela_text_start = image_start + image_feats.shape[0]
        model._bela_stage1 = {
            'arm': stage.arm_id, 'k0': stage.k0,'k1': stage.k1,'k2': stage.k2,
            'packed_len': int(image_feats.shape[0]), 'packed_patches': stage.n_patches,
            'packed_newlines': stage.n_newlines, 'native_patches': stage.native_patch_count,
            'native_newlines': stage.native_newlines,
            'native_packed_length': native_packed_length,
            'pack_mode': stage.pack_mode,
            'mid_layer': stage.mid_layer, 'qa_log': stage.qa_log,
            'img_start': image_start, 'seq_after_stage1': int(new_ids.shape[1]),
            'true_text_start': int(image_start+image_feats.shape[0]),
        }
        out = _llama_forward_with_cuts(llama, inputs_embeds=embeds, attention_mask=new_mask,
                                       position_ids=position_ids, past_key_values=None,
                                       use_cache=True, state=state, parent=model)
        model._bela_stage2 = {
            'events': state.events, 'final_patches': int(state.patch_pos.numel()),
            'final_patch_uid_sample': [] if state.patch_uids is None else state.patch_uids[:16].tolist(),
            'rope_next_pos': int(model._bela_rope_next_pos),
        }
        model._bela_patch_pos = state.patch_pos
        model._bela_prefill_hits = int(getattr(model, '_bela_prefill_hits', 0))+1
        if len(state.events) != 2 or int(state.patch_pos.numel()) != stage.k2:
            raise RuntimeError('PRUNING_NOT_ACTIVE: Stage2 events incomplete')
        from transformers.models.llava_next.modeling_llava_next import LlavaNextModelOutputWithPast
        return LlavaNextModelOutputWithPast(last_hidden_state=out.last_hidden_state,
                                            past_key_values=out.past_key_values,
                                            hidden_states=None, attentions=None, image_hidden_states=None)

    llava.forward = types.MethodType(hooked_llava_forward, llava)

    @torch.no_grad()
    def hooked_generate(self, *args, **kwargs):
        if not enabled():
            return orig_gen(*args, **kwargs)
        if args:
            if len(args) != 1 or 'input_ids' in kwargs:
                raise RuntimeError('Ambiguous positional generation arguments')
            kwargs = dict(kwargs, input_ids=args[0])
        input_ids = kwargs.get('input_ids')
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError('BELA generation requires batch1 input_ids')
        max_new, eos_ids = _generation_parameters(self, input_ids, kwargs)
        if kwargs.get('pixel_values') is None or kwargs.get('image_sizes') is None:
            raise RuntimeError('BELA generation requires a real image')
        self._bela_stage2 = None
        pre = self(input_ids=input_ids, attention_mask=kwargs.get('attention_mask'),
                   image_sizes=kwargs['image_sizes'], pixel_values=kwargs['pixel_values'],
                   use_cache=True, logits_to_keep=1, return_dict=True)
        if not getattr(self, '_bela_stage2', None):
            raise RuntimeError('PRUNING_NOT_ACTIVE: missing pruned prefill')
        evidence = self._bela_stage1['qa_log']
        if not evidence['legacy_dual_called'] or (evidence['qa_swaps_cfg'] and not evidence['legacy_qa_called']):
            raise RuntimeError('PRUNING_NOT_ACTIVE: missing required BELA operators')
        past = pre.past_key_values
        mask = self._bela_last_prefill_mask.clone()
        next_pos = int(self._bela_rope_next_pos)
        token = pre.logits[:, -1, :].argmax(-1, keepdim=True)
        if not bool(torch.isfinite(pre.logits).all()):
            raise RuntimeError('NONFINITE_LOGITS at pruned prefill')
        generated = [token]
        del pre
        stopping = kwargs.get('stopping_criteria')
        if stopping is not None and not callable(stopping):
            raise RuntimeError('stopping_criteria must be callable')
        def should_stop():
            if int(token.item()) in eos_ids:
                return True
            if stopping is None:
                return False
            sequences = torch.cat([input_ids]+generated, dim=-1)
            value = stopping(sequences, scores=None)
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise RuntimeError('stopping_criteria must return one bool for batch1')
                return bool(value.item())
            return bool(value)
        for _ in range(max_new - 1):
            if should_stop():
                break
            mask = torch.cat((mask, mask.new_ones((1,1))), -1)
            pos = torch.tensor([[next_pos]], device=token.device, dtype=torch.long)
            output = self(input_ids=token, attention_mask=mask,
                          past_key_values=past, position_ids=pos,
                          use_cache=True, logits_to_keep=1, return_dict=True)
            if not bool(torch.isfinite(output.logits).all()):
                raise RuntimeError('NONFINITE_LOGITS at pruned decode')
            past = output.past_key_values
            token = output.logits[:,-1,:].argmax(-1, keepdim=True)
            generated.append(token)
            next_pos += 1
            del output
        # One-token output is valid even if it immediately matches EOS/stop.
        self._bela_rope_next_pos = next_pos
        return torch.cat([input_ids]+generated, -1)

    model.generate = types.MethodType(hooked_generate, model)
    model._bela_next_v2_patched = True
    model._bela_orig_generate = orig_gen
    model._bela_orig_llava_fwd = orig_fwd
    model._bela_prefill_hits = 0
    return True


def summarize_trace(model):
    return dict(enabled=enabled(), arm=os.environ.get('BELA_NEXT_ARM'),
                stage1=getattr(model, '_bela_stage1', None),
                stage2=getattr(model, '_bela_stage2', None),
                prefill_hits=getattr(model, '_bela_prefill_hits', None),
                last_prefill_seq=getattr(model, '_bela_last_prefill_seq', None),
                rope_next_pos=getattr(model, '_bela_rope_next_pos', None))
