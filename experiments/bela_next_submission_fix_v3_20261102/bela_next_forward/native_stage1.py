"""BELA native LLaVA-NeXT Stage1, with explicit original-patch identity.

Transformers 5.20.2 contract: trim processor padding by image_sizes; ViT once;
real penultimate CLS->patch attention; stock pack_image_features; spatially
correct Dual COVER and question-only QA on *packed patch-only* IDs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import os
import torch
from bela_next_fix.legacy_bridge import BelaConfig, run_original_bela
from bela_next_fix.native_layout import (
    map_native_packing, pack_dual_depth, structural_keep_mask, _normalize_packer_result,
)


@dataclass
class NativeStage1Result:
    packed_features: torch.Tensor
    is_patch: torch.Tensor
    n_patches: int
    n_newlines: int
    patch_uids: torch.Tensor
    native_patch_count: int
    native_newlines: int
    qa_log: dict
    arm_id: str
    k0: int
    k1: int
    k2: int
    cut_layers: Tuple[int, int]
    pack_mode: str = 'spatial_unpad_newline'
    mid_layer: int = 9


def _arm_to_cfg(arm):
    k0, k1, k2 = map(int, arm['stage_k'])
    return BelaConfig(stage1=k0, cut12=k1, cut20=k2,
                      slots=int(arm.get('slots', 32)), qa_swaps=int(arm.get('qa_swaps', 8)),
                      donor=int(arm.get('donor_pool', 16)), drop=int(arm.get('drop_pool', 12)),
                      seed=int(arm.get('seed', 17)), mid_layer=int(arm.get('mid_layer', 9)),
                      pen_layer=int(arm.get('pen_layer', -2)), require_true_qa=True)


def _tokenizer_for(model):
    """Require the exact tokenizer instance that encoded the prompt.

    Inferring a tokenizer from a model path is not equivalent: chat templates,
    added tokens, and the fast/slow implementation can change QA boundaries.
    The lmms-eval entry is responsible for explicitly attaching this instance.
    """
    tok = getattr(model, '_tokenizer', None)
    if tok is None:
        raise RuntimeError(
            'QA_TOKENIZER_IDENTITY_REQUIRED: attach the same lmms-eval tokenizer '
            'to llava_model._tokenizer; AutoTokenizer fallback forbidden'
        )
    return tok


def _find_subseq(hay, needle, *, last=False, start=0):
    if not needle or int(hay.numel()) < len(needle):
        return -1
    positions = (range(int(hay.numel())-len(needle), start-1, -1) if last
                 else range(start, int(hay.numel())-len(needle)+1))
    for i in positions:
        if hay[i:i+len(needle)].tolist() == list(needle):
            return i
    return -1


def _question_ids_from_input(input_ids, image_token_id, tokenizer=None):
    """Slice original token IDs only; never decode/re-tokenize or silently take prompt.

    Supports single-turn Vicuna USER...ASSISTANT template. Unknown templates
    fail rather than poisoning QA with dialogue wrappers or answer tokens.
    """
    if input_ids.ndim != 1 or tokenizer is None:
        raise ValueError('1-D input and actual tokenizer required')
    ids = input_ids[input_ids != image_token_id]
    if ids.numel() and getattr(tokenizer, 'bos_token_id', None) is not None and int(ids[0]) == int(tokenizer.bos_token_id):
        ids = ids[1:]
    starts = []
    for marker in ('USER:', 'User:', 'user:'):
        tok = tokenizer(marker, add_special_tokens=False).input_ids
        i = _find_subseq(ids, tok, last=True)
        if i >= 0:
            starts.append((i, i + len(tok)))
    if not starts:
        raise RuntimeError('QA_QUESTION_BOUNDARY: USER: not found as original token subsequence; audit prompt template')
    start = max(starts)[1]
    ends = []
    for marker in ('ASSISTANT:', 'Assistant:', 'assistant:'):
        tok = tokenizer(marker, add_special_tokens=False).input_ids
        i = _find_subseq(ids, tok, start=start)
        if i >= 0:
            ends.append(i)
    if not ends:
        raise RuntimeError('QA_QUESTION_BOUNDARY: ASSISTANT: not found; refuse templated/full prompt fallback')
    end = min(ends)
    if end <= start:
        raise RuntimeError('QA_QUESTION_BOUNDARY: empty/inverted question span')
    result = ids[start:end]
    # Remove whitespace-only token pieces *without* re-tokenizing any other token.
    while result.numel() and not tokenizer.decode([int(result[0])], skip_special_tokens=False).strip():
        result = result[1:]
    while result.numel() and not tokenizer.decode([int(result[-1])], skip_special_tokens=False).strip():
        result = result[:-1]
    if not result.numel():
        raise RuntimeError('QA_QUESTION_BOUNDARY: empty question-only token sequence')
    return result


def _trim_to_native_crops(pixel_values, image_sizes, config):
    """Mirror HF get_image_features.image_size_to_num_patches for one image."""
    if image_sizes is None or len(image_sizes) != 1:
        raise RuntimeError('SINGLE_IMAGE_ONLY: one image_sizes record required')
    from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches
    required = int(image_size_to_num_patches(image_size=image_sizes[0],
                   grid_pinpoints=config.image_grid_pinpoints, patch_size=config.vision_config.image_size))
    if pixel_values.ndim == 5:
        if pixel_values.shape[0] != 1:
            raise RuntimeError('BATCH1_ONLY: split batch before pruning')
        crops = pixel_values[0]
    elif pixel_values.ndim == 4:
        crops = pixel_values
    else:
        raise RuntimeError('Invalid NeXT pixel_values shape')
    if required <= 0 or required > crops.shape[0]:
        raise RuntimeError(f'AnyRes crop capacity mismatch: needed={required}, provided={crops.shape[0]}')
    return crops[:required].contiguous(), required


@torch.no_grad()
def cls_scores_from_clip_qk(vision_tower, hidden_states, *, pen_index=-2):
    """Exact CLS softmax via CLIP penultimate block input LN/Q/K, O(C*H*S).

    Uses the same tensors/projections as TF5.20 CLIPAttention, but computes
    only its CLS query row instead of all S² rows in every ViT block.
    Compare to actual eager attentions[-2] in the oracle smoke.
    """
    try:
        layer = vision_tower.vision_model.encoder.layers[pen_index]
    except (AttributeError, IndexError) as exc:
        raise RuntimeError('CLIP_CLS_QK_UNSUPPORTED: expected CLIPVisionModel.encoder.layers') from exc
    if layer.training:
        raise RuntimeError('CLS probe requires eval()')
    # hs[0] = embeddings; hs[-3] = input to the penultimate ViT block.
    pre_h = hidden_states[pen_index - 1]
    h = layer.layer_norm1(pre_h)
    attn = layer.self_attn
    batch, length, _ = h.shape
    heads, dim = int(attn.num_heads), int(attn.head_dim)
    qcls = attn.q_proj(h[:, :1, :]).view(batch, 1, heads, dim).transpose(1, 2)
    k = attn.k_proj(h).view(batch, length, heads, dim).transpose(1, 2)
    weights = torch.matmul(qcls.float(), k.float().transpose(-1, -2)) * float(attn.scale)
    # CLS query's normalized distribution includes CLS key, then discard it.
    scores = weights.softmax(-1)[:, :, 0, 1:].mean(dim=1)
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError('Nonfinite true CLS attention')
    return scores


def _packed_cls_saliency(pack_fn, src_cls, layout, n_crops, image_sizes, strategy):
    if strategy != 'default':
        raise RuntimeError('CLS_NATIVE_UNSUPPORTED: only default CLS-removed vision strategy verified')
    if src_cls.ndim != 2 or src_cls.shape != (n_crops, layout.base_image_patch_counts[0]):
        raise RuntimeError('Real CLS scores do not match native crop patch count')
    native, lens = _normalize_packer_result(pack_fn(
        [src_cls.unsqueeze(-1)], image_sizes, vision_feature_select_strategy=strategy,
        image_newline=torch.zeros(1, device=src_cls.device, dtype=src_cls.dtype)))
    if native.shape != (layout.patch_ids.numel(), 1) or not torch.equal(lens, layout.image_lens):
        raise RuntimeError('CLS saliency packing mismatches native packed layout')
    scores = native[:,0][layout.patch_mask]
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError('CLS packed scores nonfinite')
    return scores


def _require_paper_crop_protocol(arm: dict, native_n_crops: int) -> None:
    """Fail closed when a nominal 2880-patch five-crop evaluation drifts.

    Without this gate, dynamic 1/3/5-crop inputs could be scored as a fixed
    94.4%-compression experiment despite incomparable native patch counts.
    """
    policy = arm.get('sampling_policy')
    if policy == 'locked_paper_five_crop' and int(native_n_crops) != 5:
        raise RuntimeError(
            f'PAPER_CROP_PROTOCOL_MISMATCH: expected 5 native crops, '
            f'got {native_n_crops}; audit processor/grid and matched-W before ACC'
        )


@torch.no_grad()
def run_native_stage1(llava_model, pixel_values, image_sizes, input_ids, arm, *,
                      dual_fn, qa_fn, vision_feature_layer=None, vision_feature_select_strategy=None):
    cfg = _arm_to_cfg(arm)
    cfg.check()
    if llava_model.training:
        raise RuntimeError('BELA pruning is eval-only')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError('BELA is batch1-only, must not mix image placeholders across samples')
    crops, native_n_crops = _trim_to_native_crops(pixel_values, image_sizes, llava_model.config)
    _require_paper_crop_protocol(arm, native_n_crops)
    vfl = (vision_feature_layer if vision_feature_layer is not None else llava_model.config.vision_feature_layer)
    vfs = (vision_feature_select_strategy if vision_feature_select_strategy is not None
           else llava_model.config.vision_feature_select_strategy)
    if vfs != 'default':
        raise RuntimeError('Unsupported NeXT vision_feature_select_strategy; audit before enabling')
    mode = os.environ.get('BELA_STAGE1_SALIENCY', 'cls_attention')
    if mode not in ('cls_attention', 'norm_proxy'):
        raise RuntimeError('Unknown BELA_STAGE1_SALIENCY mode')
    # Only the norm_proxy control skips true attentions; default requires real CLS.
    oracle = os.environ.get('BELA_VIT_CLS_ORACLE', '0') == '1' and mode == 'cls_attention'
    out = llava_model.vision_tower(crops, output_hidden_states=True,
                                  output_attentions=oracle, return_dict=True)
    hs = out.hidden_states
    selected = hs[vfl] if isinstance(vfl, int) else torch.cat([hs[i] for i in vfl], dim=-1)
    selected = selected[:, 1:]
    projected = llava_model.multi_modal_projector(selected)
    pack_fn = llava_model.pack_image_features
    per_image = [projected]
    native_packed, native_lens = _normalize_packer_result(pack_fn(
        per_image, image_sizes, vision_feature_select_strategy=vfs, image_newline=llava_model.image_newline))
    layout = map_native_packing(pack_fn, per_image, image_sizes, strategy=vfs)
    if os.environ.get('BELA_NATIVE_DENSE_ORACLE','0') == '1':
        # Expensive, one-sample GPU smoke ONLY: same original processor inputs,
        # actual HF get_image_features path (no BELA, no altered image grid).
        dense_visual = llava_model.get_image_features(
            pixel_values=pixel_values, image_sizes=image_sizes,
            vision_feature_layer=vfl, vision_feature_select_strategy=vfs,
            return_dict=True)
        dense_packed = torch.cat(dense_visual.pooler_output, dim=0)
        if dense_packed.shape != native_packed.shape or not torch.allclose(
                dense_packed.float(), native_packed.float(), atol=2e-3, rtol=2e-3):
            raise RuntimeError('NATIVE_DENSE_PACK_ORACLE_MISMATCH: AnyRes projected stream differs from Dense')
        del dense_visual, dense_packed
    if not torch.equal(layout.image_lens, native_lens):
        raise RuntimeError('AnyRes native pack lengths disagree with UID probe')
    mid_p, pen_p = pack_dual_depth(pack_fn, hs, per_image, image_sizes, layout,
                                   mid_index=cfg.mid_layer, pen_index=cfg.pen_layer, strategy=vfs)
    cls_scores = None
    if mode == 'cls_attention':
        cls_source = cls_scores_from_clip_qk(llava_model.vision_tower, hs, pen_index=cfg.pen_layer)
        if oracle:
            att = getattr(out, 'attentions', None)
            if att is None or att[cfg.pen_layer] is None:
                raise RuntimeError('CLS_EAGER_ORACLE_MISSING: use eager ViT attention in oracle mode')
            eager = att[cfg.pen_layer].float()[:, :, 0, 1:].mean(dim=1)
            if not torch.allclose(cls_source, eager, atol=2e-3, rtol=2e-3):
                raise RuntimeError('CLS_QK_VS_EAGER: manual input-Q/K disagrees with true ViT attention')
        cls_scores = _packed_cls_saliency(pack_fn, cls_source, layout, native_n_crops, image_sizes, vfs)
    tok = _tokenizer_for(llava_model)
    question_ids = _question_ids_from_input(input_ids[0], int(llava_model.config.image_token_id), tok)
    question_emb = llava_model.get_input_embeddings()(question_ids.unsqueeze(0))[0]
    context = {
        'model': llava_model, 'native_layout': layout, 'packed_cls_scores': cls_scores,
        'saliency_mode': mode, 'projected_full_vision': native_packed[layout.patch_mask],
        'crop_policy': 'uniform' if arm.get('crop_policy', 'equal_valid_crops') == 'equal_valid_crops'
                        else arm.get('crop_policy'),
    }
    keep_ids, qa_log = run_original_bela(mid_p, pen_p, question_emb, cfg, dual_fn, qa_fn,
                                        question_ids=question_ids, extra_context=context)
    qa_log.update({
        'legacy_dual_called': True, 'legacy_qa_called': bool(cfg.qa_swaps),
        'qa_swaps_cfg': cfg.qa_swaps, 'qa_reason': context.get('qa_reason'),
        'qa_swaps_reported': context.get('qa_swaps_reported'),
        'dual_slots_per_crop': context['dual_slots_per_crop'],
        'dual_slots_effective': context['dual_slots_effective'],
        'dual_slots_global': context['dual_slots_global'],
        'dual_quotas': context['dual_quotas'],
        'qa_crop_counts': context.get('qa_crop_counts'),
        'saliency_mode': mode, 'cls_attention_verified': mode == 'cls_attention',
        'question_n_tokens': int(question_ids.numel()),
        'native_crop_count': native_n_crops,
        'native_uid_count': int(layout.patch_count),
        # User-visible example probe: exact patch provenance, no raw image or text.
        'selected_patch_uid_sample': layout.patch_ids[layout.patch_mask][keep_ids[:16]].tolist(),
    })
    select = structural_keep_mask(layout, keep_ids)
    packed = native_packed[select]
    patch_mask = layout.patch_mask[select]
    n_patches = int(patch_mask.sum())
    if n_patches != cfg.stage1:
        raise RuntimeError(f'BELA Stage1 physical patch count {n_patches} != K0 {cfg.stage1}')
    c1, c2 = map(int, arm['cuts_after_layers'])
    if (c1, c2) != (12, 20) or int(arm['layers']) != 32:
        raise RuntimeError('NeXT7 decoder schedule unsupported: requires 32 layers and L12/L20')
    return NativeStage1Result(packed, patch_mask.detach().cpu(), n_patches,
                              int((~patch_mask).sum()), layout.patch_ids[select][patch_mask].detach().cpu(), layout.patch_count,
                              int(layout.newline_mask.sum()), qa_log,
                              str(arm['arm_id']), cfg.stage1, cfg.cut12, cfg.cut20,
                              (c1, c2), mid_layer=cfg.mid_layer)
