"""Native NeXT Stage1: stock pack + Dual COVER (L9+pen) + global QA B8.

Follows BELA_NEXT_NATIVE_FIX_KIT contracts on Transformers 5.20 LlavaNext:
vision tower once → projector crops → pack_image_features (spatial_unpad+newline)
→ Dual COVER crop-grouped in packed patch space → QA B8 → structural newline keep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from bela_next_fix.legacy_bridge import ARMS, BelaConfig, run_original_bela
from bela_next_fix.native_layout import (
    map_native_packing,
    pack_dual_depth,
    structural_keep_mask,
    _normalize_packer_result,
)


@dataclass
class NativeStage1Result:
    packed_features: torch.Tensor  # [L, D] after Stage1 keep (patches+kept newlines)
    is_patch: torch.Tensor  # [L] bool
    n_patches: int
    n_newlines: int
    native_patch_count: int
    native_newlines: int
    qa_log: dict
    arm_id: str
    k0: int
    k1: int
    k2: int
    cut_layers: Tuple[int, int]
    pack_mode: str = "spatial_unpad_newline"
    mid_layer: int = 9


def _arm_to_cfg(arm: dict) -> BelaConfig:
    k0, k1, k2 = [int(x) for x in arm["stage_k"]]
    c1, c2 = [int(x) for x in arm["cuts_after_layers"]]
    return BelaConfig(
        stage1=k0,
        cut12=k1,
        cut20=k2,
        slots=32,
        qa_swaps=int(arm.get("qa_swaps", 8)),
        donor=16,
        drop=12,
        seed=17,
        # Production Dual L9: capture_outputs puts block-9 output at hs[9]
        # (hs[0]=layer0 input). Do NOT use hs[10] — that was in the ~82 R4 run.
        mid_layer=9,
        pen_layer=-2,
        require_true_qa=True,
    )


_TOK_CACHE = {}


def _tokenizer_for(llava_model):
    """Lazy tokenizer for question-only span (Vicuna USER/ASSISTANT strip)."""
    for obj in (llava_model, getattr(llava_model, "config", None)):
        if obj is None:
            continue
        tok = getattr(obj, "tokenizer", None) or getattr(obj, "_tokenizer", None)
        if tok is not None:
            return tok
    name = getattr(getattr(llava_model, "config", None), "_name_or_path", None) or getattr(
        getattr(llava_model, "config", None), "name_or_path", None
    )
    if not name:
        name = "llava-hf/llava-v1.6-vicuna-7b-hf"
    if name not in _TOK_CACHE:
        from transformers import AutoTokenizer

        _TOK_CACHE[name] = AutoTokenizer.from_pretrained(name, use_fast=False)
    return _TOK_CACHE[name]


def _find_subseq(hay: torch.Tensor, needle: list) -> int:
    """Return start index of needle in hay, or -1."""
    if not needle or hay.numel() < len(needle):
        return -1
    n = len(needle)
    for i in range(int(hay.numel()) - n + 1):
        if all(int(hay[i + j].item()) == int(needle[j]) for j in range(n)):
            return i
    return -1


def _question_ids_from_input(
    input_ids: torch.Tensor, image_token_id: int, tokenizer=None
) -> torch.Tensor:
    """Return QUESTION-ONLY ids by slicing the original token span.

    Drop image placeholders and Vicuna USER:/ASSISTANT: wrappers without
    decode→re-encode (SentencePiece re-tokenization ≠ original ids).
    """
    assert input_ids.ndim == 1
    mask = input_ids != image_token_id
    ids = input_ids[mask]
    if ids.numel() and int(ids[0].item()) == 0:
        ids = ids[1:]
    if tokenizer is None or ids.numel() == 0:
        return ids
    start = 0
    for mark in ("USER:", "User:"):
        mid = tokenizer(mark, add_special_tokens=False).input_ids
        at = _find_subseq(ids, mid)
        if at >= 0:
            start = at + len(mid)
            break
    end = int(ids.numel())
    for mark in ("ASSISTANT:", "Assistant:"):
        mid = tokenizer(mark, add_special_tokens=False).input_ids
        at = _find_subseq(ids[start:], mid)
        if at >= 0:
            end = start + at
            break
    out = ids[start:end]
    while out.numel():
        piece = tokenizer.decode([int(out[0].item())], skip_special_tokens=False)
        if piece.strip() == "":
            out = out[1:]
            continue
        break
    return out if out.numel() else ids


@torch.no_grad()
def run_native_stage1(
    llava_model,
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor,
    input_ids: torch.Tensor,
    arm: dict,
    *,
    dual_fn,
    qa_fn,
    vision_feature_layer=None,
    vision_feature_select_strategy=None,
) -> NativeStage1Result:
    cfg = _arm_to_cfg(arm)
    cfg.check()

    if pixel_values.dim() == 5:
        assert pixel_values.shape[0] == 1
        crops = pixel_values[0]
    else:
        crops = pixel_values

    vfl = (
        vision_feature_layer
        if vision_feature_layer is not None
        else llava_model.config.vision_feature_layer
    )
    vfs = (
        vision_feature_select_strategy
        or llava_model.config.vision_feature_select_strategy
    )

    out = llava_model.vision_tower(crops, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    if isinstance(vfl, int):
        selected = hs[vfl]
    else:
        selected = torch.cat([hs[i] for i in vfl], dim=-1)
    if vfs == "default":
        selected = selected[:, 1:]
    projected = llava_model.multi_modal_projector(selected)  # [C,576,D]
    per_image = [projected]

    pack_fn = llava_model.pack_image_features
    native_packed, native_lens = _normalize_packer_result(
        pack_fn(
            per_image,
            image_sizes,
            vision_feature_select_strategy=vfs,
            image_newline=llava_model.image_newline,
        )
    )
    layout = map_native_packing(pack_fn, per_image, image_sizes, strategy=vfs)
    if not torch.equal(layout.image_lens.to(native_lens.device), native_lens):
        raise RuntimeError("native layout lens != packed lens")

    mid_p, pen_p = pack_dual_depth(
        pack_fn,
        hs,
        per_image,
        image_sizes,
        layout,
        mid_index=cfg.mid_layer,
        pen_index=cfg.pen_layer,
        strategy=vfs,
    )
    # free tower states
    try:
        out.hidden_states = None
    except Exception:
        pass
    del hs, selected, out

    ids_1d = input_ids[0] if input_ids.ndim == 2 else input_ids
    image_token_id = int(llava_model.config.image_token_id)
    tok = _tokenizer_for(llava_model)
    q_ids = _question_ids_from_input(ids_1d, image_token_id, tokenizer=tok)
    # question embeddings via LLM embed (same space QA expects after projector on vision)
    q_emb = llava_model.get_input_embeddings()(q_ids.unsqueeze(0))[0]

    context = {
        "model": llava_model,
        "native_layout": layout,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
    }
    keep_ids, qa_log = run_original_bela(
        mid_p,
        pen_p,
        q_emb,
        cfg,
        dual_fn,
        qa_fn,
        question_ids=q_ids,
        extra_context=context,
    )
    qa_log = dict(qa_log)
    qa_log.update(
        {
            "legacy_dual_called": True,
            "legacy_qa_called": bool(cfg.qa_swaps > 0),
            "mid_layer": cfg.mid_layer,
            "qa_swaps_cfg": cfg.qa_swaps,
            "dual_slots_per_crop": context.get("dual_slots_per_crop"),
            "dual_quotas": context.get("dual_quotas"),
            "dual_slots_global": context.get("dual_slots_global"),
            "qa_reason": context.get("qa_reason"),
            "qa_swaps_reported": context.get("qa_swaps_reported"),
            "question_n_tokens": int(q_ids.numel()),
        }
    )

    local_keep = structural_keep_mask(layout, keep_ids)
    packed = native_packed[local_keep]
    is_patch = layout.patch_mask[local_keep]
    n_patches = int(is_patch.sum().item())
    n_nl = int((~is_patch).sum().item())
    if n_patches != int(cfg.stage1):
        raise RuntimeError(f"Stage1 patches {n_patches} != K0 {cfg.stage1}")

    c1, c2 = [int(x) for x in arm["cuts_after_layers"]]
    return NativeStage1Result(
        packed_features=packed,
        is_patch=is_patch.detach().cpu(),
        n_patches=n_patches,
        n_newlines=n_nl,
        native_patch_count=int(layout.patch_count),
        native_newlines=int(layout.newline_mask.sum().item()),
        qa_log=qa_log,
        arm_id=str(arm["arm_id"]),
        k0=int(cfg.stage1),
        k1=int(cfg.cut12),
        k2=int(cfg.cut20),
        cut_layers=(c1, c2),
        mid_layer=cfg.mid_layer,
    )
