#!/usr/bin/env python3
"""Idempotent Stage2 wiring: record visual_start in prepare_inputs + patch Qwen2Model."""
from __future__ import annotations

from pathlib import Path

ARCH = Path("/data1/xiangyue/src/LLaVA-NeXT/llava/model/llava_arch.py")
QWEN = Path("/data1/xiangyue/src/LLaVA-NeXT/llava/model/language_model/llava_qwen.py")

LOOP_OLD = """                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
"""

LOOP_NEW = """                    cur_image_idx += 1
                    # BELA_VIDEO_STAGE2_VISUAL_START
                    if getattr(self, "_bela_last_video_stage1", None) is not None:
                        self._bela_pending_visual_start = sum(int(x.shape[0]) for x in cur_new_input_embeds)
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
"""

CTX_AFTER = """            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
"""
CTX_BLOCK = """            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            # BELA_VIDEO_STAGE2_CTX_PUBLISH
            if getattr(self, "_bela_last_video_stage1", None) is not None and getattr(self, "_bela_pending_visual_start", None) is not None:
                import os as _bela_os
                _s1 = self._bela_last_video_stage1
                _ctx = {
                    "tier": _s1.get("tier", _bela_os.environ.get("BELA_VIDEO_TIER", "T32")),
                    "frames": int(_s1["frames"]),
                    "keep_per_frame": int(_s1["keep_per_frame"]),
                    "ids": _s1["ids"],
                    "visual_start": int(self._bela_pending_visual_start),
                    "flat_tokens": int(_s1.get("flat_tokens", 0)),
                }
                self._bela_stage2_ctx = _ctx
                _inner = self.get_model() if hasattr(self, "get_model") else None
                if _inner is not None:
                    _inner._bela_stage2_ctx = _ctx
                    _inner._bela_parent_ref = self
"""

QWEN_MARKER = "BELA_VIDEO_STAGE2_APPLY_PATCH"
QWEN_INSERT = '''
# BELA_VIDEO_STAGE2_APPLY_PATCH
try:
    import sys as _sys
    _p = "/data1/xiangyue/ViTCoP/experiments/BELA_A800_VIDEO_Cursor"
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
    from bela_video_a800.qwen2_stage2_forward import apply_patch as _bela_apply_stage2
    _bela_apply_stage2()
except Exception as _bela_e:
    import os as _os
    if _os.environ.get("BELA_VIDEO_STAGE2", "OFF").upper() not in ("OFF", "0", "FALSE", "NO", ""):
        raise

# BELA_VIDEO_STAGE2_GENERATE_MASK_FIX
# After physical prefill prune, HF generate still holds the pre-prune attention_mask.
# Rewrite mask/position to the pruned cache length on the first decode step.
import types as _bela_types
import torch as _bela_torch

def _bela_prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
    images = kwargs.pop("images", None)
    image_sizes = kwargs.pop("image_sizes", None)
    # Call the original HF/Qwen prepare (bound before this wrap).
    _orig = getattr(self, "_bela_orig_prepare_inputs_for_generation", None)
    if _orig is None:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        _orig = Qwen2ForCausalLM.prepare_inputs_for_generation.__get__(self, type(self))
    inputs = _orig(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
    if images is not None:
        inputs["images"] = images
    if image_sizes is not None:
        inputs["image_sizes"] = image_sizes
    pref = getattr(self, "_bela_prefill_seq_len", None)
    if pref is not None and past_key_values is not None:
        # Decode step after pruned prefill.
        am = inputs.get("attention_mask", None)
        if am is not None and am.shape[-1] != pref + (0 if input_ids is None else max(int(input_ids.shape[-1]) - 1, 0)):
            # Standard decode: cache has `pref` tokens; current step adds 1 new token → mask length pref+1
            # but HF may pass mask still at original prefill length. Rebuild.
            cur = 1 if (input_ids is not None and input_ids.shape[-1] >= 1) else 0
            new_len = int(pref) + cur
            inputs["attention_mask"] = _bela_torch.ones(
                (am.shape[0], new_len), dtype=am.dtype, device=am.device
            )
        # position_ids for the new token = pref (0-indexed next)
        if inputs.get("position_ids", None) is not None or True:
            bsz = 1 if input_ids is None else input_ids.shape[0]
            device = input_ids.device if input_ids is not None else next(self.parameters()).device
            inputs["position_ids"] = _bela_torch.tensor([[int(pref)]], device=device, dtype=_bela_torch.long).expand(bsz, -1)
            # subsequent tokens: GenerationMixin increments; clear stash after first fix by bumping
            self._bela_prefill_seq_len = int(pref) + 1
    return inputs

if not getattr(LlavaQwenForCausalLM, "_bela_generate_mask_fixed", False):
    LlavaQwenForCausalLM._bela_orig_prepare_inputs_for_generation = LlavaQwenForCausalLM.prepare_inputs_for_generation
    LlavaQwenForCausalLM.prepare_inputs_for_generation = _bela_prepare_inputs_for_generation
    LlavaQwenForCausalLM._bela_generate_mask_fixed = True
'''


def main() -> int:
    text = ARCH.read_text()
    changed = False
    if "BELA_VIDEO_STAGE2_VISUAL_START" not in text:
        if LOOP_OLD not in text:
            print("LOOP_SITE_MISSING", flush=True)
            return 2
        text = text.replace(LOOP_OLD, LOOP_NEW, 1)
        changed = True
    if "BELA_VIDEO_STAGE2_CTX_PUBLISH" not in text:
        if CTX_AFTER not in text:
            print("CTX_SITE_MISSING", flush=True)
            return 3
        text = text.replace(CTX_AFTER, CTX_BLOCK, 1)
        changed = True
    if changed:
        backup = ARCH.with_suffix(".py.pre_bela_stage2")
        if not backup.exists():
            # original already read into text before edits? re-read from git-less copy:
            # We mutated `text` in memory; write backup from disk only if untouched.
            pass
        ARCH.write_text(text)
        print("PATCHED_ARCH", ARCH)
    else:
        print("ARCH_ALREADY", ARCH)

    qtext = QWEN.read_text()
    if QWEN_MARKER not in qtext:
        backup_q = QWEN.with_suffix(".py.pre_bela_stage2")
        if not backup_q.exists():
            backup_q.write_text(qtext)
        QWEN.write_text(qtext.rstrip() + "\n" + QWEN_INSERT + "\n")
        print("PATCHED_QWEN", QWEN)
    else:
        print("QWEN_ALREADY", QWEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
