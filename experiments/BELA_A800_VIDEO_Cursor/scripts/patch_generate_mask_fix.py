#!/usr/bin/env python3
"""Ensure LlavaQwen prepare_inputs_for_generation rewrites mask after Stage2 prune."""
from pathlib import Path

QWEN = Path("/data1/xiangyue/src/LLaVA-NeXT/llava/model/language_model/llava_qwen.py")
MARKER = "BELA_VIDEO_STAGE2_GENERATE_MASK_FIX"

BLOCK = '''
# BELA_VIDEO_STAGE2_GENERATE_MASK_FIX
import torch as _bela_torch

def _bela_prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
    images = kwargs.pop("images", None)
    image_sizes = kwargs.pop("image_sizes", None)
    _orig = getattr(self, "_bela_orig_prepare_inputs_for_generation", None)
    if _orig is not None:
        inputs = _orig(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
    else:
        inputs = Qwen2ForCausalLM.prepare_inputs_for_generation(
            self, input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
    if images is not None:
        inputs["images"] = images
    if image_sizes is not None:
        inputs["image_sizes"] = image_sizes
    pref = getattr(self, "_bela_prefill_seq_len", None)
    if pref is not None and past_key_values is not None and input_ids is not None:
        bsz = input_ids.shape[0]
        cur = int(input_ids.shape[-1])
        # After prune, HF may still carry the long pre-prune mask. Rebuild to cache+new.
        new_len = int(pref) + cur
        device = input_ids.device
        inputs["attention_mask"] = _bela_torch.ones((bsz, new_len), dtype=_bela_torch.long, device=device)
        # position of the newest token
        inputs["position_ids"] = _bela_torch.arange(
            int(pref), int(pref) + cur, device=device, dtype=_bela_torch.long
        ).unsqueeze(0).expand(bsz, -1)
        self._bela_prefill_seq_len = int(pref) + cur
    return inputs

if not getattr(LlavaQwenForCausalLM, "_bela_generate_mask_fixed", False):
    LlavaQwenForCausalLM._bela_orig_prepare_inputs_for_generation = (
        LlavaQwenForCausalLM.prepare_inputs_for_generation
    )
    LlavaQwenForCausalLM.prepare_inputs_for_generation = _bela_prepare_inputs_for_generation
    LlavaQwenForCausalLM._bela_generate_mask_fixed = True
'''


def main() -> int:
    text = QWEN.read_text()
    if MARKER in text:
        print("ALREADY", QWEN)
        return 0
    QWEN.write_text(text.rstrip() + "\n" + BLOCK + "\n")
    print("PATCHED", QWEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
