#!/usr/bin/env python3
"""Prove TF5.20 generate() path actually runs Dual COVER+Stage2 and A0≠A4."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PKG = Path("/home/xiangyue/ViTCoP/experiments/bela_next_7b13b_v100_20261101")
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "BELA_NEXT_7B13B_V100_CURSOR_20261101/src"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("BELA_SMOKE_GPU", "4")
os.environ["BELA_NEXT_V2_ENABLE"] = "1"

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from bela_next_forward.apply_llava_next import apply_patch, summarize_trace

OUT = PKG / "logs" / "diag_generate_path"
OUT.mkdir(parents=True, exist_ok=True)
_WHICH = os.environ.get("BELA_NEXT_MODEL", "7b").strip().lower()
if _WHICH in ("13", "13b", "next13"):
    MODEL = "llava-hf/llava-v1.6-vicuna-13b-hf"
    ARM_A0, ARM_A4 = "NEXT13_T160_A0_PAPER", "NEXT13_T160_A4_STRONG_LATE"
    EXP_A0, EXP_A4 = (320, 80, 40), (256, 128, 64)
else:
    MODEL = "llava-hf/llava-v1.6-vicuna-7b-hf"
    ARM_A0, ARM_A4 = "NEXT7_T160_A0_PAPER", "NEXT7_T160_A4_STRONG_LATE"
    EXP_A0, EXP_A4 = (320, 91, 46), (256, 160, 64)


def make_image(seed=0):
    img = Image.new("RGB", (672, 672), color=(40, 120, 200))
    for y in range(0, 672, 16):
        for x in range(0, 672, 16):
            img.putpixel((x, y), (((x + seed) * 3) % 255, ((y + seed) * 5) % 255, 90))
    return img


def run_arm(model, processor, arm: str, max_new: int = 16):
    os.environ["BELA_NEXT_ARM"] = arm
    for attr in ("_bela_stage1", "_bela_stage2", "_bela_patch_pos", "_bela_last_prefill_seq"):
        if hasattr(model, attr):
            setattr(model, attr, None)
    model._bela_prefill_hits = 0

    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: <image>\nWhat is the dominant color? Answer with one word.\nASSISTANT:"
    )
    # Mimic lmms chat-template style + forced pinpoints
    inputs = processor(images=make_image(7), text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    n_img_tokens = int((inputs["input_ids"] == model.config.image_token_id).sum().item())
    crop_n = int(inputs["pixel_values"].shape[1])

    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    tr = summarize_trace(model)
    s1 = tr.get("stage1") or {}
    s2 = tr.get("stage2") or {}
    events = s2.get("events") or []
    got = (
        s1.get("packed_patches"),
        events[0]["k_after"] if len(events) > 0 else None,
        events[1]["k_after"] if len(events) > 1 else None,
    )
    return {
        "arm": arm,
        "crop_n": crop_n,
        "n_img_tokens_in": n_img_tokens,
        "prefill_via": s1.get("prefill_via"),
        "prefill_hits": tr.get("prefill_hits"),
        "last_prefill_seq": tr.get("last_prefill_seq"),
        "got_k": got,
        "gen_ids_hash": hashlib.sha256(out[0].cpu().numpy().tobytes()).hexdigest()[:16],
        "text_tail": text[-80:],
        "stage1_arm": s1.get("arm"),
    }


def main():
    print("load", MODEL, flush=True)
    processor = LlavaNextProcessor.from_pretrained(MODEL)
    processor.image_processor.image_grid_pinpoints = [[672, 672]]
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda:0"
    )
    model.eval()
    apply_patch(model)

    a0 = run_arm(model, processor, ARM_A0)
    a4 = run_arm(model, processor, ARM_A4)
    dense_env = os.environ.get("BELA_NEXT_V2_ENABLE")
    # dense generate for contrast
    os.environ["BELA_NEXT_V2_ENABLE"] = "0"
    if "BELA_NEXT_ARM" in os.environ:
        del os.environ["BELA_NEXT_ARM"]
    inputs = processor(
        images=make_image(7),
        text=(
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "USER: <image>\nWhat is the dominant color? Answer with one word.\nASSISTANT:"
        ),
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        dout = model.generate(**inputs, do_sample=False, max_new_tokens=16)
    dense_hash = hashlib.sha256(dout[0].cpu().numpy().tobytes()).hexdigest()[:16]
    os.environ["BELA_NEXT_V2_ENABLE"] = dense_env or "1"

    report = {
        "a0": a0,
        "a4": a4,
        "dense_hash": dense_hash,
        "a0_eq_a4": a0["gen_ids_hash"] == a4["gen_ids_hash"],
        "a0_eq_dense": a0["gen_ids_hash"] == dense_hash,
        "a4_eq_dense": a4["gen_ids_hash"] == dense_hash,
        "expect_a0": EXP_A0,
        "expect_a4": EXP_A4,
        "model": MODEL,
    }
    ok = (
        a0["got_k"] == EXP_A0
        and a4["got_k"] == EXP_A4
        and a0["prefill_via"] in ("mm_encoder", "pixel_values")
        and a4["prefill_via"] in ("mm_encoder", "pixel_values")
        and a0["gen_ids_hash"] != a4["gen_ids_hash"]
        and a0["prefill_hits"]
        and a4["prefill_hits"]
    )
    report["PASS"] = bool(ok)
    (OUT / "DIAG_REPORT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("DIAG_PASS", ok, flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
