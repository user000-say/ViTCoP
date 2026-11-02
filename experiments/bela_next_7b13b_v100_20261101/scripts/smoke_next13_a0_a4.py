#!/usr/bin/env python3
"""Dense parity + A0–A4 single-image smoke for NeXT-13B (cuts L15/L30)."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

PKG = Path("/home/xiangyue/ViTCoP/experiments/bela_next_7b13b_v100_20261101")
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "BELA_NEXT_7B13B_V100_CURSOR_20261101/src"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("BELA_SMOKE_GPU", "4")

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from bela_next_forward.apply_llava_next import apply_patch, summarize_trace

OUT = PKG / "logs" / "smoke_next13_a0_a4"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "llava-hf/llava-v1.6-vicuna-13b-hf"
ARMS = [
    "NEXT13_T160_A0_PAPER",
    "NEXT13_T160_A1_PAPER_VAR",
    "NEXT13_T160_A2_LATE",
    "NEXT13_T160_A3_LESS_S1",
    "NEXT13_T160_A4_STRONG_LATE",
]
EXPECT = {
    "NEXT13_T160_A0_PAPER": (320, 80, 40),
    "NEXT13_T160_A1_PAPER_VAR": (320, 84, 34),
    "NEXT13_T160_A2_LATE": (320, 72, 52),
    "NEXT13_T160_A3_LESS_S1": (288, 112, 40),
    "NEXT13_T160_A4_STRONG_LATE": (256, 128, 64),
}


def make_image():
    img = Image.new("RGB", (672, 672), color=(40, 120, 200))
    for y in range(0, 672, 32):
        for x in range(0, 672, 32):
            img.putpixel((x, y), ((x * 3) % 255, (y * 5) % 255, 90))
    return img


def _clear(model):
    for attr in ("_bela_stage1", "_bela_stage2", "_bela_patch_pos", "_bela_keep_flat", "_bela_text_start"):
        if hasattr(model, attr):
            setattr(model, attr, None)


def run_prefill(model, processor, *, enable: bool, arm: str | None):
    os.environ["BELA_NEXT_V2_ENABLE"] = "1" if enable else "0"
    if arm:
        os.environ["BELA_NEXT_ARM"] = arm
    elif "BELA_NEXT_ARM" in os.environ:
        del os.environ["BELA_NEXT_ARM"]
    image = make_image()
    prompt = "USER: <image>\nWhat primary color dominates this image? Answer with one word.\nASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    _clear(model)
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
    logits = out.logits
    finite = bool(torch.isfinite(logits).all().item())
    trace = summarize_trace(model)
    pv = inputs.get("pixel_values")
    return {
        "enable": enable,
        "arm": arm,
        "crop_n": int(pv.shape[1]) if pv is not None and pv.dim() == 5 else None,
        "next_id": int(logits[0, -1].argmax().item()),
        "finite": finite,
        "trace": trace,
        "hidden_len": int(logits.shape[1]),
        "seq_logits": int(logits.shape[1]),
    }


def run_generate(model, processor, *, enable: bool, arm: str | None, max_new: int = 4):
    os.environ["BELA_NEXT_V2_ENABLE"] = "1" if enable else "0"
    if arm:
        os.environ["BELA_NEXT_ARM"] = arm
    elif "BELA_NEXT_ARM" in os.environ:
        del os.environ["BELA_NEXT_ARM"]
    image = make_image()
    prompt = "USER: <image>\nWhat primary color dominates this image? Answer with one word.\nASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    _clear(model)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return {"gen_ids": out[0].tolist(), "text_tail": text[-160:], "trace": summarize_trace(model)}


def main():
    print("loading", MODEL, "visible", os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
    processor = LlavaNextProcessor.from_pretrained(MODEL)
    if hasattr(processor, "image_processor"):
        processor.image_processor.image_grid_pinpoints = [[672, 672]]
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda:0"
    )
    model.eval()
    n_layers = int(model.config.text_config.num_hidden_layers)
    if n_layers != 40:
        raise RuntimeError(f"expected 40 decoder layers for NeXT-13B, got {n_layers}")
    apply_patch(model)
    print("patched layers", n_layers, flush=True)

    report = {"dense_parity": None, "arms": [], "failures": [], "model": MODEL, "layers": n_layers}

    d1 = run_prefill(model, processor, enable=False, arm=None)
    d2 = run_prefill(model, processor, enable=False, arm=None)
    parity = d1["next_id"] == d2["next_id"] and d1["seq_logits"] == d2["seq_logits"]
    try:
        g1 = run_generate(model, processor, enable=False, arm=None)
        g2 = run_generate(model, processor, enable=False, arm=None)
        gen_parity = g1["gen_ids"] == g2["gen_ids"]
    except Exception as e:
        g1, gen_parity = None, False
        report["failures"].append(f"dense_generate: {e!r}")
        traceback.print_exc()
    report["dense_parity"] = {
        "pass": parity and gen_parity,
        "prefill_parity": parity,
        "gen_parity": gen_parity,
        "crop_n": d1["crop_n"],
        "text_tail": None if g1 is None else g1["text_tail"],
    }
    print("DENSE_PARITY", report["dense_parity"]["pass"], "crops", d1["crop_n"], flush=True)

    for arm in ARMS:
        try:
            row = run_prefill(model, processor, enable=True, arm=arm)
            s1 = (row["trace"] or {}).get("stage1") or {}
            s2 = (row["trace"] or {}).get("stage2") or {}
            events = s2.get("events") or []
            # expect cuts after 15 and 30
            cut_layers = [e.get("layer_1based") for e in events]
            got = (
                s1.get("packed_patches"),
                events[0]["k_after"] if len(events) > 0 else None,
                events[1]["k_after"] if len(events) > 1 else s2.get("final_patches"),
            )
            exp = EXPECT[arm]
            pack_mode = s1.get("pack_mode")
            n_nl = s1.get("packed_newlines")
            pack_ok = pack_mode == "spatial_unpad_newline" and isinstance(n_nl, int) and n_nl > 0
            ok = got == exp and row["finite"] and cut_layers == [15, 30] and pack_ok
            shortened = row["hidden_len"] < 2000
            gen_ok, gen_tail = False, None
            try:
                g = run_generate(model, processor, enable=True, arm=arm)
                gen_ok, gen_tail = True, g["text_tail"]
            except Exception as ge:
                report["failures"].append(f"{arm} generate: {ge!r}")
                traceback.print_exc()
            rec = {
                "arm": arm,
                "expected": exp,
                "got": got,
                "cut_layers": cut_layers,
                "pass": bool(ok and shortened and gen_ok),
                "finite": row["finite"],
                "hidden_len": row["hidden_len"],
                "pack_mode": pack_mode,
                "packed_newlines": n_nl,
                "pack_ok": pack_ok,
                "gen_ok": gen_ok,
                "gen_tail": gen_tail,
                "stage1": s1,
                "stage2": s2,
            }
            report["arms"].append(rec)
            print("ARM", arm, "PASS" if rec["pass"] else "FAIL", "got", got, "cuts", cut_layers, flush=True)
            if not rec["pass"]:
                report["failures"].append(f"{arm}: got {got} cuts {cut_layers}")
        except Exception as e:
            report["arms"].append({"arm": arm, "pass": False, "error": repr(e)})
            report["failures"].append(f"{arm}: {e!r}")
            traceback.print_exc()
            print("ARM", arm, "ERROR", repr(e), flush=True)

    all_ok = bool(report["dense_parity"]["pass"]) and all(a.get("pass") for a in report["arms"])
    (OUT / "SMOKE_REPORT.json").write_text(json.dumps(report, indent=2, default=str))
    (OUT / "SMOKE_VERDICT.md").write_text(f"# SMOKE_VERDICT NeXT-13B\n\nALL_PASS={all_ok}\n")
    print("WROTE", OUT, "ALL_PASS", all_ok, flush=True)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
