#!/usr/bin/env python3
"""CPU/GPU: keep-all pack must match stock LlavaNext pack_image_features."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PKG = Path("/home/xiangyue/ViTCoP/experiments/bela_next_7b13b_v100_20261101")
sys.path.insert(0, str(PKG))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("BELA_SMOKE_GPU", "4"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from bela_next_forward.next_pack import pack_spatial_unpad_selected


def main():
    model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_id)
    processor.image_processor.image_grid_pinpoints = [[672, 672]]
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda:0"
    )
    model.eval()
    llava = model.model

    # square + non-square
    cases = [(672, 672), (480, 640), (1024, 768)]
    report = []
    for hw in cases:
        img = Image.new("RGB", (hw[1], hw[0]), color=(30, 90, 160))  # PIL size=(W,H)
        prompt = "USER: <image>\nDescribe briefly.\nASSISTANT:"
        inputs = processor(images=img, text=prompt, return_tensors="pt")
        pv = inputs["pixel_values"].to(model.device)
        sizes = inputs["image_sizes"].to(model.device)
        crops = pv[0]  # [C,3,H,W]
        with torch.inference_mode():
            out = llava.vision_tower(crops, output_hidden_states=True, return_dict=True)
            vfl = llava.config.vision_feature_layer
            sel = out.hidden_states[vfl][:, 1:, :]
            projected = llava.multi_modal_projector(sel)
            stock_list, stock_lens = llava.pack_image_features(
                [projected],
                sizes,
                vision_feature_select_strategy=llava.config.vision_feature_select_strategy,
                image_newline=llava.image_newline,
            )
            stock = stock_list[0]
            keep = [torch.arange(576, device=projected.device) for _ in range(projected.shape[0])]
            ours = pack_spatial_unpad_selected(
                projected,
                sizes[0].tolist(),
                keep,
                llava.image_newline,
                image_grid_pinpoints=llava.config.image_grid_pinpoints,
                vision_image_size=int(llava.config.vision_config.image_size),
                patch_size=int(llava.config.vision_config.patch_size),
            )
        same_len = int(ours.features.shape[0]) == int(stock.shape[0])
        if same_len:
            max_abs = float((ours.features.float() - stock.float()).abs().max().item())
        else:
            max_abs = float("inf")
        ok = same_len and max_abs < 1e-3
        row = {
            "hw": hw,
            "crops": int(projected.shape[0]),
            "stock_len": int(stock.shape[0]),
            "ours_len": int(ours.features.shape[0]),
            "n_patches": ours.n_patches,
            "n_newlines": ours.n_newlines,
            "max_abs": max_abs,
            "pass": ok,
        }
        report.append(row)
        print(row, flush=True)
        if not ok:
            raise SystemExit(f"PACK_PARITY_FAIL {row}")

    # subset: newlines preserved, patch count == keep size on square
    img = Image.new("RGB", (672, 672), color=(10, 20, 30))
    inputs = processor(images=img, text="USER: <image>\nQ?\nASSISTANT:", return_tensors="pt")
    pv = inputs["pixel_values"].to(model.device)
    sizes = inputs["image_sizes"].to(model.device)
    crops = pv[0]
    with torch.inference_mode():
        out = llava.vision_tower(crops, output_hidden_states=True, return_dict=True)
        sel = out.hidden_states[llava.config.vision_feature_layer][:, 1:, :]
        projected = llava.multi_modal_projector(sel)
        # 64 per crop
        keep = [torch.arange(0, 576, 9, device=projected.device)[:64] for _ in range(5)]
        assert all(k.numel() == 64 for k in keep)
        ours = pack_spatial_unpad_selected(
            projected,
            sizes[0].tolist(),
            keep,
            llava.image_newline,
            image_grid_pinpoints=llava.config.image_grid_pinpoints,
            vision_image_size=int(llava.config.vision_config.image_size),
            patch_size=int(llava.config.vision_config.patch_size),
        )
    assert ours.n_patches == 320, ours.n_patches
    assert ours.n_newlines == 48, ours.n_newlines  # 48 rows after unpad on square 2x2
    assert int(ours.is_patch.sum()) == 320
    assert int((~ours.is_patch).sum()) == 48
    print("SUBSET_OK", {"len": int(ours.features.shape[0]), "patches": 320, "nl": 48}, flush=True)
    print("ALL_PACK_PARITY_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
