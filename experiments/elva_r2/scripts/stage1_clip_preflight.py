"""Real CLIP capture for ELVA R2 Stage1 V1–V5. Payload stays pen features.

This is the required server CLIP check, not an official four-task ACC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

ROOT = Path("/home/xiangyue/ViTCoP/experiments/elva_r2")
sys.path.insert(0, str(ROOT))
from elva.torch_ops import select_stage1  # noqa: E402

TOWER = "openai/clip-vit-large-patch14-336"
PANEL = Path("/home/xiangyue/ViTCoP/experiments/e014_latency_fair_256/panel/speed_256.jsonl")
OUT = ROOT / "results" / "stage1_clip_preflight.json"


def cls_patch_scores(attn):
    # attn: [B,H,T,T] at one layer. CLS -> patches, mean over heads.
    return attn[:, :, 0, 1:].mean(1)[0].float().clamp_min(0)


def main():
    device = "cuda:0"
    processor = CLIPImageProcessor.from_pretrained(TOWER)
    model = CLIPVisionModel.from_pretrained(TOWER).to(device).eval()
    n_layers = model.config.num_hidden_layers
    # HF: hidden_states[0]=embed; hidden_states[i]=after block i-1.
    l9_index = 9  # after 9 blocks (blocks[0..8])
    pen_index = -2  # completed D-1
    row = json.loads(PANEL.read_text().splitlines()[0])
    image_path = row.get("image") or row.get("path") or row.get("image_path")
    if image_path is None:
        # panel schema may nest
        image_path = next(v for k, v in row.items() if "image" in k.lower() or str(v).endswith((".jpg", ".png")))
    image = Image.open(image_path).convert("RGB")
    pixel = processor(images=image, return_tensors="pt")["pixel_values"].to(device=device, dtype=model.dtype)
    with torch.no_grad():
        out = model(pixel, output_hidden_states=True, output_attentions=True)
    pen = out.hidden_states[pen_index][0, 1:].float()
    early = out.hidden_states[l9_index][0, 1:].float()
    pen_scores = cls_patch_scores(out.attentions[pen_index])
    early_scores = cls_patch_scores(out.attentions[l9_index - 1])
    ids = torch.arange(pen.shape[0], device=device)
    k = 90
    specs = {
        "V1_pen": dict(guide="pen"),
        "V2_l9": dict(guide="l9", early=early, early_scores=early_scores, early_ids=ids),
        "V3_dual": dict(guide="dual", beta=0.5, early=early, early_scores=early_scores, early_ids=ids),
        "V4_pen_cover_proxy": dict(guide="pen"),  # COVER bind is separate; top-k proxy until native COVER is hooked
        "V5_dual_cover_proxy": dict(guide="dual", beta=0.5, early=early, early_scores=early_scores, early_ids=ids),
    }
    keeps = {}
    payloads_ok = {}
    for name, kw in specs.items():
        sel = select_stage1(pen, pen_scores, ids, k, debug=True, use_cover=False, **kw)
        keeps[name] = set(sel["keep"].detach().cpu().tolist())
        payloads_ok[name] = bool(torch.equal(sel["payload"], pen.index_select(0, sel["keep"])))
    base = keeps["V1_pen"]

    def overlap(a, b):
        return len(a & b) / k

    report = {
        "tower": TOWER,
        "num_hidden_layers": n_layers,
        "l9_hidden_index": l9_index,
        "pen_hidden_index": pen_index if pen_index >= 0 else n_layers + pen_index + 1,
        "tokens": int(pen.shape[0]),
        "k": k,
        "image": str(image_path),
        "payload_is_pen": payloads_ok,
        "overlap_vs_V1": {n: overlap(s, base) for n, s in keeps.items()},
        "note": "V4/V5 here are score geometry only; original COVER callback is not yet bound.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
