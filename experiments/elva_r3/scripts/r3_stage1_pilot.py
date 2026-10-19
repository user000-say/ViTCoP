"""Real-CLIP Stage1 core pilot on one physical GPU (visible as cuda:0).

Runs the ten stage1_core configs on a disjoint image shard. This is selection
and payload audit, not official four-task ACC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/elva_r3"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(REPO / "LLaVA"))

from elva.clip_stage1 import CLIPStage1  # noqa: E402
from elva.cover_binding import NativeCover, file_sha  # noqa: E402
from elva.low_budget_cover import POLICY, callback as pcov_callback  # noqa: E402
from native_cover import legacy_callback  # noqa: E402

PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
TOWER = "openai/clip-vit-large-patch14-336"


def spec_for(family):
    if family == "legacy_native":
        files = {
            str((PKG / "native_cover.py").resolve()): file_sha(PKG / "native_cover.py"),
            str((REPO / "LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py").resolve()):
                file_sha(REPO / "LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py"),
        }
        return dict(family=family, entrypoint="native_cover:legacy_callback",
                    supported_k=[172], replacement_slots={"172": 32}, source_files=files,
                    callback=legacy_callback)
    if family == "protected_coverage":
        path = PKG / "elva/low_budget_cover.py"
        files = {str(path.resolve()): file_sha(path)}
        return dict(family=family, entrypoint="elva.low_budget_cover:callback",
                    supported_k=[60], replacement_slots={"60": 12}, allowed_slots=[6, 12, 18],
                    coverage_policy=POLICY, source_files=files, callback=pcov_callback)
    raise RuntimeError(family)


def covers():
    out = {}
    for family in ("legacy_native", "protected_coverage"):
        spec = spec_for(family)
        out[family] = NativeCover(spec.pop("callback"), spec)
    return out


def core_configs():
    rows = []
    for path in sorted((PKG / "configs").glob("R3_*.json")):
        cfg = json.loads(path.read_text())
        if cfg.get("phase") == "stage1_core":
            rows.append(cfg)
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 stage1_core configs, found {len(rows)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    device = "cuda:0"
    rows = [json.loads(line) for line in PANEL.read_text().splitlines() if line.strip()]
    mine = [r for i, r in enumerate(rows) if i % args.shards == args.shard][: args.limit]
    processor = CLIPImageProcessor.from_pretrained(TOWER)
    model = CLIPVisionModel.from_pretrained(TOWER).to(device).eval()
    blocks = list(model.vision_model.encoder.layers)
    ids = torch.arange(576, device=device)
    bound = covers()
    summary = []
    with torch.no_grad():
        for cfg in core_configs():
            family = cfg["method"]["stage1"]["cover_family"]
            cover = bound[family] if family != "none" else None
            keep_ok = True
            payload_ok = 0
            for row in mine:
                image = Image.open(row["image"]).convert("RGB")
                pixel = processor(images=image, return_tensors="pt")["pixel_values"].to(device=device, dtype=model.dtype)
                with CLIPStage1(blocks, cfg, cover=cover) as capture:
                    model(pixel)
                    pen = capture.features[capture.pen][0, 1:]
                    selected = capture.select(pen, ids)
                    got = selected["payload"]
                    expect = pen.to(dtype=got.dtype).index_select(0, selected["keep"])
                    same = torch.equal(got, expect)
                payload_ok += int(same)
                keep_ok = keep_ok and int(selected["keep"].numel()) == cfg["method"]["schedule"]["counts"][0]
            summary.append({
                "id": cfg["id"],
                "budget": cfg["method"]["budget"],
                "guide": cfg["method"]["stage1"]["guide"],
                "cover_family": family,
                "k": int(cfg["method"]["schedule"]["counts"][0]),
                "n": len(mine),
                "keep_ok": keep_ok,
                "payload_is_pen_rows": payload_ok == len(mine),
            })
            print(summary[-1], flush=True)
    report = {"gpu": args.gpu, "shard": args.shard, "n_images": len(mine), "configs": summary}
    dest = PKG / "results" / f"r3_pilot_gpu{args.gpu}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print("WROTE", dest)


if __name__ == "__main__":
    main()
