"""CDPruner TTFT / Fixed32 bench (same protocol shape as starpro_x0_t64/bench_ttft.py).

Warmup + timed generate; CUDA synchronize; synthetic 336 images.
Writes logs/cdpruner_x0/ttft/CDPRUNER_K{K}/latency_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO = Path("/home/xiangyue/ViTCoP")
sys.path = [
    str(REPO / "third_party/CDPruner"),
    str(REPO / "experiments/cdpruner_x0"),
    str(REPO / "lmms-eval"),
] + [p for p in sys.path if "ViTCoP/llava" not in p and p != str(REPO / "llava")]

MODEL_PATH = os.environ.get(
    "CDPRUNER_MODEL_PATH",
    "/home/xiangyue/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234",
)
N_WARM = int(os.environ.get("CDPRUNER_TTFT_WARM", "10"))
N_MEAS = int(os.environ.get("CDPRUNER_TTFT_MEAS", "50"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual-token-num", type=int, required=True, choices=[32, 64, 128, 192, 256])
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()
    k = int(args.visual_token_num)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    out = REPO / "logs/cdpruner_x0/ttft" / f"CDPRUNER_K{k}"
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "latency_summary.json"
    if summary_path.exists():
        print(f"[skip] {summary_path}", flush=True)
        return

    tok, model, proc, _ = load_pretrained_model(
        MODEL_PATH,
        None,
        "llava-v1.5-7b",
        device_map="cuda:0",
        visual_token_num=k,
    )
    model.eval()
    qtext = "Describe this image briefly."
    images = [
        Image.new("RGB", (336, 336), (i * 5 % 256, 80, 160))
        for i in range(max(N_WARM, N_MEAS))
    ]

    def one(img: Image.Image, new_tokens: int) -> float:
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + qtext)
        conv.append_message(conv.roles[1], None)
        ids = (
            tokenizer_image_token(
                conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        imgs = process_images([img], proc, model.config).to(
            dtype=torch.float16, device="cuda"
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                ids,
                images=imgs,
                image_sizes=[img.size],
                texts=qtext,
                do_sample=False,
                temperature=0,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                use_cache=True,
            )
        torch.cuda.synchronize()
        _ = out_ids[0] if isinstance(out_ids, tuple) else out_ids
        return (time.perf_counter() - t0) * 1000.0

    for i in range(N_WARM):
        one(images[i % len(images)], 1)
        print(f"[warmup TTFT] K={k} {i+1}/{N_WARM}", flush=True)
    ttft = [one(images[i % len(images)], 1) for i in range(N_MEAS)]
    print(f"[meas TTFT] K={k} n={N_MEAS} mean={statistics.mean(ttft):.2f} ms", flush=True)

    for i in range(N_WARM):
        one(images[i % len(images)], 32)
        print(f"[warmup Fixed32] K={k} {i+1}/{N_WARM}", flush=True)
    fixed32 = [one(images[i % len(images)], 32) for i in range(N_MEAS)]
    print(
        f"[meas Fixed32] K={k} n={N_MEAS} mean={statistics.mean(fixed32):.2f} ms",
        flush=True,
    )

    payload = {
        "method": "cdpruner",
        "visual_token_num": k,
        "W_proxy_note": f"fixed K={k} all layers; W≈{k}*32 only if counted that way (not BELA schedule)",
        "pretrained": MODEL_PATH,
        "n_warm": N_WARM,
        "n_meas": N_MEAS,
        "ttft_ms": {
            "mean": statistics.mean(ttft),
            "std": statistics.pstdev(ttft) if len(ttft) > 1 else 0.0,
            "median": statistics.median(ttft),
            "samples": ttft,
        },
        "fixed32_ms": {
            "mean": statistics.mean(fixed32),
            "std": statistics.pstdev(fixed32) if len(fixed32) > 1 else 0.0,
            "median": statistics.median(fixed32),
            "samples": fixed32,
        },
        "note": "Synthetic gray images; wall-clock CUDA-synced generate. Matches starpro_x0_t64 protocol shape.",
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"[ok] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
