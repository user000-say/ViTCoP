"""Official LLaVA-Video Dense generate. No pruning, no QA-COVER."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model

MODEL = os.environ["BELA_VIDEO_MODEL"]
VIDEO = os.environ["BELA_SMOKE_VIDEO"]
OUT = Path(os.environ.get("BELA_SMOKE_OUT", "/data1/xiangyue/logs/VIDEO_DENSE_SMOKE.json"))
FRAMES = [int(x) for x in os.environ.get("BELA_SMOKE_FRAMES", "8,16,64").split(",")]


def load_video(path, n):
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idx = np.linspace(0, total - 1, n, dtype=int).tolist()
    fps = vr.get_avg_fps()
    frames = vr.get_batch(idx).asnumpy()
    times = [i / fps for i in idx]
    return frames, times, total / fps


def main():
    raw = json.loads(Path(MODEL, "config.json").read_text())
    t0 = time.time()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL,
        None,
        "llava_qwen",
        torch_dtype="bfloat16",
        device_map="cuda:0",
        attn_implementation="sdpa",
        overwrite_config={"mm_spatial_pool_mode": "average", "mm_spatial_pool_stride": 2},
    )
    model.eval()
    load_s = time.time() - t0
    cfg = model.config
    report = {
        "model": MODEL,
        "prune": "none",
        "attn": "sdpa",
        "dtype": "bfloat16",
        "load_s": round(load_s, 2),
        "checkpoint_pool_mode": raw.get("mm_spatial_pool_mode"),
        "checkpoint_pool_stride": raw.get("mm_spatial_pool_stride"),
        "runtime_pool_mode": cfg.mm_spatial_pool_mode,
        "runtime_pool_stride": cfg.mm_spatial_pool_stride,
        "decoder_layers": cfg.num_hidden_layers,
        "vision_layers": getattr(cfg, "mm_vision_select_layer", None),
        "runs": [],
    }
    for n in FRAMES:
        frames, times, duration = load_video(VIDEO, n)
        pixel = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        pixel = pixel.cuda().bfloat16()
        with torch.inference_mode():
            tower = model.get_model().get_vision_tower()
            raw_feat = tower(pixel)
            projected = model.get_model().mm_projector(raw_feat)
            pooled = model.get_2dPool(projected, cfg.mm_spatial_pool_stride)
        instruction = (
            f"The video lasts for {duration:.2f} seconds, and {n} frames are uniformly sampled from it. "
            f"These frames are located at {','.join(f'{t:.2f}s' for t in times)}. "
            "Please answer the following questions related to this video."
        )
        question = DEFAULT_IMAGE_TOKEN + f"\n{instruction}\nWhat is the dominant color? Answer with one word."
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                images=[pixel],
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=16,
            )
        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        report["runs"].append({
            "frames": n,
            "pixel_shape": list(pixel.shape),
            "vision_shape": list(raw_feat.shape),
            "projected_shape": list(projected.shape),
            "pooled_shape": list(pooled.shape),
            "pooled_tokens_per_frame": int(pooled.shape[1]),
            "generate_s": round(time.time() - t1, 2),
            "peak_mib": int(torch.cuda.max_memory_allocated() / 2**20),
            "text": text,
        })
        print(json.dumps(report["runs"][-1]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print("SMOKE_WRITTEN", OUT)


if __name__ == "__main__":
    main()
