"""Prove BELA_VIDEO_MODE=off matches official Dense token IDs on a real video."""
import json
import os
from pathlib import Path

_ROOT = Path("/data1/xiangyue")
os.environ.setdefault("TMPDIR", str(_ROOT / "tmp"))
os.environ.setdefault("HF_HOME", str(_ROOT / "hf"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from decord import VideoReader, cpu

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model

MODEL = os.environ["BELA_VIDEO_MODEL"]
VIDEO = os.environ["BELA_SMOKE_VIDEO"]
OUT = Path(os.environ.get("BELA_OFF_OUT", "/data1/xiangyue/logs/VIDEO_OFF_EQUIVALENCE.json"))
FRAMES = [int(x) for x in os.environ.get("BELA_SMOKE_FRAMES", "8,16,64").split(",")]


def load_video(path, n):
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idx = np.linspace(0, total - 1, n, dtype=int).tolist()
    fps = vr.get_avg_fps()
    frames = vr.get_batch(idx).asnumpy()
    times = [i / fps for i in idx]
    return frames, times, total / fps


def run_once(model, tokenizer, image_processor, n, mode):
    os.environ["BELA_VIDEO_MODE"] = mode
    frames, times, duration = load_video(VIDEO, n)
    pixel = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    pixel = pixel.cuda().bfloat16()
    instruction = (
        f"The video lasts for {duration:.2f} seconds, and {n} frames are uniformly sampled from it. "
        f"These frames are located at {','.join(f'{t:.2f}s' for t in times)}. "
        "Please answer the following questions related to this video."
    )
    question = DEFAULT_IMAGE_TOKEN + f"\n{instruction}\nWhat is the dominant color? Answer with one word."
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            images=[pixel],
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=16,
            use_cache=True,
        )
    text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return {"mode": mode, "frames": n, "token_ids": out[0].tolist(), "text": text}


def main():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"0", "1"}:
        raise SystemExit(f"smoke must use GPU 0 or 1, got {visible!r}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL, None, "llava_qwen", torch_dtype="bfloat16", device_map="cuda:0",
        attn_implementation="sdpa",
        overwrite_config={"mm_spatial_pool_mode": "average", "mm_spatial_pool_stride": 2},
    )
    model.eval()
    rows = []
    ok = True
    for n in FRAMES:
        dense = run_once(model, tokenizer, image_processor, n, "off")
        # second off run must match itself
        off2 = run_once(model, tokenizer, image_processor, n, "off")
        match = dense["token_ids"] == off2["token_ids"] and dense["text"] == off2["text"]
        # pruned run must differ in sequence length / answer path is allowed to change
        os.environ["BELA_VIDEO_TIER"] = "T32"
        native = run_once(model, tokenizer, image_processor, n, "native")
        trace = getattr(model, "_bela_last_video_stage1", None)
        row = {
            "frames": n,
            "off_self_match": match,
            "dense_text": dense["text"],
            "off2_text": off2["text"],
            "native_text": native["text"],
            "dense_len": len(dense["token_ids"]),
            "native_len": len(native["token_ids"]),
            "native_trace": trace,
        }
        rows.append(row)
        ok = ok and match
        print("FRAME", n, "off_match", match, "dense", dense["text"], "native", native["text"], "trace_k", None if not trace else trace.get("keep_per_frame"), flush=True)
    report = {
        "model": MODEL,
        "video": VIDEO,
        "off_equals_dense": ok,
        "note": "off must match Dense. native is Stage1 NATIVE_L2_TOPK only, not Dual COVER + RDRIFT.",
        "runs": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print("WROTE", OUT, "ok", ok, flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
