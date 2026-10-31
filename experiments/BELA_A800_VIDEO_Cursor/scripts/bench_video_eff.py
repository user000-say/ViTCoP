#!/usr/bin/env python3
"""A800 video EFF: TTFT(1) + Fixed32 + optional operator FLOPs. Batch=1, CUDA sync."""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path("/data1/xiangyue")
os.environ.setdefault("TMPDIR", str(_ROOT / "tmp"))
os.environ.setdefault("HF_HOME", str(_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_ROOT / "hf/hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_ROOT / "hf/transformers"))
os.environ.setdefault("XDG_CACHE_HOME", str(_ROOT / "cache"))
os.environ.setdefault("TORCH_HOME", str(_ROOT / "torch"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _stats(xs):
    s = sorted(xs)
    return {
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p90": s[min(len(s) - 1, int(0.9 * len(s)))],
        "min": s[0],
        "max": s[-1],
        "n": len(s),
    }


def main() -> int:
    gpu = os.environ.get("BELA_EFF_GPU", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(_ROOT / "ViTCoP/experiments/BELA_A800_VIDEO_Cursor"))
    sys.path.insert(0, str(_ROOT / "src/LLaVA-NeXT"))

    from bela_video_a800.qwen2_stage2_forward import apply_patch
    apply_patch()
    if mode.startswith("dual") or os.environ.get("BELA_VIDEO_DUAL") == "1":
        from bela_video_a800.siglip_dual_capture import install_vision_capture
        # install after model load below
        os.environ["_BELA_NEED_DUAL_CAPTURE"] = "1"

    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    mode = os.environ.get("BELA_VIDEO_MODE", "off")
    tier = os.environ.get("BELA_VIDEO_TIER", "T32")
    s2 = os.environ.get("BELA_VIDEO_STAGE2", "OFF")
    arm = os.environ.get("BELA_EFF_ARM", f"{mode}_{s2}_{tier}")
    frames_n = int(os.environ.get("BELA_EFF_FRAMES", "64"))
    warm = int(os.environ.get("BELA_EFF_WARMUP", "10"))
    meas = int(os.environ.get("BELA_EFF_MEAS", "50"))
    panel = os.environ.get(
        "BELA_EFF_VIDEO",
        str(_ROOT / "hf/videomme/data/-3t1rj8g6yg.mp4"),
    )
    outp = Path(os.environ.get("BELA_EFF_OUT", str(_ROOT / "logs/video_slim_eff" / arm / "latency_summary.json")))
    outp.parent.mkdir(parents=True, exist_ok=True)

    os.environ["BELA_VIDEO_MODE"] = mode
    os.environ["BELA_VIDEO_TIER"] = tier
    os.environ["BELA_VIDEO_STAGE2"] = s2

    model_path = str(_ROOT / "models/LLaVA-Video-7B-Qwen2")
    tok, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        torch_dtype="bfloat16", device_map="cuda:0", attn_implementation="sdpa",
        overwrite_config={"mm_spatial_pool_mode": "average", "mm_spatial_pool_stride": 2},
    )
    model.eval()
    if os.environ.get("_BELA_NEED_DUAL_CAPTURE") == "1":
        from bela_video_a800.siglip_dual_capture import install_vision_capture
        install_vision_capture(model)

    vr = VideoReader(panel, ctx=cpu(0), num_threads=1)
    idxs = np.linspace(0, len(vr) - 1, frames_n, dtype=int).tolist()
    frames = vr.get_batch(idxs).asnumpy()
    fps = float(vr.get_avg_fps() or 1.0)
    times = [i / fps for i in idxs]
    duration = len(vr) / fps
    pixel = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    instruction = (
        f"The video lasts for {duration:.2f} seconds, and {frames_n} frames are uniformly sampled from it. "
        f"These frames are located at {','.join(f'{t:.2f}s' for t in times)}. "
        "Please answer the following questions related to this video."
    )
    question = DEFAULT_IMAGE_TOKEN + f"\n{instruction}\nWhat is happening? Answer briefly."
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    def one_gen(max_new: int):
        # reset stage2 stash each call
        if hasattr(model, "_bela_prefill_seq_len"):
            model._bela_prefill_seq_len = None
        with torch.inference_mode():
            return model.generate(
                input_ids,
                images=[pixel],
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new,
                use_cache=True,
            )

    # warmup
    for _ in range(warm):
        one_gen(1)
        torch.cuda.synchronize()

    ttft, f32 = [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(meas):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_gen(1)
        torch.cuda.synchronize()
        ttft.append((time.perf_counter() - t0) * 1000)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_gen(32)
        torch.cuda.synchronize()
        f32.append((time.perf_counter() - t0) * 1000)

    # light FLOPs probe (1 forward generate(1) under profiler matmul/conv)
    flops = None
    try:
        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
            one_gen(1)
            torch.cuda.synchronize()
        total = 0
        for e in prof.key_averages():
            if e.flops:
                total += int(e.flops)
        flops = {"operator_counted_flops": total, "tflops": total / 1e12, "note": "torch.profiler with_flops; fused attn/topk may be undercounted"}
    except Exception as e:
        flops = {"error": str(e)}

    s2t = getattr(model, "_bela_last_stage2", None)
    report = {
        "arm": arm,
        "mode": mode,
        "tier": tier,
        "stage2": s2,
        "gpu": gpu,
        "frames": frames_n,
        "video": panel,
        "warmup": warm,
        "measured": meas,
        "protocol": "STAR-like: batch1, CUDA sync, exclude CPU video decode from timed fn (frames pre-decoded once)",
        "ttft_wall_ms": _stats(ttft),
        "fixed32_wall_ms": _stats(f32),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "stage2_last_events": None if not s2t else s2t.get("events"),
        "flops": flops,
    }
    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps({"arm": arm, "ttft_mean": report["ttft_wall_ms"]["mean"], "fixed32_mean": report["fixed32_wall_ms"]["mean"], "out": str(outp)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
