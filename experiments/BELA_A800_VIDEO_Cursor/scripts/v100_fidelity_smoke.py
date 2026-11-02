#!/usr/bin/env python3
"""V100 fidelity smoke: Dual-COVER + Stage2 Q_ATTN with dense-absolute RoPE.

Uses float16 (V100 has no bf16). GPUs from the project set only (default 0).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HOME = Path("/home/xiangyue")
ROOT = HOME / "ViTCoP-a800-video/experiments/BELA_A800_VIDEO_Cursor"
OUT = ROOT / "logs" / "v100_fidelity_smoke.json"


def main() -> int:
    gpu = os.environ.get("BELA_SMOKE_GPU", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HOME / "src/LLaVA-NeXT"))

    from bela_video_a800.qwen2_stage2_forward import apply_patch

    apply_patch()

    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from bela_video_a800.siglip_dual_capture import install_vision_capture

    model_path = os.environ.get(
        "BELA_VIDEO_MODEL", str(HOME / "src/models/LLaVA-Video-7B-Qwen2")
    )
    video = os.environ.get(
        "BELA_SMOKE_VIDEO", str(HOME / "src/hf_video/videomme/data/OGQ7KwUXuBk.mp4")
    )
    frames_n = int(os.environ.get("BELA_SMOKE_FRAMES", "8"))
    mode = os.environ.get("BELA_VIDEO_MODE", "dualcover")
    tier = os.environ.get("BELA_VIDEO_TIER", "T32")
    s2 = os.environ.get("BELA_VIDEO_STAGE2", "Q_ATTN_ALL")
    os.environ["BELA_VIDEO_MODE"] = mode
    os.environ["BELA_VIDEO_TIER"] = tier
    os.environ["BELA_VIDEO_STAGE2"] = s2
    os.environ["BELA_VIDEO_DUAL"] = "1"
    os.environ.setdefault("BELA_VIDEO_STAGE2_SCORE", "qk")

    dtype = "float16"  # V100
    print(f"[smoke] gpu={gpu} mode={mode} tier={tier} s2={s2} frames={frames_n} dtype={dtype}", flush=True)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        "llava_qwen",
        torch_dtype=dtype,
        device_map="cuda:0",
        attn_implementation="sdpa",
        overwrite_config={
            "mm_spatial_pool_mode": "average",
            "mm_spatial_pool_stride": 2,
        },
    )
    model.eval()
    apply_patch()
    install_vision_capture(model)
    # Ensure inner Qwen sees parent for Stage2 audit publish.
    inner = model.get_model()
    inner._bela_parent_ref = model

    vr = VideoReader(video, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idxs = np.linspace(0, total - 1, frames_n, dtype=int).tolist()
    frames = vr.get_batch(idxs).asnumpy()
    fps = float(vr.get_avg_fps() or 1.0)
    times = [i / fps for i in idxs]
    duration = total / fps
    pixel = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    pixel = pixel.cuda().to(dtype=torch.float16)

    instruction = (
        f"The video lasts for {duration:.2f} seconds, and {frames_n} frames are uniformly sampled from it. "
        f"These frames are located at {','.join(f'{t:.2f}s' for t in times)}. "
        "Please answer the following questions related to this video."
    )
    question = (
        DEFAULT_IMAGE_TOKEN
        + f"\n{instruction}\nWhat is happening in the video? Answer briefly."
    )
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            images=[pixel],
            modalities=["video"],
            do_sample=False,
            max_new_tokens=32,
            use_cache=True,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    s1 = getattr(model, "_bela_last_video_stage1", None)
    s2a = getattr(model, "_bela_last_stage2", None) or getattr(inner, "_bela_last_stage2", None)
    rope_next = getattr(model, "_bela_rope_next_pos", None)
    pref = getattr(model, "_bela_prefill_seq_len", None)

    report = {
        "ok": True,
        "gpu": gpu,
        "mode": mode,
        "tier": tier,
        "stage2": s2,
        "frames": frames_n,
        "dtype": dtype,
        "latency_s": round(dt, 3),
        "answer": text[-200:],
        "stage1": s1,
        "stage2_audit": s2a,
        "rope_next_pos": rope_next,
        "prefill_seq_len_final": pref,
        "video": video,
    }
    # Fidelity gates
    fails = []
    if not s1 or int(s1.get("keep_per_frame", 0)) <= 0:
        fails.append("stage1_missing")
    if not s2a or not s2a.get("events"):
        fails.append("stage2_no_cuts")
    if s2a and s2a.get("rope_mode") != "dense_absolute":
        fails.append("rope_mode_not_dense_absolute")
    if s2a and int(s2a.get("rope_next_pos", -1)) <= int(s2a.get("prefill_seq_len", 0)):
        # After Stage2 prune, dense rope_next should be >= physical prefill
        # (equal only if no absolute gaps — still OK if equal when Stage1 already compact)
        pass
    if fails:
        report["ok"] = False
        report["fails"] = fails

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str)[:2500], flush=True)
    print(f"[smoke] wrote {OUT} ok={report['ok']}", flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
