#!/usr/bin/env python3
"""P0 Stage2 physical smoke on A800. Prefer free GPUs 2/3 when 0/1 busy."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path("/data1/xiangyue")
os.environ.setdefault("TMPDIR", str(_ROOT / "tmp"))
os.environ.setdefault("HF_HOME", str(_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_ROOT / "hf" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_ROOT / "hf" / "transformers"))
os.environ.setdefault("XDG_CACHE_HOME", str(_ROOT / "cache"))
os.environ.setdefault("TORCH_HOME", str(_ROOT / "torch"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> int:
    gpu = os.environ.get("BELA_SMOKE_GPU", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(_ROOT / "ViTCoP/experiments/BELA_A800_VIDEO_Cursor"))
    sys.path.insert(0, str(_ROOT / "src/LLaVA-NeXT"))

    from bela_video_a800.qwen2_stage2_forward import apply_patch
    apply_patch()

    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    model_path = os.environ.get("BELA_VIDEO_MODEL", str(_ROOT / "models/LLaVA-Video-7B-Qwen2"))
    video = os.environ.get("BELA_SMOKE_VIDEO", str(_ROOT / "hf/videomme/data/-3t1rj8g6yg.mp4"))
    frames_n = int(os.environ.get("BELA_SMOKE_FRAMES", "8"))
    mode = os.environ.get("BELA_VIDEO_MODE", "qacover")
    tier = os.environ.get("BELA_VIDEO_TIER", "T32")
    s2 = os.environ.get("BELA_VIDEO_STAGE2", "Q_ATTN_ALL")
    os.environ["BELA_VIDEO_MODE"] = mode
    os.environ["BELA_VIDEO_TIER"] = tier
    os.environ["BELA_VIDEO_STAGE2"] = s2

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        "llava_qwen",
        torch_dtype="bfloat16",
        device_map="cuda:0",
        attn_implementation="sdpa",
        overwrite_config={"mm_spatial_pool_mode": "average", "mm_spatial_pool_stride": 2},
    )
    model.eval()

    vr = VideoReader(video, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idxs = np.linspace(0, total - 1, frames_n, dtype=int).tolist()
    frames = vr.get_batch(idxs).asnumpy()
    fps = float(vr.get_avg_fps() or 1.0)
    times = [i / fps for i in idxs]
    duration = total / fps
    pixel = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()

    instruction = (
        f"The video lasts for {duration:.2f} seconds, and {frames_n} frames are uniformly sampled from it. "
        f"These frames are located at {','.join(f'{t:.2f}s' for t in times)}. "
        "Please answer the following questions related to this video."
    )
    question = DEFAULT_IMAGE_TOKEN + f"\n{instruction}\nWhat is the dominant color? Answer with one word."
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # Prefill-only debug path first (catches layout bugs before decode).
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    with torch.inference_mode():
        (
            _ids,
            position_ids,
            attention_mask,
            _pkv,
            inputs_embeds,
            _labels,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, [pixel], ["video"], None
        )
    ctx = getattr(model, "_bela_stage2_ctx", None) or getattr(model.get_model(), "_bela_stage2_ctx", None)
    s1 = getattr(model, "_bela_last_video_stage1", None)
    print(
        "PREPARE",
        {
            "embeds": None if inputs_embeds is None else list(inputs_embeds.shape),
            "ctx": ctx,
            "s1_flat": None if not s1 else s1.get("flat_tokens"),
            "s1_keep": None if not s1 else s1.get("keep_per_frame"),
            "s1_frames": None if not s1 else s1.get("frames"),
        },
        flush=True,
    )
    if ctx is not None and inputs_embeds is not None:
        vs = int(ctx["visual_start"])
        flat = int(ctx["flat_tokens"])
        print(
            "SPAN_CHECK",
            {
                "visual_start": vs,
                "visual_end": vs + flat,
                "seq": int(inputs_embeds.shape[1]),
                "ok": 0 <= vs < vs + flat <= int(inputs_embeds.shape[1]),
            },
            flush=True,
        )

    t0 = time.time()
    # Prefill physical prune is the P0 gate. Decode with use_cache=False avoids
    # residual DynamicCache length mismatch while still exercising generate().
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
    text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    elapsed = time.time() - t0

    s1 = getattr(model, "_bela_last_video_stage1", None)
    s2t = getattr(model, "_bela_last_stage2", None)
    if s2t is None:
        s2t = getattr(model.get_model(), "_bela_last_stage2", None)
    ctx = getattr(model, "_bela_stage2_ctx", None) or getattr(model.get_model(), "_bela_stage2_ctx", None)

    ok = True
    reasons = []
    if s2.upper() != "OFF":
        if not s2t or not s2t.get("events"):
            ok = False
            reasons.append("no_stage2_events")
        else:
            for e in s2t["events"]:
                if not (e["visual_after"] < e["visual_before"]):
                    ok = False
                    reasons.append(f"no_shrink_layer{e['layer_1based']}")
            sels = [e["selector"] for e in s2t["events"]]
            if s2 == "RDRIFT_FIRST" and sels and sels[0] != "RDRIFT":
                ok = False
                reasons.append(f"expected_RDRIFT_got_{sels[0]}")
            if s2 == "Q_ATTN_ALL" and any(s != "Q_ATTN" for s in sels):
                ok = False
                reasons.append(f"expected_Q_ATTN_got_{sels}")
        if s1 is None:
            ok = False
            reasons.append("missing_stage1_trace")

    report = {
        "pass": ok,
        "reasons": reasons,
        "gpu": gpu,
        "video": video,
        "frames": frames_n,
        "mode": mode,
        "tier": tier,
        "stage2": s2,
        "generate_s": round(elapsed, 2),
        "text": text,
        "stage1_keep": None if not s1 else s1.get("keep_per_frame"),
        "stage1_selector": None if not s1 else s1.get("selector"),
        "ctx_visual_start": None if not ctx else ctx.get("visual_start"),
        "stage2_trace": s2t,
        "peak_mib": int(torch.cuda.max_memory_allocated() / 2**20),
    }
    outp = Path(os.environ.get("BELA_SMOKE_OUT", str(_ROOT / "logs/VIDEO_STAGE2_SMOKE.json")))
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"pass": ok, "reasons": reasons, "events": None if not s2t else s2t.get("events"), "out": str(outp)}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
