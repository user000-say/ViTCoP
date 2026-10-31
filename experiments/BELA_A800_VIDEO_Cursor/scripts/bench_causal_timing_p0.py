#!/usr/bin/env python3
"""P0: NATIVE_ONLINE vs QA_ONLINE vs QA_REPLAY timing on frozen video panel."""
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _stats(xs):
    s = sorted(xs)
    return {
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p90": s[min(len(s) - 1, int(0.9 * len(s)))],
        "n": len(s),
    }


def main() -> int:
    gpu = os.environ.get("BELA_EFF_GPU", "2")
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

    frames_n = int(os.environ.get("BELA_EFF_FRAMES", "64"))
    warm = int(os.environ.get("BELA_EFF_WARMUP", "5"))
    meas = int(os.environ.get("BELA_EFF_MEAS", "20"))
    panel = os.environ.get(
        "BELA_EFF_VIDEO",
        str(_ROOT / "hf/videomme/data/-3t1rj8g6yg.mp4"),
    )
    outp = Path(os.environ.get(
        "BELA_P0_OUT",
        str(_ROOT / "logs/VIDEO_CAUSAL_TIMING.json"),
    ))
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Always use Stage2 Q_ATTN_ALL T32 for C-matched path
    os.environ["BELA_VIDEO_TIER"] = "T32"
    os.environ["BELA_VIDEO_STAGE2"] = "Q_ATTN_ALL"
    os.environ["BELA_VIDEO_QA_BATCH"] = os.environ.get("BELA_VIDEO_QA_BATCH", "1")

    model_path = str(_ROOT / "models/LLaVA-Video-7B-Qwen2")
    tok, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        torch_dtype="bfloat16", device_map="cuda:0", attn_implementation="sdpa",
        overwrite_config={"mm_spatial_pool_mode": "average", "mm_spatial_pool_stride": 2},
    )
    model.eval()

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

    def one_gen(mode: str, max_new: int = 1):
        os.environ["BELA_VIDEO_MODE"] = mode
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

    def timed(mode: str):
        for _ in range(warm):
            one_gen(mode, 1)
            torch.cuda.synchronize()
        xs = []
        for _ in range(meas):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = one_gen(mode, 1)
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t0) * 1000)
        text = tok.decode(out[0], skip_special_tokens=True)
        return _stats(xs), text, out[0].detach().cpu().tolist()

    # Capture QA keep IDs once for REPLAY (stash via stage1 trace on model)
    os.environ["BELA_VIDEO_MODE"] = "qacover"
    one_gen("qacover", 1)
    torch.cuda.synchronize()
    s1 = getattr(model, "_bela_last_video_stage1", None)
    replay_ids = None if not s1 else s1.get("ids")

    report = {
        "gpu": gpu,
        "frames": frames_n,
        "video": panel,
        "qa_batch": os.environ.get("BELA_VIDEO_QA_BATCH"),
        "warmup": warm,
        "measured": meas,
        "arms": {},
    }

    for name, mode in (
        ("NATIVE_ONLINE", "native"),
        ("QA_ONLINE", "qacover"),
    ):
        st, text, toks = timed(mode)
        report["arms"][name] = {"ttft_ms": st, "text": text[-200:], "n_out_tokens": len(toks)}

    # QA_REPLAY: inject frozen ids if hook supports; else note unavailable
    # Minimal support: env BELA_VIDEO_MODE=replay + stash ids on model
    if replay_ids is not None:
        model._bela_replay_stage1_ids = replay_ids
        os.environ["BELA_VIDEO_MODE"] = "replay"
        try:
            st, text, toks = timed("replay")
            report["arms"]["QA_REPLAY"] = {
                "ttft_ms": st,
                "text": text[-200:],
                "note": "diagnostic only — not online method",
            }
        except Exception as e:
            report["arms"]["QA_REPLAY"] = {"error": str(e), "note": "replay hook missing or failed"}
        finally:
            if hasattr(model, "_bela_replay_stage1_ids"):
                delattr(model, "_bela_replay_stage1_ids")
    else:
        report["arms"]["QA_REPLAY"] = {"error": "no stage1 ids captured"}

    nat = report["arms"]["NATIVE_ONLINE"]["ttft_ms"]["mean"]
    qa = report["arms"]["QA_ONLINE"]["ttft_ms"]["mean"]
    report["qa_minus_native_ms"] = qa - nat
    if "QA_REPLAY" in report["arms"] and "ttft_ms" in report["arms"]["QA_REPLAY"]:
        rep = report["arms"]["QA_REPLAY"]["ttft_ms"]["mean"]
        report["online_minus_replay_ms"] = qa - rep
        report["replay_minus_native_ms"] = rep - nat

    outp.write_text(json.dumps(report, indent=2))
    md = outp.with_suffix(".md")
    lines = [
        "# VIDEO_CAUSAL_TIMING",
        "",
        f"- panel: `{panel}` F={frames_n} GPU={gpu} QA_BATCH={report['qa_batch']}",
        f"- NATIVE_ONLINE TTFT mean: {nat:.2f} ms",
        f"- QA_ONLINE TTFT mean: {qa:.2f} ms",
        f"- QA−Native: {qa - nat:.2f} ms",
    ]
    if "online_minus_replay_ms" in report:
        lines.append(f"- QA_REPLAY TTFT mean: {report['arms']['QA_REPLAY']['ttft_ms']['mean']:.2f} ms")
        lines.append(f"- Online−Replay (selector tax): {report['online_minus_replay_ms']:.2f} ms")
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
