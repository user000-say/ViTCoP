#!/usr/bin/env python3
"""V100 Video-MME arm runner (home paths, fp16). Same Stage2/Dual wiring as A800."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HOME = Path("/home/xiangyue")
ROOT = HOME / "ViTCoP-a800-video/experiments/BELA_A800_VIDEO_Cursor"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HOME / "src/LLaVA-NeXT"))
sys.path.insert(0, str(HOME / "ViTCoP-a800-video/lmms-eval"))


def _force_offline() -> None:
    hf = os.environ.setdefault("HF_HOME", str(HOME / "src/hf_video"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf) / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    data = Path(hf) / "videomme" / "data"
    if not data.is_dir() or not any(data.glob("*.mp4")):
        raise SystemExit(f"offline Video-MME videos missing under {data}")


def main() -> int:
    _force_offline()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--stage2", default="Q_ATTN_ALL")
    ap.add_argument("--tier", default="T32")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_args", required=True)
    args = ap.parse_args()

    os.environ["BELA_VIDEO_MODE"] = args.mode
    os.environ["BELA_VIDEO_TIER"] = args.tier
    os.environ["BELA_VIDEO_STAGE2"] = args.stage2
    os.environ.setdefault("BELA_VIDEO_QA_BATCH", "1")
    os.environ.setdefault("BELA_VIDEO_STAGE2_SCORE", "qk")
    if args.mode.startswith("dual"):
        os.environ["BELA_VIDEO_DUAL"] = "1"

    from bela_video_a800.qwen2_stage2_forward import apply_patch

    apply_patch()

    import llava.model.builder as builder

    _orig = builder.load_pretrained_model

    def _load(*a, **k):
        # V100: force fp16 if caller passed bf16
        if "torch_dtype" in k and str(k["torch_dtype"]) in ("bfloat16", "bf16", "torch.bfloat16"):
            k["torch_dtype"] = "float16"
        tok, model, image_processor, max_len = _orig(*a, **k)
        apply_patch()
        if args.mode.startswith("dual") or os.environ.get("BELA_VIDEO_DUAL") == "1":
            from bela_video_a800.siglip_dual_capture import install_vision_capture

            install_vision_capture(model)
        return tok, model, image_processor, max_len

    builder.load_pretrained_model = _load

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    argv = [
        "lmms_eval",
        "--model",
        "llava_vid",
        "--model_args",
        args.model_args,
        "--tasks",
        "videomme",
        "--batch_size",
        "1",
        "--output_path",
        str(out),
        "--log_samples",
    ]
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    sys.argv = argv
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()
    hits = list(out.glob("**/results*.json")) + list(out.glob("**/*results*.json"))
    if not hits:
        raise SystemExit(f"FAIL {args.arm}: no results*.json under {out}")
    print(f"OK {args.arm} results={hits[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
