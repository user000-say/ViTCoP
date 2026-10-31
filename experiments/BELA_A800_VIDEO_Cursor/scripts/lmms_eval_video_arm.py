#!/usr/bin/env python3
"""Run lmms_eval videomme for one BELA video arm with Stage2 + optional Dual capture."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path("/data1/xiangyue")
sys.path.insert(0, str(_ROOT / "ViTCoP/experiments/BELA_A800_VIDEO_Cursor"))
sys.path.insert(0, str(_ROOT / "src/LLaVA-NeXT"))
sys.path.insert(0, str(_ROOT / "src/lmms-eval"))


def main() -> int:
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

    # Wrap model builder so Dual capture installs after load.
    import llava.model.builder as builder
    _orig = builder.load_pretrained_model

    def _load(*a, **k):
        tok, model, image_processor, max_len = _orig(*a, **k)
        apply_patch()
        if args.mode.startswith("dual") or os.environ.get("BELA_VIDEO_DUAL") == "1":
            from bela_video_a800.siglip_dual_capture import install_vision_capture
            install_vision_capture(model)
        # also ensure Stage2 parent ref if used
        return tok, model, image_processor, max_len

    builder.load_pretrained_model = _load

    # Also hook lmms_eval llava_vid path if it imports differently
    argv = [
        "lmms_eval",
        "--model", "llava_vid",
        "--model_args", args.model_args,
        "--tasks", "videomme",
        "--batch_size", "1",
        "--output_path", args.output,
        "--log_samples",
    ]
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    sys.argv = argv
    from lmms_eval.__main__ import cli_evaluate
    cli_evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
