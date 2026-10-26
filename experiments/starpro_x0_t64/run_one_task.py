#!/usr/bin/env python3
"""Run one lmms-eval task with official STAR-Pro T=64 in the isolated env.

Usage (must use /home/xiangyue/envs/starpro/bin/python):
  CUDA_VISIBLE_DEVICES=0 python run_one_task.py --task gqa --gpu-index 0
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
LLAVA_STAR = REPO / "third_party/LLaVA-starpro"
PKG = REPO / "experiments/starpro_x0_t64"
OUT_ROOT = REPO / "logs/starpro_x0_t64/full"

# Prefer STAR-Pro LLaVA overlay; keep repo lmms-eval; never use ViTCoP/llava.
sys.path = [
    str(LLAVA_STAR),
    str(PKG),
    str(REPO / "lmms-eval"),
    str(REPO),
] + [p for p in sys.path if "ViTCoP/llava" not in p and p != str(REPO / "llava")]

MODEL_PATH = os.environ.get(
    "STARPRO_MODEL_PATH",
    "/home/xiangyue/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234",
)

FULL_N = {
    "gqa": 12578,
    "pope": 9000,
    "mme": 2374,
    "mmbench_en_dev": 4329,
    "mmbench_cn_dev": 4329,
    "scienceqa_img": 2017,
    "mmmu_val": 900,
    "ai2d": 3088,
    "textvqa_val_ocr": 5000,
    "textvqa_val": 5000,
    "vizwiz_vqa_val": 4319,
}


def pick(block, name):
    for key, value in block.items():
        if key.split(",")[0] == name and isinstance(value, (int, float)):
            return float(value)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--T", type=int, default=64)
    args = ap.parse_args()
    if args.task not in FULL_N:
        raise SystemExit(f"unknown task {args.task}")

    # Official STAR-Pro runner env (scripts/run_eval.sh)
    os.environ["STAGE1_SCORER"] = "qr"
    os.environ["STAGE1_MULT"] = "2"
    os.environ["TEXT_AGG_MODE"] = "average_all"
    os.environ["STAGE2_SELECTOR"] = "topk"
    os.environ["PRUNING_SCHEDULE_MODE"] = "progressive"
    os.environ["STAGE2_ANCHOR_M"] = "0"
    os.environ["STAR_CAUSAL_FIX"] = "0"
    os.environ["STAR_KEEP_POSIDS"] = "0"
    os.environ.pop("CUSTOM_PRUNING_SCHEDULE", None)
    # Absolute snapshot path for weights. Unset hub/datasets offline so lmms-eval
    # can resolve cached HF datasets (datasets 2.16 still hits hub metadata).
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    # Do NOT clobber a parent-set CUDA_VISIBLE_DEVICES (would force all jobs onto GPU0).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    dest = OUT_ROOT / f"STARPRO_T{args.T}" / args.task
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "metrics.json").exists() and not args.limit:
        print(f"[skip] {dest / 'metrics.json'}", flush=True)
        return

    # Register custom model into lmms-eval's AVAILABLE_MODELS table.
    import llava_starpro_lmms  # noqa: F401
    import lmms_eval.models as _lmms_models

    _lmms_models.AVAILABLE_MODELS["llava_starpro"] = "llava_starpro_lmms.LlavaStarPro"

    model_args = (
        f"pretrained={MODEL_PATH},"
        f"model_name=llava-v1.5-7b,"
        f"visual_token_num={args.T},"
        f"pruning_method=star_pro,"
        f"conv_template=vicuna_v1,"
        f"device_map=cuda:0"
    )
    argv = [
        "lmms_eval",
        "--model",
        "llava_starpro",
        "--model_args",
        model_args,
        "--tasks",
        args.task,
        "--batch_size",
        "1",
        "--log_samples",
        "--output_path",
        str(dest),
    ]
    if args.limit:
        argv.extend(["--limit", str(args.limit)])
    sys.argv = argv
    print("[starpro] " + " ".join(argv), flush=True)
    try:
        runpy.run_module("lmms_eval", run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise SystemExit(exc.code)

    files = [p for p in dest.rglob("*_results.json") if "submissions" not in p.parts]
    if not files:
        raise SystemExit("ENGINEERING_FAIL results json missing")
    path = max(files, key=lambda item: item.stat().st_mtime)
    payload = json.loads(path.read_text())
    block = payload["results"][args.task]
    effective = int(payload.get("n-samples", {}).get(args.task, {}).get("effective", -1))
    expected_n = args.limit if args.limit else FULL_N[args.task]
    if effective != expected_n:
        raise SystemExit(f"ENGINEERING_FAIL {args.task} effective={effective} expected={expected_n}")
    metrics = {
        "method": "star_pro",
        "T": args.T,
        "pretrained": MODEL_PATH,
        "task": args.task,
        "effective": effective,
        "results_json": str(path),
        "exact_match": pick(block, "exact_match"),
        "pope_accuracy": pick(block, "pope_accuracy"),
        "pope_f1_score": pick(block, "pope_f1_score"),
        "pope_precision": pick(block, "pope_precision"),
        "pope_recall": pick(block, "pope_recall"),
        "mme_perception_score": pick(block, "mme_perception_score"),
        "mme_cognition_score": pick(block, "mme_cognition_score"),
        "gpt_eval_score": pick(block, "gpt_eval_score"),
        "mmmu_acc": pick(block, "mmmu_acc"),
        "llava_root": str(LLAVA_STAR),
        "env_python": sys.executable,
    }
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[ok] wrote {dest / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
