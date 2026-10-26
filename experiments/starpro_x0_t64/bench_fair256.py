"""STAR-Pro T64/T32 on the canonical fair-256 latency panel (match BELA E3 protocol).

Same panel, separate TTFT(1) vs Fixed32(32) generates, CUDA Event + wall,
preprocess outside timer. Does NOT use E014 verify (STAR fork).

Writes logs/starpro_x0_t64/fair256_ttft/STARPRO_T{T}/latency_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import LogitsProcessorList

REPO = Path("/home/xiangyue/ViTCoP")
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL_PATH = os.environ.get(
    "STARPRO_MODEL_PATH",
    "/home/xiangyue/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantile(vals, q):
    v = sorted(vals)
    if not v:
        return float("nan")
    at = (len(v) - 1) * q
    lo = int(at)
    hi = min(len(v) - 1, lo + 1)
    return v[lo] + (v[hi] - v[lo]) * (at - lo)


def load_panel(path: Path, limit: int):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not Path(r["image"]).is_file():
                raise FileNotFoundError(r["image"])
            rows.append(r)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError("empty panel")
    return rows


class StepCounter:
    def __init__(self):
        self.n = 0

    def __call__(self, input_ids, scores):
        self.n += 1
        return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=64, choices=[32, 64])
    ap.add_argument("--limit", type=int, default=256, help="0 = full panel")
    ap.add_argument("--warmups", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    sys.path = [
        str(REPO / "third_party/LLaVA-starpro"),
        str(REPO / "experiments/starpro_x0_t64"),
        str(REPO / "lmms-eval"),
    ] + [p for p in sys.path if "ViTCoP/llava" not in p and p != str(REPO / "llava")]

    os.environ.setdefault("STAGE1_SCORER", "qr")
    os.environ.setdefault("STAGE1_MULT", "2")
    os.environ.setdefault("TEXT_AGG_MODE", "average_all")
    os.environ.setdefault("STAGE2_SELECTOR", "topk")
    os.environ.setdefault("PRUNING_SCHEDULE_MODE", "progressive")
    os.environ.setdefault("STAGE2_ANCHOR_M", "0")
    os.environ.setdefault("STAR_CAUSAL_FIX", "0")
    os.environ.setdefault("STAR_KEEP_POSIDS", "0")

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    out = REPO / "logs/starpro_x0_t64/fair256_ttft" / f"STARPRO_T{args.T}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latency_summary.json").exists():
        print(f"[skip] {out / 'latency_summary.json'}", flush=True)
        return

    if torch.cuda.device_count() != 1:
        raise RuntimeError("expect one visible GPU")

    rows = load_panel(PANEL, args.limit)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok, model, proc, _ = load_pretrained_model(
        MODEL_PATH,
        None,
        "llava-v1.5-7b",
        device_map="cuda:0",
        pruning_method="star_pro",
        visual_token_num=args.T,
        use_star=True,
        star_config={"T": args.T, "mode": "star_pro"},
        use_text_tower=True,
    )
    model.eval()
    attn_impl = getattr(model.config, "_attn_implementation", None)

    def prepare(row):
        with Image.open(row["image"]) as src:
            image = src.convert("RGB")
            image_size = tuple(src.size)
        image_tensor = process_images([image], proc, model.config).to(
            dtype=torch.float16, device="cuda:0"
        )
        context = str(row.get("eval_context") or row["question"])
        qtext = context.replace("\nAnswer the question using a single word or phrase.", "")
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ids = tokenizer_image_token(
            prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to("cuda:0")
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        attention_mask = ids.ne(pad_id)
        return dict(
            input_ids=ids,
            attention_mask=attention_mask,
            pad_token_id=pad_id,
            image_tensor=image_tensor,
            image_sizes=[image_size],
            texts=qtext,
            prompt=prompt,
        )

    def measure(prepared, steps: int):
        counter = StepCounter()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        kwargs = dict(
            images=prepared["image_tensor"],
            image_sizes=prepared["image_sizes"],
            texts=prepared["texts"],
            attention_mask=prepared["attention_mask"],
            pad_token_id=prepared["pad_token_id"],
            do_sample=False,
            num_beams=1,
            use_cache=True,
            min_new_tokens=steps,
            max_new_tokens=steps,
            logits_processor=LogitsProcessorList([counter]),
            temperature=0,
        )
        t0 = time.perf_counter()
        ev0.record()
        with torch.inference_mode():
            out_ids = model.generate(prepared["input_ids"], **kwargs)
        ev1.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0
        cuda_ms = ev0.elapsed_time(ev1)
        if counter.n != steps:
            raise RuntimeError(f"step mismatch {counter.n}!={steps}")
        _ = out_ids
        return dict(
            wall_ms=wall,
            cuda_ms=cuda_ms,
            peak_gpu_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
            generated_tokens=counter.n,
        )

    for j in range(args.warmups):
        measure(prepare(rows[j % len(rows)]), 1 if j % 2 == 0 else 32)
        print(f"[warmup] {j+1}/{args.warmups}", flush=True)

    samples = []
    with (out / "latency_samples.jsonl").open("w", encoding="utf-8") as fp:
        for idx, row in enumerate(rows):
            prepared = prepare(row)
            order = (1, 32) if idx % 2 == 0 else (32, 1)
            res = {c: measure(prepared, c) for c in order}
            rec = {
                "sample_id": row["sample_id"],
                "dataset": row.get("dataset"),
                "ttft_wall_ms": res[1]["wall_ms"],
                "ttft_cuda_ms": res[1]["cuda_ms"],
                "fixed32_wall_ms": res[32]["wall_ms"],
                "fixed32_cuda_ms": res[32]["cuda_ms"],
                "peak_gpu_allocated_mb": max(
                    res[1]["peak_gpu_allocated_mb"], res[32]["peak_gpu_allocated_mb"]
                ),
            }
            fp.write(json.dumps(rec) + "\n")
            samples.append(rec)
            if (idx + 1) % 16 == 0:
                print(f"BENCH_PROGRESS {idx+1}/{len(rows)}", flush=True)

    def stats(key):
        v = [float(x[key]) for x in samples]
        return {
            "mean": statistics.mean(v),
            "median": statistics.median(v),
            "p90": quantile(v, 0.9),
            "min": min(v),
            "max": max(v),
        }

    summary = {
        "status": "COMPLETE_RAW_GPU",
        "method": "star_pro",
        "T": args.T,
        "n": len(samples),
        "scope": "model_generate_preprocessed",
        "panel_sha256": sha256_file(PANEL),
        "panel_path": str(PANEL),
        "model_path": MODEL_PATH,
        "warmups": args.warmups,
        "seed": args.seed,
        "gpu_physical_id": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "attn_implementation": attn_impl,
        "ttft_wall_ms": stats("ttft_wall_ms"),
        "ttft_cuda_ms": stats("ttft_cuda_ms"),
        "fixed32_wall_ms": stats("fixed32_wall_ms"),
        "fixed32_cuda_ms": stats("fixed32_cuda_ms"),
        "definition": "TTFT is separate one-token generate walltime; Fixed32 is separate 32-token generate walltime",
        "note": "Matched fair-256 panel vs BELA E3; STAR fork + starpro env. Compare to E3 latency_summary.json.",
        "prior_synthetic_bench": "logs/starpro_x0_t64/ttft/STARPRO_T64 (NOT comparable to fair-256)",
    }
    (out / "latency_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "BENCH_DONE",
        len(samples),
        "TTFT",
        round(summary["ttft_wall_ms"]["median"], 3),
        "Fixed32",
        round(summary["fixed32_wall_ms"]["median"], 3),
        flush=True,
    )


if __name__ == "__main__":
    main()
