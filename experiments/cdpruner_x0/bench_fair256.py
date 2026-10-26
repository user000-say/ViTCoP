"""CDPruner K32/K64 on canonical fair-256 panel (match BELA/STAR protocol)."""
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
    "CDPRUNER_MODEL_PATH",
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
    at = (len(v) - 1) * q
    lo = int(at)
    hi = min(len(v) - 1, lo + 1)
    return v[lo] + (v[hi] - v[lo]) * (at - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual-token-num", type=int, required=True, choices=[32, 64])
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--warmups", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    sys.path = [
        str(REPO / "third_party/CDPruner"),
        str(REPO / "experiments/cdpruner_x0"),
        str(REPO / "lmms-eval"),
    ] + [p for p in sys.path if "ViTCoP/llava" not in p and p != str(REPO / "llava")]

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    out = REPO / "logs/cdpruner_x0/fair256_ttft" / f"CDPRUNER_K{args.visual_token_num}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latency_summary.json").exists():
        print(f"[skip] {out / 'latency_summary.json'}", flush=True)
        return

    rows = []
    with PANEL.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok, model, proc, _ = load_pretrained_model(
        MODEL_PATH, None, "llava-v1.5-7b", device_map="cuda:0",
        visual_token_num=args.visual_token_num,
    )
    model.eval()

    class StepCounter:
        def __init__(self):
            self.n = 0

        def __call__(self, input_ids, scores):
            self.n += 1
            return scores

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
        return dict(
            input_ids=ids,
            attention_mask=ids.ne(pad_id),
            pad_token_id=pad_id,
            images=image_tensor,
            image_sizes=[image_size],
            texts=qtext,
        )

    def measure(prepared, steps):
        counter = StepCounter()
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        ev0.record()
        with torch.inference_mode():
            _ = model.generate(
                prepared["input_ids"],
                images=prepared["images"],
                image_sizes=prepared["image_sizes"],
                texts=prepared["texts"],
                attention_mask=prepared["attention_mask"],
                pad_token_id=prepared["pad_token_id"],
                do_sample=False,
                temperature=0,
                num_beams=1,
                use_cache=True,
                min_new_tokens=steps,
                max_new_tokens=steps,
                logits_processor=LogitsProcessorList([counter]),
            )
        ev1.record()
        torch.cuda.synchronize()
        if counter.n != steps:
            raise RuntimeError(f"step mismatch {counter.n}")
        return (time.perf_counter() - t0) * 1000.0, ev0.elapsed_time(ev1)

    for j in range(args.warmups):
        measure(prepare(rows[j % len(rows)]), 1 if j % 2 == 0 else 32)

    samples = []
    with (out / "latency_samples.jsonl").open("w") as fp:
        for idx, row in enumerate(rows):
            prepared = prepare(row)
            order = (1, 32) if idx % 2 == 0 else (32, 1)
            res = {c: measure(prepared, c) for c in order}
            rec = {
                "sample_id": row["sample_id"],
                "ttft_wall_ms": res[1][0],
                "ttft_cuda_ms": res[1][1],
                "fixed32_wall_ms": res[32][0],
                "fixed32_cuda_ms": res[32][1],
            }
            fp.write(json.dumps(rec) + "\n")
            samples.append(rec)
            if (idx + 1) % 16 == 0:
                print(f"BENCH_PROGRESS {idx+1}/{len(rows)}", flush=True)

    def stats(key):
        v = [float(x[key]) for x in samples]
        return {"mean": statistics.mean(v), "median": statistics.median(v),
                "p90": quantile(v, 0.9), "min": min(v), "max": max(v)}

    summary = {
        "status": "COMPLETE_RAW_GPU",
        "method": "cdpruner",
        "visual_token_num": args.visual_token_num,
        "n": len(samples),
        "panel_sha256": sha256_file(PANEL),
        "warmups": args.warmups,
        "gpu_name": torch.cuda.get_device_name(0),
        "ttft_wall_ms": stats("ttft_wall_ms"),
        "fixed32_wall_ms": stats("fixed32_wall_ms"),
        "definition": "TTFT=generate(1); Fixed32=generate(32); preprocess outside timer",
    }
    (out / "latency_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("BENCH_DONE", args.visual_token_num,
          "TTFT", round(summary["ttft_wall_ms"]["mean"], 2),
          "Fixed32", round(summary["fixed32_wall_ms"]["mean"], 2), flush=True)


if __name__ == "__main__":
    main()
