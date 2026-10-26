"""E3 layer_local vs scrub_all: decode KV lengths + short fair-256 latency.

Overrides E018_CACHE_POLICY after elva overlay (which hardcodes layer_local).
Does not change production defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
CFG = REPO / "experiments/bela_cvpr27_final_handoff_20261025/configs/p1_arms/E3_RET24_TO16.json"


def setup_env(policy: str) -> None:
    for p in (
        str(REPO),
        str(REPO / "experiments"),
        str(REPO / "experiments/elva_r3"),
        str(REPO / "LLaVA"),
        str(REPO / "e020_lib"),
        str(REPO / "lmms-eval"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
    from e014_arms import apply_arm_env
    from apply_llava import overlay

    os.environ["ELVA_R3_CONFIG"] = str(CFG)
    os.environ["E014_ARM"] = "E018_COVER_TWO32"
    overlay(apply_arm_env("E018_COVER_TWO32"))
    os.environ["E018_CACHE_POLICY"] = policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["layer_local", "scrub_all"])
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--warmups", type=int, default=4)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    setup_env(args.policy)
    assert os.environ["E018_CACHE_POLICY"] == args.policy, os.environ["E018_CACHE_POLICY"]

    import torch
    from PIL import Image
    from transformers import LogitsProcessorList

    import e006_span
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.model import e018_stage2_support as e018mod

    out = Path(os.environ["BELA_SCRUB_OUT"]) if os.environ.get("BELA_SCRUB_OUT") else (REPO / "logs/bela_scrub_ab_20261026" / f"E3_{args.policy}")
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "latency_summary.json"
    if summary_path.exists():
        print(f"[skip] {summary_path}", flush=True)
        return

    rows = []
    with PANEL.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    name = get_model_name_from_path(str(MODEL))
    tok, model, proc, _ = load_pretrained_model(
        str(MODEL),
        None,
        name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    # Stash per-layer KV lengths at end of each Runtime.forward
    kv_box: dict = {"prefill": None, "decode": None}
    _orig_fwd = e018mod.Runtime.forward

    def _fwd(self, *a, **kw):
        out_t = _orig_fwd(self, *a, **kw)
        cache = out_t.past_key_values if hasattr(out_t, "past_key_values") else None
        if cache is None:
            return out_t
        lens = []
        n_layers = len(self.model.layers)
        for i in range(n_layers):
            try:
                if hasattr(cache, "key_cache"):
                    lens.append(int(cache.key_cache[i].shape[-2]))
                else:
                    lens.append(int(cache[i][0].shape[-2]))
            except Exception:
                lens.append(-1)
        # Prefill: seq grows from 0; decode adds 1 token to each layer.
        if kv_box["prefill"] is None and max(lens) > 1:
            kv_box["prefill"] = lens
            if self.state is not None:
                kv_box["visual_n"] = int(self.state.visual_ids.numel())
                kv_box["per_layer_n_prefill"] = [int(x) for x in self.state.per_layer_n]
        elif kv_box["prefill"] is not None and kv_box["decode"] is None:
            kv_box["decode"] = lens
        return out_t

    e018mod.Runtime.forward = _fwd  # type: ignore

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
            device="cuda:0", dtype=torch.float16
        )
        context = str(row.get("eval_context") or row["question"])
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inp = tokenizer_image_token(
            prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        ids = inp.unsqueeze(0).to("cuda:0")
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        doc = {"question": str(row.get("doc_question") or row["question"])}
        start, end, *_ = e006_span.explicit_span(
            tok, prompt, inp.tolist(), doc, context, IMAGE_TOKEN_INDEX
        )
        if end <= start:
            raise RuntimeError("empty question span")
        return dict(
            input_ids=ids,
            attention_mask=ids.ne(pad_id),
            pad_token_id=pad_id,
            images=image_tensor,
            image_sizes=[image_size],
            span=(start, end),
        )

    def measure(prepared, steps: int):
        counter = StepCounter()
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        kwargs = dict(
            images=prepared["images"],
            image_sizes=prepared["image_sizes"],
            attention_mask=prepared["attention_mask"],
            pad_token_id=prepared["pad_token_id"],
            do_sample=False,
            num_beams=1,
            use_cache=True,
            min_new_tokens=steps,
            max_new_tokens=steps,
            logits_processor=LogitsProcessorList([counter]),
            e006_question_span=prepared["span"],
        )
        t0 = time.perf_counter()
        ev0.record()
        with torch.inference_mode():
            _ = model.generate(prepared["input_ids"], **kwargs)
        ev1.record()
        torch.cuda.synchronize()
        if counter.n != steps:
            raise RuntimeError(f"step mismatch {counter.n}!={steps}")
        return (time.perf_counter() - t0) * 1000.0, ev0.elapsed_time(ev1)

    # KV profile: one TTFT generate
    kv_box["prefill"] = None
    kv_box["decode"] = None
    measure(prepare(rows[0]), 1)
    kv_info = {
        "policy": args.policy,
        "E018_CACHE_POLICY": os.environ.get("E018_CACHE_POLICY"),
        "prefill_kv_len_per_layer": kv_box.get("prefill"),
        "decode_kv_len_per_layer": kv_box.get("decode"),
        "unique_prefill_lens": sorted(set(kv_box["prefill"] or [])),
        "unique_decode_lens": sorted(set(kv_box["decode"] or [])),
        "visual_n_final": kv_box.get("visual_n"),
        "per_layer_n_prefill": kv_box.get("per_layer_n_prefill"),
        "decode_kv_sum": sum(kv_box["decode"]) if kv_box.get("decode") else None,
        "prefill_kv_sum": sum(kv_box["prefill"]) if kv_box.get("prefill") else None,
    }
    (out / "kv_profile.json").write_text(json.dumps(kv_info, indent=2) + "\n")
    print("KV_PROFILE", json.dumps(kv_info), flush=True)

    for j in range(args.warmups):
        measure(prepare(rows[j % len(rows)]), 1 if j % 2 == 0 else 32)

    samples = []
    for idx, row in enumerate(rows):
        prepared = prepare(row)
        order = (1, 32) if idx % 2 == 0 else (32, 1)
        res = {}
        for steps in order:
            wall, cuda_ms = measure(prepared, steps)
            res[steps] = (wall, cuda_ms)
        samples.append(
            {
                "ttft_wall_ms": res[1][0],
                "ttft_cuda_ms": res[1][1],
                "fixed32_wall_ms": res[32][0],
                "fixed32_cuda_ms": res[32][1],
            }
        )
        if (idx + 1) % 16 == 0:
            print(f"BENCH_PROGRESS {idx+1}/{len(rows)}", flush=True)

    def stats(key):
        v = [float(x[key]) for x in samples]
        return {
            "mean": statistics.mean(v),
            "median": statistics.median(v),
            "min": min(v),
            "max": max(v),
        }

    summary = {
        "policy": args.policy,
        "n": len(samples),
        "ttft_wall_ms": stats("ttft_wall_ms"),
        "fixed32_wall_ms": stats("fixed32_wall_ms"),
        "kv_profile": kv_info,
        "note": "E3 schedule; only E018_CACHE_POLICY differs from production overlay",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "BENCH_DONE",
        args.policy,
        "TTFT",
        round(summary["ttft_wall_ms"]["mean"], 2),
        "Fixed32",
        round(summary["fixed32_wall_ms"]["mean"], 2),
        flush=True,
    )


if __name__ == "__main__":
    main()
