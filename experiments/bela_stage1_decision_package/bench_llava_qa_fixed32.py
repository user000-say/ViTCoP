#!/usr/bin/env python3
"""Fair-256 Fixed32/TTFT with QA-COVER request binding (same panel as e014 bench).

The stock benchmark_llava_e2e_anygpu.py never calls begin_question; QA-COVER
encode_images then aborts. This wrapper keeps the identical timer boundary and
only adds the request-local question ids/mask used by ACC.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path("/home/xiangyue/ViTCoP")
BENCH_PATH = REPO / "experiments/e014_latency_fair_256/scripts/benchmark_llava_e2e_anygpu.py"


def _load_bench():
    spec = importlib.util.spec_from_file_location("e014_bench", BENCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _qa_bind(tokenizer, prepared, row):
    sys.path.insert(0, str(REPO / "experiments"))
    from bela_query_anchor.question_protocol import content_mask_from_decodes
    from bela_query_anchor.request_context import begin_question
    from e006_span import explicit_span
    from llava.constants import IMAGE_TOKEN_INDEX

    prompt = prepared["prompt"]
    raw_ids = prepared["input_ids"][0].detach().cpu().tolist()
    doc = {"question": str(row.get("doc_question") or row["question"])}
    context = str(row.get("eval_context") or row["question"])
    _s, _e, raw_start, raw_end, definition, question = explicit_span(
        tokenizer, prompt, raw_ids, doc, context, IMAGE_TOKEN_INDEX
    )
    ids = prepared["input_ids"][0][raw_start:raw_end].detach().cpu().long()
    if ids.numel() and torch.any(ids < 0):
        raise RuntimeError("question span still contains an image placeholder")
    mask = None
    if os.environ.get("BELA_QA_CONTENT_MASK", "1") != "0":
        pieces = [tokenizer.decode([int(tok)], skip_special_tokens=False) for tok in ids.tolist()]
        mask = torch.tensor(content_mask_from_decodes(pieces), dtype=torch.bool)
    return begin_question(
        ids,
        mask,
        question_text=question,
        definition=definition,
        source=os.environ.get("BELA_QA_QUESTION_SOURCE", "actual"),
        sample_id=str(row.get("sample_id") or ""),
        image_id=str(row.get("image_id") or ""),
    )


def _measure(bench, tokenizer, model, prepared, row, steps):
    from bela_query_anchor.request_context import end_question

    token = _qa_bind(tokenizer, prepared, row)
    try:
        return bench.measure_generate(model, prepared, steps)
    finally:
        end_question(token)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-base", default=None)
    p.add_argument("--conv-mode", default="llava_v1")
    p.add_argument("--repo-root", type=Path, default=REPO)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--warmups", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261010)
    a = p.parse_args()

    bench = _load_bench()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expect one CUDA_VISIBLE_DEVICES id")
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical.strip() or "," in physical:
        raise RuntimeError("Expect exactly one CUDA_VISIBLE_DEVICES id")

    panel_path = a.panel.resolve()
    rows = bench.load_panel(panel_path, a.limit)
    repo = a.repo_root.resolve()
    repo_hash = bench.snapshot(repo)
    required = [
        "USE_VITCOP",
        "E014_ENABLE",
        "E006_ENABLE",
        "E012_ENABLE",
        "E014_VARIANT",
        "E014_S",
        "E014_D",
        "E006_TARGET_K",
    ]
    env = {k: os.environ.get(k, "UNSET") for k in required}
    if env["E014_ENABLE"] != "True" or env["E006_ENABLE"] != "True" or env["E012_ENABLE"] != "True":
        raise RuntimeError("QA Fixed32 requires E014/E006/E012 True")
    if os.environ.get("BELA_QA_ENTRY_MODE", "off") in ("", "off"):
        raise RuntimeError("BELA_QA_ENTRY_MODE must be on for this bench")

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok, model, proc = bench.load_model(a.model_path, a.model_base, a.conv_mode)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if (a.out_dir / "latency_samples.jsonl").exists() or (a.out_dir / "latency_summary.json").exists():
        raise FileExistsError("output already exists; use a fresh out-dir")

    for j in range(a.warmups):
        rid = rows[j % len(rows)]
        prepared = bench.prepare(rid, tok, model, proc, a.conv_mode)
        _measure(bench, tok, model, prepared, rid, 1 if j % 2 == 0 else 32)

    samples = []
    datafile = a.out_dir / "latency_samples.jsonl"
    with datafile.open("w", encoding="utf-8") as fp:
        for idx, r in enumerate(rows):
            prepared = bench.prepare(r, tok, model, proc, a.conv_mode)
            order = (1, 32) if idx % 2 == 0 else (32, 1)
            res = {}
            for count in order:
                res[count] = _measure(bench, tok, model, prepared, r, count)
            record = {
                **{k: r[k] for k in ("sample_id", "image_id", "dataset")},
                "prompt_tokens_before_image_insertion": int(prepared["input_ids"].shape[-1]),
                "prompt_sha256": hashlib.sha256(prepared["prompt"].encode()).hexdigest(),
                "e006_question_span": list(prepared["span"]) if prepared.get("span") else None,
                "ttft_wall_ms": res[1]["wall_ms"],
                "ttft_cuda_ms": res[1]["cuda_ms"],
                "fixed32_wall_ms": res[32]["wall_ms"],
                "fixed32_cuda_ms": res[32]["cuda_ms"],
                "ttft_generated_tokens": res[1]["generated_tokens"],
                "fixed32_generated_tokens": res[32]["generated_tokens"],
                "ttft_sequence_sha256": res[1]["sequence_sha256"],
                "fixed32_sequence_sha256": res[32]["sequence_sha256"],
                "peak_gpu_allocated_mb": max(
                    res[1]["peak_gpu_allocated_mb"], res[32]["peak_gpu_allocated_mb"]
                ),
                "qa_bound": True,
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            fp.flush()
            samples.append(record)
            if (idx + 1) % 16 == 0:
                print(f"BENCH_PROGRESS {idx + 1}/{len(rows)}", flush=True)

    def stats(key):
        v = [float(x[key]) for x in samples]
        return {
            "mean": statistics.mean(v),
            "median": statistics.median(v),
            "p90": bench.quantile(v, 0.9),
            "min": min(v),
            "max": max(v),
        }

    summary = dict(
        status="COMPLETE_RAW_GPU",
        n=len(samples),
        scope="model_generate_preprocessed_qa_bound",
        panel_sha256=bench.sha256(panel_path),
        panel_path=str(panel_path),
        model_path=a.model_path,
        model_base=a.model_base,
        conv_mode=a.conv_mode,
        warmups=a.warmups,
        seed=a.seed,
        gpu_visible_index=0,
        gpu_physical_id=physical.strip(),
        gpu_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        source=repo_hash,
        env=env,
        bela_qa_entry_mode=os.environ.get("BELA_QA_ENTRY_MODE"),
        first_selector_hint=os.environ.get("E020_KIND") or os.environ.get("E018_SELECTOR"),
        ttft_wall_ms=stats("ttft_wall_ms"),
        ttft_cuda_ms=stats("ttft_cuda_ms"),
        fixed32_wall_ms=stats("fixed32_wall_ms"),
        fixed32_cuda_ms=stats("fixed32_cuda_ms"),
        peak_gpu_allocated_mb=stats("peak_gpu_allocated_mb"),
        generated_tokens=32,
        definition="TTFT=generate(1); Fixed32=generate(32); QA begin_question bound per call",
        warning="QA-COVER path; compare only against other qa_bound runs on same GPU/panel.",
    )
    (a.out_dir / "latency_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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
