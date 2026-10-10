#!/usr/bin/env python3
"""E013 latency on one physical GPU. Timed region = raw.generate (same as e012/e010)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
ROOT = REPO / "logs/e013_p1"
sys.path.insert(0, str(REPO))

import e010_common
from e010_common import sha256
from e013_arms import ARMS, apply_arm_env

e010_common.ROOT = ROOT
e010_common.RUNTIME = ROOT / "code_runtime"
e010_common.bind()
_support = "LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py"
_dst = e010_common.RUNTIME / _support
_dst.parent.mkdir(parents=True, exist_ok=True)
if not _dst.exists() or sha256(_dst) != sha256(REPO / _support):
    shutil.copy2(REPO / _support, _dst)

from e009_common import apply_profile, hard, load_model, panel_lookup
from e010_common import set_e010

MODEL_SHA = "4481d270cc22fd5c4d1bb5df129622006ccd9234"
CODE = REPO / "LLaVA/llava/model/language_model/vitcop_modeling_llama.py"
# Crossover uses E012 paths with E013 off; LEGACY disables E012.
CROSSOVER_SPECIAL = {"LEGACY_L12", "E012_S12D12"}


def probe(gpu: int) -> tuple[str, str]:
    text = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader"],
        text=True,
    ).strip()
    temp, power = [part.strip() for part in text.split(",")]
    return temp, power


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def force(arm: str) -> dict:
    apply_profile("ANCHOR_A", audit=False, strict=True)
    enable, variant, pick, b, s, d = ARMS[arm]
    set_e010(enabled=True, layer=s)
    if arm == "LEGACY_L12":
        os.environ.update(
            E012_ENABLE="False",
            E012_DELETE_LAYER="12",
            E013_ENABLE="False",
            E006_SELECTOR="QED",
            E007_ENABLE="True",
            E012_DIAG="False",
            E012_KV_AUDIT="False",
            E013_AUDIT="False",
            E013_STRICT="False",
            COMPUTE_EFFICIENCY="False",
            E006_PROFILE="False",
            E006_DEBUG="False",
        )
        import e006_span

        e006_span.STAGE1_VISUAL = 172
        return {"arm": arm, "enable": False, "S": 12, "D": 12, "visual": 172, "b": 0, "variant": "EXTRA", "pick": "ATTN"}
    if arm == "E012_S12D12":
        os.environ.update(
            E012_ENABLE="True",
            E012_DELETE_LAYER="12",
            E013_ENABLE="False",
            E006_SELECTOR="QED",
            E007_ENABLE="True",
            E012_DIAG="False",
            E012_KV_AUDIT="False",
            E013_AUDIT="False",
            E013_STRICT="False",
            COMPUTE_EFFICIENCY="False",
            E006_PROFILE="False",
            E006_DEBUG="False",
        )
        import e006_span

        e006_span.STAGE1_VISUAL = 172
        return {"arm": arm, "enable": False, "S": 12, "D": 12, "visual": 172, "b": 0, "variant": "EXTRA", "pick": "ATTN"}
    meta = apply_arm_env(arm)
    os.environ["E013_AUDIT"] = "False"
    os.environ["E013_STRICT"] = "False"
    os.environ["E012_KV_AUDIT"] = "False"
    import e006_span

    e006_span.STAGE1_VISUAL = meta["visual"]
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=tuple(ARMS))
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--tag", default="", help="optional output suffix for ABBA repeats")
    args = ap.parse_args()
    if "RANDOM" in args.arm and args.arm not in CROSSOVER_SPECIAL:
        # Accuracy negative control: allow but label; handoff asks not to treat as Pareto main.
        pass
    meta = force(args.arm)
    label = args.arm if not args.tag else f"{args.arm}_{args.tag}"
    dest = ROOT / "latency" / label
    dest.mkdir(parents=True, exist_ok=True)
    merged = dest / "latency.jsonl"
    if merged.exists() and len(read_jsonl(merged)) == 128 * 5:
        print(f"[latency skip] {merged}", flush=True)
        return
    uuid, eng, lm, raw, core, torch = load_model(args.gpu)
    import e006_lmms_patch as patch
    from e006_lmms_patch import install

    patch._audit = lambda raw_model: None
    install()
    lookup, keys = panel_lookup(eng, lm)
    if len(keys) != 128:
        hard(f"panel {len(keys)}")
    dtype = str(next(raw.parameters()).dtype)
    code_sha = sha256(CODE)
    original = raw.generate
    expect_vis = meta["visual"]

    def run_request(req, new_tokens: int):
        holder = {}

        def wrapped(*ga, **gk):
            kw = dict(gk)
            kw.update(
                do_sample=False, num_beams=1, use_cache=True,
                min_new_tokens=new_tokens, max_new_tokens=new_tokens, temperature=0.0,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            answer = original(*ga, **kw)
            torch.cuda.synchronize()
            holder["elapsed_ms"] = (time.perf_counter_ns() - t0) / 1e6
            holder["seq_len"] = int(answer.shape[-1]) if hasattr(answer, "shape") else -1
            return answer

        raw.generate = wrapped
        try:
            pred = lm.generate_until([req])
        finally:
            raw.generate = original
        if not pred or not str(pred[0]).strip() or "elapsed_ms" not in holder:
            raise RuntimeError("timer or answer missing")
        seq_len = int(holder["seq_len"])
        generated = seq_len - 1 if seq_len == new_tokens + 1 else seq_len
        last = getattr(core, "e006_last", {}) or {}
        if generated != new_tokens:
            hard(f"generated={generated} seq={seq_len} expected={new_tokens}")
        if int(last.get("prefill_count", -1)) != 1 or int(last.get("decode_count", -1)) != 0:
            hard(f"selector prefill={last.get('prefill_count')} decode={last.get('decode_count')}")
        if int(getattr(core, "e001_last_final_visual_tokens", -1)) != 96:
            hard(f"final={getattr(core, 'e001_last_final_visual_tokens', None)}")
        stage1 = int(getattr(core, "e001_last_stage1_visual_tokens", -1))
        if stage1 != expect_vis:
            hard(f"stage1={stage1} expected={expect_vis}")
        return holder["elapsed_ms"], generated

    def measure(new_tokens: int, path: Path) -> None:
        done = {(row["sample_id"], int(row["repeat"])) for row in read_jsonl(path)}
        if len(done) == 128 * 5:
            print(f"[latency resume] {path.name} complete", flush=True)
            return
        for index in range(20):
            run_request(lookup[keys[index % len(keys)]], new_tokens)
            print(f"[warmup] {label} n{new_tokens} {index + 1}/20", flush=True)
        for repeat_id in range(5):
            for key in keys:
                sample_id = f"{key[0]}:{key[1]}:{key[2]}"
                if (sample_id, repeat_id) in done:
                    continue
                temp, power = probe(args.gpu)
                elapsed, generated = run_request(lookup[key], new_tokens)
                append_jsonl(path, {
                    "sample_id": sample_id,
                    "repeat": repeat_id,
                    "elapsed_ms": elapsed,
                    "generated_tokens": generated,
                    "temperature_c": temp,
                    "power_w": power,
                })
            print(f"[latency] {label} n{new_tokens} repeat {repeat_id + 1}/5", flush=True)

    measure(1, dest / "ttft.jsonl")
    measure(32, dest / "fixed32.jsonl")
    ttft = {(row["sample_id"], int(row["repeat"])): row for row in read_jsonl(dest / "ttft.jsonl")}
    fixed = {(row["sample_id"], int(row["repeat"])): row for row in read_jsonl(dest / "fixed32.jsonl")}
    if len(ttft) != 640 or len(fixed) != 640:
        hard(f"incomplete ttft={len(ttft)} fixed32={len(fixed)}")
    lines = []
    order = 0
    for repeat_id in range(5):
        for key in keys:
            sample_id = f"{key[0]}:{key[1]}:{key[2]}"
            left = ttft[(sample_id, repeat_id)]
            right = fixed[(sample_id, repeat_id)]
            if int(left["generated_tokens"]) != 1 or int(right["generated_tokens"]) != 32:
                hard(f"token count {sample_id}")
            lines.append({
                "sample_id": sample_id,
                "repeat": repeat_id,
                "warmup": False,
                "ttft_ms": left["elapsed_ms"],
                "fixed32_ms": right["elapsed_ms"],
                "generated_tokens": 32,
                "ttft_generated_tokens": 1,
                "gpu_uuid": uuid,
                "input_policy": "model_inference",
                "dtype": dtype,
                "batch_size": 1,
                "arm": args.arm,
                "tag": args.tag,
                "selection_layer": meta["S"],
                "deletion_layer": meta["D"],
                "e013": {k: meta[k] for k in ("enable", "variant", "pick", "b", "visual")},
                "model_sha": MODEL_SHA,
                "code_sha": code_sha,
                "order": order,
                "temperature_c": right["temperature_c"],
                "power_w": right["power_w"],
            })
            order += 1
    merged.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in lines))
    print(f"[latency done] {merged}", flush=True)


if __name__ == "__main__":
    main()
