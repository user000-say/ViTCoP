#!/usr/bin/env python3
"""One E010 full-split accuracy job. QED formula is unchanged; only the layer index moves."""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

from e010_common import bind, gpu4_claim_owner, set_e010

bind()
from e009_common import FULL_N, TASKS, UUIDS, apply_profile, configure_runtime_constants, hard

ap = argparse.ArgumentParser()
ap.add_argument("--profile", required=True)
ap.add_argument("--task", required=True, choices=TASKS)
ap.add_argument("--output", required=True)
ap.add_argument("--gpu-index", type=int, required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--exit-layer", type=int, default=-1)
ap.add_argument("--keep-context", choices=("True", "False"), default="False")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--context-mode", default="E009")
ap.add_argument("--full576", choices=("True", "False"), default="False")
ap.add_argument("--side-only", choices=("True", "False"), default="False")
args = ap.parse_args()

apply_profile(args.profile, audit=False, strict=True)
set_e010(
    enabled=True,
    layer=args.layer,
    exit_layer=args.exit_layer,
    keep=args.keep_context == "True",
    full576=args.full576 == "True",
    side_only=args.side_only == "True",
    context_mode=args.context_mode,
)
if args.full576 == "True":
    import e006_span
    e006_span.STAGE1_VISUAL = 576
    os.environ["E007_ENABLE"] = "False"
configure_runtime_constants()
dest = Path(args.output)
if (dest / "metrics.json").exists():
    print(f"[eval skip] {dest / 'metrics.json'}", flush=True)
    raise SystemExit(0)
while True:
    owner = gpu4_claim_owner(str(dest))
    if not owner:
        break
    if (dest / "metrics.json").exists():
        print(f"[eval skip] {dest / 'metrics.json'}", flush=True)
        raise SystemExit(0)
    print(f"[eval wait] {dest} owner={owner}", flush=True)
    time.sleep(30)
from e006_model import assert_physical_gpu, assert_runtime_hash, prepare_paths

assert_physical_gpu(args.gpu_index, UUIDS[args.gpu_index])
assert_runtime_hash()
prepare_paths()
from e003_a_offline_dataset_adapter_v1_20260923 import install
from e006_lmms_patch import install as install_span
import e006_lmms_patch as patch

install()


def _audit(raw_model) -> None:
    core = raw_model.get_model()
    last = getattr(core, "e006_last", {}) or {}
    reasons = []
    expected = int(os.environ["E009_EXPECT_VISUAL"])
    target_k = int(os.environ["E006_TARGET_K"])
    layers = int(os.environ["E006_EXPECT_LAYERS"])
    stage1_raw = getattr(core, "e001_last_stage1_visual_tokens", -1)
    stage1 = int(stage1_raw) if stage1_raw is not None else -1
    full576 = os.environ.get("E010_ANALYSIS_FULL576") == "True"
    side_only = os.environ.get("E010_QED_SIDE_ONLY") == "True"
    e018 = os.environ.get("E018_ENABLE", "False") == "True"
    if not full576 and stage1 != expected:
        reasons.append(f"stage1={stage1}")
    if int(core.config.num_hidden_layers) != layers:
        reasons.append(f"layers={core.config.num_hidden_layers}")
    if core.config._attn_implementation != "sdpa":
        reasons.append(f"attention={core.config._attn_implementation}")
    if bool(core.config.output_attentions):
        reasons.append("output_attentions")
    if core.stage3_policy != "skip":
        reasons.append(f"stage3={core.stage3_policy}")
    final_raw = getattr(core, "e001_last_final_visual_tokens", -1)
    final = int(final_raw) if final_raw is not None else -1
    if not side_only and final != target_k:
        reasons.append(f"final={final}")
    visual_in = int(last.get("visual_in") or -1)
    exit_layer = int(os.environ.get("E010_CONTEXT_EXIT_LAYER", "-1"))
    qed_layer = int(os.environ["E010_QED_LAYER"])
    # Context removed before QED changes the count that reaches the selector.
    expected_visual = expected
    if exit_layer >= 0 and exit_layer < qed_layer:
        expected_visual = expected - 16 if expected > 172 else expected
    if not side_only and visual_in != expected_visual:
        reasons.append(f"visual_entering_qed={visual_in} expected={expected_visual}")
    if e018:
        # Isolated Stage2 path: require explicit span + SDPA; skip legacy QED counters.
        if last.get("span_source") != "explicit":
            reasons.append(f"span_source={last.get('span_source')}")
        if not bool(last.get("sdpa_preserved", False)):
            reasons.append("sdpa_not_preserved")
        if int(last.get("e018_final_k", -1)) != target_k:
            reasons.append(f"e018_final_k={last.get('e018_final_k')}")
    else:
        if not side_only and (int(last.get("prefill_count", -1)) != 1 or int(last.get("decode_count", -1)) != 0):
            reasons.append(f"prefill={last.get('prefill_count')} decode={last.get('decode_count')}")
        if not side_only and int(last.get("e010_qed_layer", -1)) != qed_layer:
            reasons.append(f"qed_layer={last.get('e010_qed_layer')}")
        if not side_only and last.get("span_source") != "explicit":
            reasons.append(f"span_source={last.get('span_source')}")
        if not side_only and not bool(last.get("sdpa_preserved", False)):
            reasons.append("sdpa_not_preserved")
    if reasons:
        patch._hard("; ".join(reasons))
    patch._JOB_AUDIT["requests_audited"] += 1
    patch._JOB_AUDIT["stage1_tokens"] = stage1
    patch._JOB_AUDIT["visual_entering_layer12"] = visual_in
    patch._JOB_AUDIT["final_visual_tokens"] = target_k
    patch._JOB_AUDIT["question_span_source"] = last.get("span_source")
    patch._JOB_AUDIT["selector_prefill_count"] = int(last.get("prefill_count", -1))
    patch._JOB_AUDIT["selector_decode_count"] = int(last.get("decode_count", -1))
    patch._JOB_AUDIT["num_hidden_layers"] = layers
    patch._JOB_AUDIT["attention_backend"] = "sdpa"
    patch._JOB_AUDIT["stage3_policy"] = str(core.stage3_policy)
    patch._JOB_AUDIT["e010_qed_layer"] = qed_layer
    if last.get("e018_pre_block_counts"):
        patch._JOB_AUDIT["e018_pre_block_counts"] = list(last["e018_pre_block_counts"])
        patch._JOB_AUDIT["e018_events"] = last.get("e018_events")
        patch._JOB_AUDIT["e018_final_k"] = last.get("e018_final_k")


patch._audit = _audit
install_span()

dest.mkdir(parents=True, exist_ok=True)
if (dest / "metrics.json").exists():
    print(f"[eval skip] {dest / 'metrics.json'}", flush=True)
    raise SystemExit(0)
argv = [
    "lmms_eval", "--model", "llava",
    "--model_args", f"pretrained={os.environ['E009_PRETRAINED']}",
    "--tasks", args.task, "--batch_size", "1", "--log_samples",
    "--output_path", str(dest),
]
if args.limit:
    argv.extend(["--limit", str(args.limit)])
sys.argv = argv
print("[eval] " + " ".join(argv), flush=True)
try:
    runpy.run_module("lmms_eval", run_name="__main__")
except SystemExit as exc:
    code = exc.code
    if code not in (0, None):
        print(f"ENGINEERING_FAIL lmms exit {code}", flush=True)
        raise SystemExit(1 if code != 99 else 99)


def pick(block, name):
    for key, value in block.items():
        if key.split(",")[0] == name and isinstance(value, (int, float)):
            return float(value)
    return None


files = [p for p in dest.rglob("*_results.json") if "submissions" not in p.parts]
if not files:
    print("ENGINEERING_FAIL results json missing", flush=True)
    raise SystemExit(1)
path = max(files, key=lambda item: item.stat().st_mtime)
payload = json.loads(path.read_text())
block = payload["results"][args.task]
effective = int(payload.get("n-samples", {}).get(args.task, {}).get("effective", -1))
expected_n = args.limit if args.limit else FULL_N[args.task]
if effective != expected_n:
    hard(f"{args.task} effective={effective} expected={expected_n}")
metrics = {
    "profile": args.profile,
    "layer": args.layer,
    "pretrained": os.environ["E009_PRETRAINED"],
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
}
required = {
    "gqa": ("exact_match",),
    "pope": ("pope_accuracy", "pope_f1_score", "pope_precision", "pope_recall"),
    "mme": ("mme_perception_score", "mme_cognition_score"),
    "mmbench_en_dev": ("gpt_eval_score",),
    "mmbench_cn_dev": ("gpt_eval_score",),
    "scienceqa_img": ("exact_match",),
    "textvqa_val": ("exact_match",),
    "textvqa_val_ocr": ("exact_match",),
    "vqav2_val": ("exact_match",),
    "vqav2_val_lite": ("exact_match",),
    "mmmu_val": ("mmmu_acc",),
    "ai2d": ("exact_match",),
}[args.task]
if any(metrics[name] is None for name in required):
    print(f"ENGINEERING_FAIL missing metrics {metrics}", flush=True)
    raise SystemExit(1)
audit = patch.job_audit()
if int(audit.get("requests_audited") or 0) != effective:
    print(f"ENGINEERING_FAIL audited {audit.get('requests_audited')} != {effective}", flush=True)
    raise SystemExit(1)
if int(audit.get("e010_qed_layer") or -1) != args.layer:
    hard(f"layer audit {audit}")
metrics["audit"] = audit
(dest / "metrics.json").write_text(json.dumps(metrics, indent=2))
(dest / "audit.json").write_text(json.dumps(audit, indent=2))
print("[eval done] " + json.dumps({k: v for k, v in metrics.items() if k != "audit"}), flush=True)
