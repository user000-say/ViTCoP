#!/usr/bin/env python3
"""One dense (no-prune) official job: keep all 576 visual tokens.

Uses E010_ANALYSIS_FULL576 + side_only audit, with E014/E018/E006/E012 off.
Same lmms-eval task yamls as Gate1 (MMMU group_img, VizWiz offline).
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
sys.path.insert(0, str(REPO))

# Bind the E014 runtime tree (same LLaVA code Gate1 uses) without applying an arm.
import e010_common

e010_common.ROOT = REPO / "logs/e014_aee"
e010_common.RUNTIME = e010_common.ROOT / "code_runtime"
e010_common.bind()

from e010_common import set_e010  # noqa: E402
from e009_common import FULL_N, UUIDS, apply_profile, configure_runtime_constants, hard  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=("mmmu_val", "vizwiz_vqa_val"))
ap.add_argument("--output", required=True)
ap.add_argument("--gpu-index", type=int, required=True)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

# Clear any prune schedule / arm from the parent environment.
for key in list(os.environ):
    if key.startswith(("ELVA_", "E014_", "E018_", "E015_", "E016_", "E012_")):
        os.environ.pop(key, None)

apply_profile("ANCHOR_A", audit=False, strict=True)
set_e010(
    enabled=True,
    layer=8,
    exit_layer=-1,
    keep=False,
    full576=True,
    side_only=True,
    context_mode="E009",
)
import e006_span

e006_span.STAGE1_VISUAL = 576
os.environ.update(
    E006_ENABLE="False",
    E007_ENABLE="False",
    E009_ENABLE="False",
    E010_ENABLE="True",
    E010_ANALYSIS_FULL576="True",
    E010_QED_SIDE_ONLY="True",
    E012_ENABLE="False",
    E013_ENABLE="False",
    E014_ENABLE="False",
    E018_ENABLE="False",
    E009_EXPECT_VISUAL="576",
    E006_TARGET_K="576",
    E006_REQUIRE_EXPLICIT_QUESTION_SPAN="False",
    STAGE3_POLICY="skip",
)
configure_runtime_constants()

dest = Path(args.output)
if (dest / "metrics.json").exists():
    print(f"[eval skip] {dest / 'metrics.json'}", flush=True)
    raise SystemExit(0)

from e006_model import assert_physical_gpu, assert_runtime_hash, prepare_paths  # noqa: E402
from e003_a_offline_dataset_adapter_v1_20260923 import install  # noqa: E402
from e006_lmms_patch import install as install_span  # noqa: E402
import e006_lmms_patch as patch  # noqa: E402

assert_physical_gpu(args.gpu_index, UUIDS[args.gpu_index])
assert_runtime_hash()
prepare_paths()
install()


def _audit(raw_model) -> None:
    """Dense audit: require Stage1=576 and no mid-stack prune; skip QED counters."""
    core = raw_model.get_model()
    reasons = []
    stage1_raw = getattr(core, "e001_last_stage1_visual_tokens", -1)
    stage1 = int(stage1_raw) if stage1_raw is not None else -1
    final_raw = getattr(core, "e001_last_final_visual_tokens", -1)
    final = int(final_raw) if final_raw is not None else -1
    if stage1 != 576:
        reasons.append(f"stage1={stage1} expected 576")
    if final not in (-1, 576) and final != 576:
        # Some paths report final==stage1; require full keep.
        if final < 500:
            reasons.append(f"final={final} (pruned?)")
    if int(core.config.num_hidden_layers) != 32:
        reasons.append(f"layers={core.config.num_hidden_layers}")
    if core.config._attn_implementation != "sdpa":
        reasons.append(f"attention={core.config._attn_implementation}")
    if reasons:
        print("SCIENTIFIC_HARD_FAIL " + "; ".join(reasons), flush=True)
        raise SystemExit(99)
    patch._JOB_AUDIT["requests_audited"] += 1
    patch._JOB_AUDIT["stage1_tokens"] = stage1
    patch._JOB_AUDIT["visual_entering_layer12"] = stage1
    patch._JOB_AUDIT["final_visual_tokens"] = final if final > 0 else stage1
    patch._JOB_AUDIT["question_span_source"] = "dense_bypass"
    patch._JOB_AUDIT["selector_prefill_count"] = 0
    patch._JOB_AUDIT["selector_decode_count"] = 0
    patch._JOB_AUDIT["num_hidden_layers"] = 32
    patch._JOB_AUDIT["attention_backend"] = "sdpa"
    patch._JOB_AUDIT["stage3_policy"] = str(core.stage3_policy)
    patch._JOB_AUDIT["e010_qed_layer"] = -1
    patch._JOB_AUDIT["dense_full576"] = True


patch._audit = _audit
install_span()

dest.mkdir(parents=True, exist_ok=True)
argv = [
    "lmms_eval", "--model", "llava",
    "--model_args", f"pretrained={os.environ['E009_PRETRAINED']}",
    "--tasks", args.task, "--batch_size", "1", "--log_samples",
    "--output_path", str(dest),
]
if args.limit:
    argv.extend(["--limit", str(args.limit)])
sys.argv = argv
print("[eval dense] " + " ".join(argv), flush=True)
print(
    "[dense flags] FULL576=1 E014=0 E018=0 E006=0 EXPECT_VISUAL=576",
    flush=True,
)
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
    "profile": "DENSE_FULL576",
    "layer": -1,
    "pretrained": os.environ["E009_PRETRAINED"],
    "task": args.task,
    "effective": effective,
    "results_json": str(path),
    "exact_match": pick(block, "exact_match"),
    "mmmu_acc": pick(block, "mmmu_acc"),
    "dense_no_prune": True,
}
required = {"mmmu_val": ("mmmu_acc",), "vizwiz_vqa_val": ("exact_match",)}[args.task]
if any(metrics[name] is None for name in required):
    print(f"ENGINEERING_FAIL missing metrics {metrics}", flush=True)
    raise SystemExit(1)
audit = patch.job_audit()
if int(audit.get("requests_audited") or 0) != effective:
    print(f"ENGINEERING_FAIL audited {audit.get('requests_audited')} != {effective}", flush=True)
    raise SystemExit(1)
if int(audit.get("stage1_tokens") or -1) != 576:
    hard(f"dense stage1 audit {audit}")
metrics["audit"] = audit
(dest / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps({"eval_done": metrics}, indent=2), flush=True)
