#!/usr/bin/env python3
"""Fair-256 Fixed32/TTFT: S1 Dual+QA+RDRIFT vs P3 Dual+QA+Q_ATTN (best R6 today).

Same panel / warmups / generate(1) & generate(32) protocol as elva_r3 TTFT.
Owns one physical GPU (default 4). Does not claim ACC.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
DEC = REPO / "experiments/bela_stage1_decision_package"
QA = REPO / "experiments/bela_query_anchor"
R3 = REPO / "experiments/elva_r3"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(QA))

from launch_qacover_pilot import PY, env_for, write_run_meta  # noqa: E402

# QA-COVER needs begin_question; stock fair-256 bench has no request bind.
BENCH = DEC / "bench_llava_qa_fixed32.py"
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
OUT_ROOT = DEC / "latency"
LOG_DIR = DEC / "logs"

# Best Q_ATTN six-task today vs production RDRIFT anchor (same Stage1 QA B8 / L2 W1024).
ARMS = [
    ("S1_QANCHOR_L2_RDRIFT", QA / "configs/S1_QANCHOR_L2.json"),
    ("P3_QA_QATTN", DEC / "configs/P3_QA_QATTN.json"),
]


def _child_env(cfg_path: Path, out: Path) -> dict:
    child = os.environ.copy()
    child.update(env_for(cfg_path, out))
    cfg = json.loads(cfg_path.read_text())
    init = cfg.get("method", {}).get("stage1_init", "cover")
    child["BELA_STAGE1_INIT"] = init
    child.pop("BELA_QA_PASS_CAPS", None)
    child["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{QA}:{DEC}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
    )
    child["PYTHONUNBUFFERED"] = "1"
    child["HF_HUB_OFFLINE"] = "1"
    child["HF_DATASETS_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    return child


def main() -> None:
    gpu = int(os.environ.get("VITCOP_FIXED32_GPU", "4"))
    warmups = int(os.environ.get("VITCOP_FIXED32_WARMUPS", "8"))
    limit = int(os.environ.get("VITCOP_FIXED32_LIMIT", "0"))  # 0 = full panel
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"FIXED32_PAIR gpu={gpu} warmups={warmups} limit={limit}", flush=True)

    for arm_id, cfg_path in ARMS:
        out = OUT_ROOT / arm_id
        summary = out / "latency_summary.json"
        if summary.exists():
            print(f"SKIP_HAVE {arm_id} {summary}", flush=True)
            continue
        if not cfg_path.exists():
            raise SystemExit(f"missing config {cfg_path}")
        out.mkdir(parents=True, exist_ok=True)
        child = _child_env(cfg_path, out)
        child["CUDA_VISIBLE_DEVICES"] = str(gpu)
        write_run_meta(out, arm_id, cfg_path, child)
        # Record Stage2 identity for the paper agent.
        (out / "arm_note.json").write_text(
            json.dumps(
                {
                    "arm": arm_id,
                    "config": str(cfg_path),
                    "stage2": json.loads(cfg_path.read_text())["method"]["stage2"],
                    "stage1_qa_max_swaps": json.loads(cfg_path.read_text())["method"]["qa_cover"]["anchor"]["max_swaps"],
                    "protocol": "e014_latency_fair_256 generate(1)+generate(32) panel speed_256",
                },
                indent=2,
            )
            + "\n"
        )
        log = LOG_DIR / f"ttft_{arm_id}.log"
        cmd = [
            PY,
            str(BENCH),
            "--repo-root",
            str(REPO),
            "--panel",
            str(PANEL),
            "--model-path",
            str(MODEL),
            "--conv-mode",
            "llava_v1",
            "--out-dir",
            str(out),
            "--warmups",
            str(warmups),
            "--limit",
            str(limit),
        ]
        print(f"START {arm_id} gpu={gpu} log={log}", flush=True)
        t0 = time.time()
        with log.open("w") as handle:
            rc = subprocess.call(cmd, env=child, stdout=handle, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if rc != 0 or not summary.exists():
            raise SystemExit(f"FAIL {arm_id} rc={rc} after {dt:.0f}s — see {log}")
        data = json.loads(summary.read_text())
        print(
            f"DONE {arm_id} {dt:.0f}s "
            f"TTFT_mean={data['ttft_wall_ms']['mean']:.2f} "
            f"Fixed32_mean={data['fixed32_wall_ms']['mean']:.2f}",
            flush=True,
        )
    print("FIXED32_PAIR_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
