"""Independent handoff P0: official STAR-Pro F1024_QS (T=32) full-6 ACC.

Schedule shape matches L2 / paper T32: Stage1 pool 64, then (12,17),(20,10).
Isolated env /home/xiangyue/envs/starpro. GPUs {0,2,4,7} via gpu_exclusive.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
from gpu_exclusive import PROJECT_GPUS, release, try_acquire  # noqa: E402

PKG = REPO / "experiments/bela_cvpr27_independent_20261025"
STAR_PKG = REPO / "experiments/starpro_x0_t64"
PY = "/home/xiangyue/envs/starpro/bin/python"
OUT_ROOT = REPO / "logs/starpro_x0_t64/full/STARPRO_T32"
LOG_DIR = PKG / "logs"
TASKS6 = (
    "gqa",
    "pope",
    "mme",
    "mmbench_en_dev",
    "scienceqa_img",
    "textvqa_val_ocr",
)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [t for t in TASKS6 if not (OUT_ROOT / t / "metrics.json").exists()]
    print(f"F1024_QS STAR T32 pending {len(jobs)}: {jobs}", flush=True)
    running: list[tuple[subprocess.Popen, object, str]] = []
    while jobs or running:
        still = []
        for proc, lease, name in running:
            if proc.poll() is None:
                still.append((proc, lease, name))
            else:
                release(lease)  # type: ignore[arg-type]
                print(f"DONE {name} rc={proc.returncode}", flush=True)
        running = still
        while jobs:
            leased = None
            for gpu in PROJECT_GPUS:
                if any(g.gpu == gpu for _, g, _ in running):  # type: ignore[attr-defined]
                    continue
                leased = try_acquire(gpu, f"star_t32:{jobs[0]}", allowed=PROJECT_GPUS)
                if leased is not None:
                    break
            if leased is None:
                break
            task = jobs.pop(0)
            log = LOG_DIR / f"acc_STARPRO_T32_{task}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(leased.gpu)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = (
                f"{REPO / 'third_party/LLaVA-starpro'}:"
                f"{STAR_PKG}:{REPO / 'lmms-eval'}"
            )
            # allow HF dataset metadata (starpro runner pops offline inside)
            env.pop("HF_HUB_OFFLINE", None)
            env.pop("HF_DATASETS_OFFLINE", None)
            env.pop("TRANSFORMERS_OFFLINE", None)
            cmd = [
                PY,
                str(STAR_PKG / "run_one_task.py"),
                "--task",
                task,
                "--gpu-index",
                str(leased.gpu),
                "--T",
                "32",
            ]
            handle = log.open("w")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd="/tmp",
            )
            running.append((proc, leased, task))
            print(f"START gpu={leased.gpu} STAR_T32 {task} pid={proc.pid}", flush=True)
        time.sleep(20)
    print("F1024_QS STAR T32 ACC drained", flush=True)
    missing = [t for t in TASKS6 if not (OUT_ROOT / t / "metrics.json").exists()]
    if missing:
        print(f"MISSING {missing}", flush=True)
        raise SystemExit(1)
    # brief score dump
    dense = {
        "gqa": 61.9,
        "pope": 85.8,
        "mme": 1852.0,
        "mmbench_en_dev": 64.1,
        "scienceqa_img": 69.51,
        "textvqa_val_ocr": 58.20,
    }

    def abs_score(task, m):
        if task == "pope":
            return 100.0 * float(m["pope_f1_score"])
        if task == "mme":
            return float(m["mme_perception_score"]) + float(m["mme_cognition_score"])
        if task == "mmbench_en_dev":
            return float(m["gpt_eval_score"])
        v = m["exact_match"]
        return 100.0 * float(v) if float(v) <= 1.5 else float(v)

    scores = {
        t: abs_score(t, json.loads((OUT_ROOT / t / "metrics.json").read_text()))
        for t in TASKS6
    }
    r4 = sum(100 * scores[t] / dense[t] for t in TASKS6[:4]) / 4
    m6 = sum(100 * scores[t] / dense[t] for t in TASKS6) / 6
    print(f"STAR_T32 R4={r4:.2f} mean6={m6:.2f} {scores}", flush=True)


if __name__ == "__main__":
    main()
