"""CDPruner official source: K=32 and K=64 on 6 judgment tasks.

Uses third_party/CDPruner + starpro env (matching deps). Same lmms-eval scorer
as BELA/STAR so mean6 is comparable. GPUs {0,2,4,7} via gpu_exclusive.
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

PKG = REPO / "experiments/cdpruner_x0"
PY = "/home/xiangyue/envs/starpro/bin/python"
OUT_ROOT = REPO / "logs/cdpruner_x0/full"
LOG_DIR = PKG / "logs"
TASKS6 = (
    "gqa",
    "pope",
    "mme",
    "mmbench_en_dev",
    "scienceqa_img",
    "textvqa_val_ocr",
)
KS = (64, 32)  # paper-common retain counts; 64 first


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k in KS:
        for task in TASKS6:
            if (OUT_ROOT / f"CDPRUNER_K{k}" / task / "metrics.json").exists():
                print(f"HAVE K{k}/{task}", flush=True)
                continue
            jobs.append((k, task))
    print(f"CDPruner ACC pending {len(jobs)}", flush=True)
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
                leased = try_acquire(
                    gpu, f"cdp:K{jobs[0][0]}:{jobs[0][1]}", allowed=PROJECT_GPUS
                )
                if leased is not None:
                    break
            if leased is None:
                break
            k, task = jobs.pop(0)
            log = LOG_DIR / f"acc_CDPRUNER_K{k}_{task}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(leased.gpu)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = (
                f"{REPO / 'third_party/CDPruner'}:"
                f"{PKG}:{REPO / 'lmms-eval'}"
            )
            env.pop("HF_HUB_OFFLINE", None)
            env.pop("HF_DATASETS_OFFLINE", None)
            env.pop("TRANSFORMERS_OFFLINE", None)
            cmd = [
                PY,
                str(PKG / "run_one_task.py"),
                "--task",
                task,
                "--gpu-index",
                str(leased.gpu),
                "--visual-token-num",
                str(k),
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
            running.append((proc, leased, f"K{k}:{task}"))
            print(f"START gpu={leased.gpu} CDPruner K{k} {task} pid={proc.pid}", flush=True)
        time.sleep(20)
    print("CDPruner ACC drained", flush=True)

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

    for k in KS:
        scores = {}
        for t in TASKS6:
            p = OUT_ROOT / f"CDPRUNER_K{k}" / t / "metrics.json"
            if not p.exists():
                print(f"MISSING K{k}/{t}", flush=True)
                scores = None
                break
            scores[t] = abs_score(t, json.loads(p.read_text()))
        if not scores:
            continue
        r4 = sum(100 * scores[t] / dense[t] for t in TASKS6[:4]) / 4
        m6 = sum(100 * scores[t] / dense[t] for t in TASKS6) / 6
        print(f"CDPRUNER_K{k} R4={r4:.2f} mean6={m6:.2f} {scores}", flush=True)


if __name__ == "__main__":
    main()
