"""STAR-Pro + CDPruner ACC on every lmms task that appeared on this server.

R9 core/extras for STAR/CD are owned by launch_r9_extras_fill.py (or already
complete). This launcher fills the remaining historical tasks that are NOT in
the default R9 set: mmbench_cn_dev + textvqa_val (non-OCR), and any R9 hole
that is still missing after a skip check (safe if the other queue already wrote
metrics.json).
Skips existing metrics.json. Shares GPUs {0,2,4,7} via gpu_exclusive.
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

PKG = REPO / "experiments/bela_r9_fill_20261026"
LOG_DIR = PKG / "logs_star_cd"
PY = "/home/xiangyue/envs/starpro/bin/python"

# Historical tasks that appeared on this server but are outside default R9.
# R9 holes for STAR T32 / CD are filled by launch_r9_extras_fill.py.
SERVER_TASKS = (
    "mmbench_cn_dev",
    "textvqa_val",
)

STAR_ARMS = (
    ("STARPRO_T64", 64, REPO / "logs/starpro_x0_t64/full/STARPRO_T64"),
    ("STARPRO_T32", 32, REPO / "logs/starpro_x0_t64/full/STARPRO_T32"),
)
CD_ARMS = (
    (64, REPO / "logs/cdpruner_x0/full/CDPRUNER_K64"),
    (32, REPO / "logs/cdpruner_x0/full/CDPRUNER_K32"),
)


def build_jobs() -> list[tuple]:
    jobs = []
    for arm_id, T, root in STAR_ARMS:
        for task in SERVER_TASKS:
            out = root / task
            if (out / "metrics.json").exists():
                print(f"HAVE STAR {arm_id}/{task}", flush=True)
                continue
            jobs.append(("star", arm_id, task, out, T))
    for k, root in CD_ARMS:
        arm_id = f"CDPRUNER_K{k}"
        for task in SERVER_TASKS:
            out = root / task
            if (out / "metrics.json").exists():
                print(f"HAVE CD {arm_id}/{task}", flush=True)
                continue
            jobs.append(("cd", arm_id, task, out, k))
    return jobs


def start(kind: str, name: str, task: str, out: Path, payload, gpu: int) -> subprocess.Popen:
    out.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"acc_{name}_{task}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("HF_DATASETS_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    if kind == "star":
        env["PYTHONPATH"] = (
            f"{REPO / 'third_party/LLaVA-starpro'}:"
            f"{REPO / 'experiments/starpro_x0_t64'}:{REPO / 'lmms-eval'}"
        )
        cmd = [
            PY,
            str(REPO / "experiments/starpro_x0_t64/run_one_task.py"),
            "--task",
            task,
            "--gpu-index",
            str(gpu),
            "--T",
            str(int(payload)),
        ]
    else:
        env["PYTHONPATH"] = (
            f"{REPO / 'third_party/CDPruner'}:"
            f"{REPO / 'experiments/cdpruner_x0'}:{REPO / 'lmms-eval'}"
        )
        cmd = [
            PY,
            str(REPO / "experiments/cdpruner_x0/run_one_task.py"),
            "--task",
            task,
            "--gpu-index",
            str(gpu),
            "--visual-token-num",
            str(int(payload)),
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
    print(f"START gpu={gpu} {kind}:{name}:{task} pid={proc.pid}", flush=True)
    return proc


def main() -> None:
    jobs = build_jobs()
    print(f"STAR/CD server-task pending {len(jobs)}", flush=True)
    print(json.dumps([f"{k}:{n}:{t}" for k, n, t, _, _ in jobs], indent=2), flush=True)
    running: list[tuple[subprocess.Popen, object, str]] = []
    while jobs or running:
        still = []
        for proc, lease, label in running:
            if proc.poll() is None:
                still.append((proc, lease, label))
            else:
                release(lease)  # type: ignore[arg-type]
                print(f"DONE {label} rc={proc.returncode}", flush=True)
        running = still
        while jobs:
            leased = None
            for gpu in PROJECT_GPUS:
                if any(g.gpu == gpu for _, g, _ in running):  # type: ignore[attr-defined]
                    continue
                leased = try_acquire(
                    gpu, f"sc:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS
                )
                if leased is not None:
                    break
            if leased is None:
                break
            kind, name, task, out, payload = jobs.pop(0)
            proc = start(kind, name, task, out, payload, leased.gpu)
            running.append((proc, leased, f"{kind}:{name}:{task}"))
        time.sleep(20)
    print("STAR/CD server-task queue drained", flush=True)
    missing = []
    for kind, name, task, out, _ in build_jobs():
        missing.append(f"{kind}:{name}:{task}")
    # rebuild after drain
    missing = []
    for arm_id, _T, root in STAR_ARMS:
        for task in SERVER_TASKS:
            if not (root / task / "metrics.json").exists():
                missing.append(f"STAR:{arm_id}:{task}")
    for k, root in CD_ARMS:
        for task in SERVER_TASKS:
            if not (root / task / "metrics.json").exists():
                missing.append(f"CD:K{k}:{task}")
    if missing:
        print("MISSING:", missing, flush=True)
        raise SystemExit(1)
    print("ALL STAR/CD server tasks complete", flush=True)


if __name__ == "__main__":
    main()
