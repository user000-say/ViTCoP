"""Dense LLaVA-1.5 baseline (no token prune) on mmmu_val + vizwiz_vqa_val.

Keeps all 576 visual tokens via E010_ANALYSIS_FULL576; E014/E018/E006/E012 off.
Same offline task yamls as Gate1 (MMMU group_img, VizWiz load_from_disk).
GPUs 0/2/4/7 only; waits for free cards so it does not interrupt Gate1.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/bela_20261020/BELA_CURSOR_20261020"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
GPUS = (0, 2, 4, 7)
OUT_ROOT = REPO / "logs/e014_aee/full/dense_s0_extras"
TASKS = ("mmmu_val", "vizwiz_vqa_val")


def free_gpus():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = []
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(",")]
        if int(idx) in GPUS and float(mem) < 2000:
            free.append(int(idx))
    return free


def main():
    jobs = []
    for task in TASKS:
        out = OUT_ROOT / task
        if (out / "metrics.json").exists():
            print("HAVE dense", task, flush=True)
            continue
        jobs.append((task, out))
    print(f"Dense pending {len(jobs)}", flush=True)
    running = []
    while jobs or running:
        running = [(p, gpu, name) for p, gpu, name in running if p.poll() is None]
        busy = {gpu for _, gpu, _ in running}
        for gpu in free_gpus():
            if not jobs or gpu in busy:
                continue
            task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = PKG / "logs" / f"dense_s0_{task}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            # e010_eval_job + S0, then hard-disable prune stacks via env override
            # applied inside a tiny runner so E014 overlay never re-enables COVER.
            runner = PKG / "scripts" / "_run_dense_s0_job.py"
            cmd = [
                PY, str(runner),
                "--task", task,
                "--output", str(out),
                "--gpu-index", str(gpu),
            ]
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            child_env["PYTHONPATH"] = (
                f"{REPO}:{REPO / 'experiments'}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}:{PKG}"
            )
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["HF_HUB_OFFLINE"] = "1"
            child_env["HF_DATASETS_OFFLINE"] = "1"
            child_env["TRANSFORMERS_OFFLINE"] = "1"
            # ensure no ELVA schedule / E014 arm from parent shells
            for k in list(child_env):
                if k.startswith(("ELVA_", "E014_", "E018_", "E015_", "E016_")):
                    child_env.pop(k, None)
            handle = log.open("w")
            proc = subprocess.Popen(
                cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True)
            running.append((proc, gpu, task))
            busy.add(gpu)
            print(f"START dense gpu={gpu} {task} pid={proc.pid}", flush=True)
        time.sleep(30)
    print("Dense queue drained", flush=True)


if __name__ == "__main__":
    main()
