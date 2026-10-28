"""QA-COVER protect set: MME, MMBench-EN, ScienceQA-IMG, TextVQA-OCR.

Same S1_QANCHOR_L2 config as the GQA/POPE pilot. GPUs stay inside 0/2/4/7;
busy cards in that set are skipped.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG_DIR = REPO / "experiments/bela_query_anchor"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(PKG_DIR))
from gpu_exclusive import PROJECT_GPUS, release, try_acquire  # noqa: E402
from launch_qacover_pilot import (  # noqa: E402
    LOG_DIR,
    OUT_ROOT,
    PKG,
    PY,
    R3,
    env_for,
    write_run_meta,
)

TASKS = ("textvqa_val_ocr", "mmbench_en_dev", "mme", "scienceqa_img")
ARM = "S1_QANCHOR_L2"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = PKG / "configs" / f"{ARM}.json"
    root = OUT_ROOT / ARM
    jobs = []
    for task in TASKS:
        out = root / task
        if (out / "metrics.json").exists():
            print(f"HAVE {ARM}/{task}", flush=True)
            continue
        jobs.append((task, out))
    print(f"QA-COVER protect pending {len(jobs)} on {PROJECT_GPUS}", flush=True)
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
                leased = try_acquire(gpu, f"qacover:{ARM}:{jobs[0][0]}", allowed=PROJECT_GPUS)
                if leased is not None:
                    break
            if leased is None:
                break
            task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = LOG_DIR / f"acc_{ARM}_{task}.log"
            child_env = os.environ.copy()
            child_env.update(env_for(cfg, out))
            child_env["CUDA_VISIBLE_DEVICES"] = str(leased.gpu)
            child_env["PYTHONPATH"] = (
                f"{REPO}:{REPO / 'experiments'}:{R3}:{PKG}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
            )
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["HF_HUB_OFFLINE"] = "1"
            child_env["HF_DATASETS_OFFLINE"] = "1"
            child_env["TRANSFORMERS_OFFLINE"] = "1"
            write_run_meta(out, ARM, cfg, child_env)
            cmd = [
                PY, str(REPO / "e014_boot.py"),
                "--profile", "ANCHOR_A", "--task", task,
                "--output", str(out), "--gpu-index", str(leased.gpu), "--layer", "8",
            ]
            proc = subprocess.Popen(
                cmd, env=child_env, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True,
            )
            running.append((proc, leased, f"{ARM}:{task}"))
            print(f"START gpu={leased.gpu} {ARM} {task} pid={proc.pid}", flush=True)
        time.sleep(20)
    missing = [task for task in TASKS if not (root / task / "metrics.json").exists()]
    if missing:
        print("MISSING", missing, flush=True)
        raise SystemExit(1)
    print("QA-COVER protect drained", flush=True)


if __name__ == "__main__":
    main()
