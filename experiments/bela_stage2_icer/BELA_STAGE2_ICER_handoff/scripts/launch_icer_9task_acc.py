"""BELA Stage2 ICER: 5 new arms × 9-task ACC on GPUs 0/2/4.

Skips re-running Q-Attn / RDRIFT Top-K baselines (reuse Gate1 metrics).
No smoke. User-directed full nine-task protocol.
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
from elva_acc_tasks import ACC_TASKS

PKG = REPO / "experiments/bela_stage2_icer/BELA_STAGE2_ICER_handoff"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
ACC_GPUS = (0, 2, 4)
BASE_ARM = "E018_COVER_TWO32"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_icer_g1"
CFG = PKG / "configs"

ARMS = (
    ("S2_W2048_QATTN_ICER", "icer", {}),
    ("S3_W2048_QATTN_ICER_DRIFT", "icer_drift", {}),
    ("S4_W2048_RDRIFT_ICER", "icer", {}),
    ("S5_W2048_SINGLETON_PROXY", "singleton_proxy", {}),
    ("S7_W2048_RANDOM_SWAP", "random_swap", {}),
)


def env_for(cfg_path: Path, audit_dir: Path, mode: str) -> dict:
    sys.path.insert(0, str(R3))
    from e014_arms import apply_arm_env
    from apply_llava import overlay
    os.environ["ELVA_R3_CONFIG"] = str(cfg_path)
    os.environ["E014_ARM"] = BASE_ARM
    meta = overlay(apply_arm_env(BASE_ARM))
    keep = {}
    for key, value in os.environ.items():
        if key.startswith(("E0", "ELVA_", "USE_", "STAGE", "COMPUTE", "BELA_")) or key == "E014_ARM":
            keep[key] = value
    keep["ELVA_R3_CONFIG"] = str(cfg_path)
    keep["ELVA_R3_AUDIT_DIR"] = str(audit_dir)
    keep["E014_ARM"] = meta["arm"]
    keep["BELA_ICER_MODE"] = mode
    keep["BELA_ICER_MAX_SWAPS"] = "3"
    keep["BELA_ICER_DONOR_POOL"] = "24"
    keep["BELA_ICER_DROP_POOL"] = "16"
    keep["BELA_ICER_MIN_QMASS"] = "0.97"
    keep["BELA_ICER_DRIFT_WEIGHT"] = "0.20"
    keep["BELA_ICER_DIAG"] = "False"
    return keep


def main():
    jobs = []
    for out_id, mode, _extra in ARMS:
        cfg_path = CFG / f"{out_id}.json"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        cfg = json.loads(cfg_path.read_text())
        root = OUT_ROOT / out_id
        root.mkdir(parents=True, exist_ok=True)
        for task in ACC_TASKS:
            out = root / task
            if (out / "metrics.json").exists():
                print(f"HAVE {out_id}/{task}", flush=True)
                continue
            jobs.append((cfg_path, mode, out_id, task, out))
    print(f"ICER ACC pending {len(jobs)} / {len(ARMS)*len(ACC_TASKS)}", flush=True)
    running = []
    while jobs or running:
        running = [(p, gpu, name) for p, gpu, name in running if p.poll() is None]
        busy = {gpu for _, gpu, _ in running}
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        free = []
        for line in out.strip().splitlines():
            idx, mem = [x.strip() for x in line.split(",")]
            if int(idx) in ACC_GPUS and float(mem) < 2000:
                free.append(int(idx))
        for gpu in free:
            if not jobs or gpu in busy:
                continue
            cfg_path, mode, out_id, task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = PKG / "logs" / f"acc_{out_id}_{task}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                PY, str(REPO / "e014_boot.py"),
                "--profile", "ANCHOR_A", "--task", task,
                "--output", str(out), "--gpu-index", str(gpu),
                "--layer", "8",
            ]
            child_env = os.environ.copy()
            child_env.update(env_for(cfg_path, out, mode))
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            child_env["PYTHONPATH"] = (
                f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:"
                f"{REPO / 'e020_lib'}:{PKG}"
            )
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["HF_HUB_OFFLINE"] = "1"
            child_env["HF_DATASETS_OFFLINE"] = "1"
            child_env["TRANSFORMERS_OFFLINE"] = "1"
            handle = log.open("w")
            proc = subprocess.Popen(
                cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True)
            running.append((proc, gpu, f"{out_id}:{task}"))
            busy.add(gpu)
            print(f"START gpu={gpu} {out_id} {task} mode={mode} pid={proc.pid}", flush=True)
        time.sleep(20)
    print("ICER ACC queue drained", flush=True)


if __name__ == "__main__":
    main()
