"""Backfill vizwiz_vqa_val onto Gate1 arms (GPUs 0/2/4; leave 7 for TTFT)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/bela_20261020/BELA_CURSOR_20261020"
R3 = REPO / "experiments/elva_r3"
CVPR = REPO / "experiments/elva_cvpr_final/ELVA_FINAL_HANDOFF_20261020"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
ACC_GPUS = (0, 2, 4)
BASE_ARM = "E018_COVER_TWO32"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_gate1_8task"
TASK = "vizwiz_vqa_val"

ARMS = (
    (R3 / "configs/R3_W2048_V5_a5a7f59fc3.json", "W2048_V5_COVER_QATTN"),
    (CVPR / "configs/H2_W2048_V5_COVER_RDRIFT.json", "W2048_V5_COVER_RDRIFT"),
    (R3 / "configs/R3_W1024_V4_PCOV60_B12_08205a4243.json", "W1024_V4_PCOV_QATTN"),
)


def env_for(cfg_path: Path, audit_dir: Path) -> dict:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(R3))
    from e014_arms import apply_arm_env
    from apply_llava import overlay
    os.environ["ELVA_R3_CONFIG"] = str(cfg_path)
    os.environ["E014_ARM"] = BASE_ARM
    meta = overlay(apply_arm_env(BASE_ARM))
    keep = {}
    for key, value in os.environ.items():
        if key.startswith(("E0", "ELVA_", "USE_", "STAGE", "COMPUTE")) or key == "E014_ARM":
            keep[key] = value
    keep["ELVA_R3_CONFIG"] = str(cfg_path)
    keep["ELVA_R3_AUDIT_DIR"] = str(audit_dir)
    keep["E014_ARM"] = meta["arm"]
    return keep


def main():
    # Wait until ACC GPUs are free enough (current gate1 extras may still run).
    jobs = []
    for cfg_path, out_id in ARMS:
        out = OUT_ROOT / out_id / TASK
        if (out / "metrics.json").exists():
            print("HAVE", out_id, TASK, flush=True)
            continue
        cfg = json.loads(cfg_path.read_text())
        jobs.append((cfg_path, cfg, out_id, out))
    print(f"VizWiz pending {len(jobs)}", flush=True)
    # Wait for free slots among 0/2/4 (memory < 2GiB)
    def free_gpus():
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        free = []
        for line in out.strip().splitlines():
            idx, mem = [x.strip() for x in line.split(",")]
            if int(idx) in ACC_GPUS and float(mem) < 2000:
                free.append(int(idx))
        return free

    running = []
    while jobs or running:
        running = [(p, gpu, name) for p, gpu, name in running if p.poll() is None]
        busy = {gpu for _, gpu, _ in running}
        for gpu in free_gpus():
            if not jobs or gpu in busy:
                continue
            cfg_path, cfg, out_id, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = PKG / "logs" / f"acc_{out_id}_{TASK}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                PY, str(REPO / "e014_boot.py"),
                "--profile", "ANCHOR_A", "--task", TASK,
                "--output", str(out), "--gpu-index", str(gpu),
                "--layer", "8",
            ]
            child_env = os.environ.copy()
            child_env.update(env_for(cfg_path, out))
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            child_env["PYTHONPATH"] = (
                f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}:{PKG}"
            )
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["HF_HUB_OFFLINE"] = "1"
            child_env["HF_DATASETS_OFFLINE"] = "1"
            child_env["TRANSFORMERS_OFFLINE"] = "1"
            handle = log.open("w")
            proc = subprocess.Popen(
                cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True)
            running.append((proc, gpu, f"{out_id}:{TASK}"))
            busy.add(gpu)
            print(f"START gpu={gpu} {out_id} {TASK} pid={proc.pid}", flush=True)
        time.sleep(30)
    print("VizWiz queue drained", flush=True)


if __name__ == "__main__":
    main()
