"""Backfill EXTRA_TASKS onto completed ELVA ACC roots that only have R4 four.

Targets schedule-shape + qed-vs-qattn + p0 dirs under logs/e014_aee/full.
GPUs 0/2/4/7. Skips tasks that already have metrics.json.
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
from elva_acc_tasks import EXTRA_TASKS

PKG = REPO / "experiments/elva_cvpr_final/ELVA_FINAL_HANDOFF_20261020"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
ACC_GPUS = (0, 2, 4, 7)
BASE_ARM = "E018_COVER_TWO32"

# (config_json_path, output_root_for_arm)
ARMS = [
    (PKG / "configs/H2_W2048_V5_WIDE_SHORT.json",
     REPO / "logs/e014_aee/full/elva_w2048_schedule_shapes/H2_W2048_V5_WIDE_SHORT"),
    (PKG / "configs/H2_W2048_V5_NARROW_LONG.json",
     REPO / "logs/e014_aee/full/elva_w2048_schedule_shapes/H2_W2048_V5_NARROW_LONG"),
    (R3 / "configs/R3_W2048_V5_a5a7f59fc3.json",
     REPO / "logs/e014_aee/full/elva_w2048_schedule_shapes/H2_W2048_V5_ANCHOR_172_90_32"),
    (PKG / "configs/H2_W2048_V5_COVER_QED.json",
     REPO / "logs/e014_aee/full/elva_w2048_qed_vs_qattn/H2_W2048_V5_COVER_QED"),
    (R3 / "configs/R3_W2048_V5_a5a7f59fc3.json",
     REPO / "logs/e014_aee/full/elva_w2048_qed_vs_qattn/R3_W2048_V5_Q_ATTN"),
]


def env_for(cfg_path: Path, audit_dir: Path) -> dict:
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
    jobs = []
    for cfg_path, root in ARMS:
        if not cfg_path.exists():
            print("SKIP missing config", cfg_path, flush=True)
            continue
        cfg = json.loads(cfg_path.read_text())
        for task in EXTRA_TASKS:
            out = root / task
            if (out / "metrics.json").exists():
                continue
            # require R4 gqa present so we only backfill finished arms
            if not (root / "gqa" / "metrics.json").exists():
                print(f"WAIT arm incomplete {root.name}", flush=True)
                break
            jobs.append((cfg_path, cfg, task, out))
    print(f"EXTRA pending {len(jobs)}", flush=True)
    slots = list(ACC_GPUS)
    running = []
    while jobs or running:
        running = [(p, gpu, name) for p, gpu, name in running if p.poll() is None]
        busy = {gpu for _, gpu, _ in running}
        while jobs and len(slots) > len(busy):
            gpu = next(g for g in slots if g not in busy)
            cfg_path, cfg, task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = PKG / "logs" / f"acc_{cfg['id']}_{task}.log"
            cmd = [
                PY, str(REPO / "e014_boot.py"),
                "--profile", "ANCHOR_A", "--task", task,
                "--output", str(out), "--gpu-index", str(gpu),
                "--layer", "8",
            ]
            child_env = os.environ.copy()
            child_env.update(env_for(cfg_path, out))
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            child_env["PYTHONPATH"] = (
                f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
            )
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["HF_HUB_OFFLINE"] = "1"
            child_env["HF_DATASETS_OFFLINE"] = "1"
            child_env["TRANSFORMERS_OFFLINE"] = "1"
            handle = log.open("w")
            proc = subprocess.Popen(
                cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True)
            running.append((proc, gpu, f"{cfg['id']}:{task}"))
            busy.add(gpu)
            print(f"START gpu={gpu} {cfg['id']} {task} pid={proc.pid}", flush=True)
        time.sleep(20)
    print("EXTRA queue drained", flush=True)


if __name__ == "__main__":
    main()
