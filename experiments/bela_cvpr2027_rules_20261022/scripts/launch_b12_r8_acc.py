"""Handoff V1 P1: B1/B2 full 8-task ACC (no VizWiz) + A0 mmmu/ai2d fill.

B1 172/90/20 W1808; B2 172/120/32 W2288. COVER+RDRIFT. GPUs 0/2/4/7.
Skip any metrics.json already present. VizWiz intentionally excluded.
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
from elva_acc_tasks import ACC_TASKS  # noqa: E402

PKG = REPO / "experiments/bela_cvpr2027_rules_20261022"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
ACC_GPUS = (0, 2, 4, 7)
BASE_ARM = "E018_COVER_TWO32"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_cvpr27_handoff_v1_b12"

# R8 = all ACC_TASKS except vizwiz
R8_TASKS = tuple(t for t in ACC_TASKS if t != "vizwiz_vqa_val")

ARMS = (
    (PKG / "configs/B1_COVER_RDRIFT_W1808.json", "B1_COVER_RDRIFT"),
    (PKG / "configs/B2_COVER_RDRIFT_W2288.json", "B2_COVER_RDRIFT"),
)

# Paired baseline fill for A0 (reuse T1 out root)
A0_FILL = (
    PKG / "configs/A0_COVER_RDRIFT_W2048.json",
    "A0_COVER_RDRIFT",
    REPO / "logs/e014_aee/full/bela_cvpr2027_t1_6task",
    ("mmmu_val", "ai2d"),
)


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


def run_acc() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cfg_path, out_id in ARMS:
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        root = OUT_ROOT / out_id
        root.mkdir(parents=True, exist_ok=True)
        for task in R8_TASKS:
            if (root / task / "metrics.json").exists():
                print(f"HAVE {out_id}/{task}", flush=True)
                continue
            jobs.append((cfg_path, out_id, task, root / task))

    a0_cfg, a0_id, a0_root, a0_tasks = A0_FILL
    a0_root.mkdir(parents=True, exist_ok=True)
    for task in a0_tasks:
        out = a0_root / a0_id / task
        if (out / "metrics.json").exists():
            print(f"HAVE {a0_id}/{task}", flush=True)
            continue
        jobs.append((a0_cfg, a0_id, task, out))

    print(f"ACC pending {len(jobs)} (B1/B2 R8 + A0 mmmu/ai2d; no VizWiz)", flush=True)
    print(json.dumps([f"{o}:{t}" for _, o, t, _ in jobs], indent=2), flush=True)

    def free_slots(busy: set[int], max_used_mib: int = 4000) -> list[int]:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        free = []
        for line in out.strip().splitlines():
            idx_s, used_s = [x.strip() for x in line.split(",")]
            idx, used = int(idx_s), int(float(used_s))
            if idx in ACC_GPUS and idx not in busy and used < max_used_mib:
                free.append(idx)
        return free

    running = []
    while jobs or running:
        running = [(p, gpu, name) for p, gpu, name in running if p.poll() is None]
        busy = {gpu for _, gpu, _ in running}
        while jobs:
            candidates = free_slots(busy)
            if not candidates:
                break
            gpu = candidates[0]
            cfg_path, out_id, task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = PKG / "logs" / f"acc_{out_id}_{task}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                PY,
                str(REPO / "e014_boot.py"),
                "--profile",
                "ANCHOR_A",
                "--task",
                task,
                "--output",
                str(out),
                "--gpu-index",
                str(gpu),
                "--layer",
                "8",
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
                cmd,
                env=child_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running.append((proc, gpu, f"{out_id}:{task}"))
            busy.add(gpu)
            print(f"START gpu={gpu} {out_id} {task} pid={proc.pid}", flush=True)
        time.sleep(20)
    print("ACC queue drained", flush=True)


if __name__ == "__main__":
    run_acc()
