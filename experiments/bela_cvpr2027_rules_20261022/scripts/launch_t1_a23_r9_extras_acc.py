"""T1 A2/A3: fill missing R9 extras only (skip existing metrics).

Already done for both arms: gqa, pope, mme, mmbench_en_dev, scienceqa_img, textvqa_val_ocr.
Pending: mmmu_val, ai2d, vizwiz_vqa_val.
GPUs 0/2/4/7. Same OUT_ROOT as prior T1 6-task run.
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
OUT_ROOT = REPO / "logs/e014_aee/full/bela_cvpr2027_t1_6task"

ARMS = (
    (PKG / "configs/A2_COVER_RDRIFT_W2048.json", "A2_COVER_RDRIFT"),
    (PKG / "configs/A3_COVER_RDRIFT_W2048.json", "A3_COVER_RDRIFT"),
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
        for task in ACC_TASKS:
            if (root / task / "metrics.json").exists():
                print(f"HAVE {out_id}/{task}", flush=True)
                continue
            jobs.append((cfg_path, out_id, task, root / task))
    print(f"ACC pending {len(jobs)} (A2/A3 R9 extras only)", flush=True)
    print(json.dumps([f"{o}:{t}" for _, o, t, _ in jobs], indent=2), flush=True)
    slots = list(ACC_GPUS)
    running = []
    while jobs or running:
        running = [(p, gpu, name) for p, gpu, name in running if p.poll() is None]
        busy = {gpu for _, gpu, _ in running}
        while jobs and len(slots) > len(busy):
            gpu = next(g for g in slots if g not in busy)
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
