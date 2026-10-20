"""BELA Gate1: frozen Stage1 + Q_ATTN/RDRIFT baselines, ACC via ACC_TASKS (R4+extras).

Does NOT claim BELA LifetimeState model integration (missing from zip).
GPUs 0/2/4/7 only. Reuses existing R4 metrics when present; queues the rest.
VizWiz added after the first 8-task wave is launched via launch_vizwiz_backfill.py.
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

PKG = REPO / "experiments/bela_20261020/BELA_CURSOR_20261020"
R3 = REPO / "experiments/elva_r3"
CVPR = REPO / "experiments/elva_cvpr_final/ELVA_FINAL_HANDOFF_20261020"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
# ACC on 0/2/4; GPU 7 reserved for SparseVLM-style TTFT/FLOPs (see launch_gate1_ttft.py).
ACC_GPUS = (0, 2, 4)
BASE_ARM = "E018_COVER_TWO32"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_gate1_8task"

# (config_path, out_id, reuse_r4_root_or_None)
ARMS = (
    (
        R3 / "configs/R3_W2048_V5_a5a7f59fc3.json",
        "W2048_V5_COVER_QATTN",
        REPO / "logs/e014_aee/full/elva_r3_stage1/R3_W2048_V5_a5a7f59fc3",
    ),
    (
        CVPR / "configs/H2_W2048_V5_COVER_RDRIFT.json",
        "W2048_V5_COVER_RDRIFT",
        REPO / "logs/e014_aee/full/elva_cvpr_p0/H2_W2048_V5_COVER_RDRIFT",
    ),
    (
        R3 / "configs/R3_W1024_V4_PCOV60_B12_08205a4243.json",
        "W1024_V4_PCOV_QATTN",
        REPO / "logs/e014_aee/full/elva_r3_stage1/R3_W1024_V4_PCOV60_B12_08205a4243",
    ),
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


def link_reuse(src: Path | None, dest: Path, task: str) -> bool:
    """Return True if metrics already satisfied via reuse or prior run."""
    target = dest / task / "metrics.json"
    if target.exists():
        return True
    if src is None:
        return False
    sm = src / task / "metrics.json"
    if not sm.exists():
        return False
    (dest / task).mkdir(parents=True, exist_ok=True)
    os.symlink(sm.resolve(), target)
    for name in ("audit.json", "trace.jsonl", "stage1.jsonl"):
        s2, t2 = src / task / name, dest / task / name
        if s2.exists() and not t2.exists():
            os.symlink(s2.resolve(), t2)
    return True


def run_acc():
    jobs = []
    for cfg_path, out_id, reuse in ARMS:
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        cfg = json.loads(cfg_path.read_text())
        root = OUT_ROOT / out_id
        root.mkdir(parents=True, exist_ok=True)
        for task in ACC_TASKS:
            if link_reuse(reuse, root, task):
                print(f"HAVE {out_id}/{task}", flush=True)
                continue
            out = root / task
            jobs.append((cfg_path, cfg, out_id, task, out))
    print(f"ACC pending {len(jobs)} / total arms×tasks={len(ARMS)*len(ACC_TASKS)}", flush=True)
    slots = list(ACC_GPUS)
    running = []
    while jobs or running:
        running = [(p, gpu, name) for p, gpu, name in running if p.poll() is None]
        busy = {gpu for _, gpu, _ in running}
        while jobs and len(slots) > len(busy):
            gpu = next(g for g in slots if g not in busy)
            cfg_path, cfg, out_id, task, out = jobs.pop(0)
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
            running.append((proc, gpu, f"{out_id}:{task}"))
            busy.add(gpu)
            print(f"START gpu={gpu} {out_id} {task} pid={proc.pid}", flush=True)
        time.sleep(20)
    print("ACC queue drained", flush=True)


if __name__ == "__main__":
    run_acc()
