"""W2048 V5 COVER+Q_ATTN: wide-short / narrow-long vs anchor schedule.

All three share the first 4 blocks at K=172 and exact W=2048.
Anchor metrics reused from R3 V5; only the two new shapes are queued.
GPUs 0/2/4/7 only.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/elva_cvpr_final/ELVA_FINAL_HANDOFF_20261020"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
ACC_GPUS = (0, 2, 4, 7)
import sys
sys.path.insert(0, str(REPO))
from elva_acc_tasks import ACC_TASKS
TASKS = ACC_TASKS
BASE_ARM = "E018_COVER_TWO32"
CONFIGS = (
    "H2_W2048_V5_WIDE_SHORT",
    "H2_W2048_V5_NARROW_LONG",
)
OUT_ROOT = REPO / "logs/e014_aee/full/elva_w2048_schedule_shapes"
ANCHOR_SRC = REPO / "logs/e014_aee/full/elva_r3_stage1/R3_W2048_V5_a5a7f59fc3"
ANCHOR_DST = OUT_ROOT / "H2_W2048_V5_ANCHOR_172_90_32"


def load_cfg(name: str) -> dict:
    return json.loads((PKG / "configs" / f"{name}.json").read_text())


def env_for(cfg: dict, audit_dir: Path) -> dict:
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(R3))
    from e014_arms import apply_arm_env
    from apply_llava import overlay
    path = PKG / "configs" / f"{cfg['id']}.json"
    os.environ["ELVA_R3_CONFIG"] = str(path)
    os.environ["E014_ARM"] = BASE_ARM
    meta = overlay(apply_arm_env(BASE_ARM))
    keep = {}
    for key, value in os.environ.items():
        if key.startswith(("E0", "ELVA_", "USE_", "STAGE", "COMPUTE")) or key == "E014_ARM":
            keep[key] = value
    keep["ELVA_R3_CONFIG"] = str(path)
    keep["ELVA_R3_AUDIT_DIR"] = str(audit_dir)
    keep["E014_ARM"] = meta["arm"]
    return keep


def link_anchor():
    ANCHOR_DST.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        src = ANCHOR_SRC / task / "metrics.json"
        out = ANCHOR_DST / task
        out.mkdir(parents=True, exist_ok=True)
        target = out / "metrics.json"
        if target.exists() or not src.exists():
            continue
        os.symlink(src.resolve(), target)
        for name in ("audit.json", "trace.jsonl", "stage1.jsonl"):
            s2 = ANCHOR_SRC / task / name
            t2 = out / name
            if s2.exists() and not t2.exists():
                os.symlink(s2.resolve(), t2)


def run_acc():
    link_anchor()
    jobs = []
    for name in CONFIGS:
        cfg = load_cfg(name)
        for task in TASKS:
            out = OUT_ROOT / cfg["id"] / task
            if (out / "metrics.json").exists():
                continue
            jobs.append((cfg, task, out))
    print(f"ACC pending {len(jobs)}", flush=True)
    slots = list(ACC_GPUS)
    running = []
    while jobs or running:
        running = [(p, gpu, name) for p, gpu, name in running if p.poll() is None]
        busy = {gpu for _, gpu, _ in running}
        while jobs and len(slots) > len(busy):
            gpu = next(g for g in slots if g not in busy)
            cfg, task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = PKG / "logs" / f"acc_{cfg['id']}_{task}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                PY, str(REPO / "e014_boot.py"),
                "--profile", "ANCHOR_A", "--task", task,
                "--output", str(out), "--gpu-index", str(gpu),
                "--layer", "8",
            ]
            child_env = os.environ.copy()
            child_env.update(env_for(cfg, out))
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
    print("ACC queue drained", flush=True)


if __name__ == "__main__":
    run_acc()
