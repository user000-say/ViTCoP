"""W2048 V5 COVER: Stage2 QED vs Q_ATTN four-task ACC.

Same budget/schedule as R3 V5 (172×4 + 90×8 + 32×20). GPUs 0/2/4/7 only.
AEE is Stage1 CLIP selection and cannot replace E018 Q_ATTN; not launched here.
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
# Q_ATTN baseline already complete under R3; only QED is queued unless missing.
CONFIGS = (
    "R3_W2048_V5_a5a7f59fc3",  # Q_ATTN (reuse elva_r3 config file)
    "H2_W2048_V5_COVER_QED",
)
OUT_ROOT = REPO / "logs/e014_aee/full/elva_w2048_qed_vs_qattn"
# Prefer copying completed R3 Q_ATTN metrics instead of re-running.
R3_QATTN_ROOT = REPO / "logs/e014_aee/full/elva_r3_stage1/R3_W2048_V5_a5a7f59fc3"


def cfg_path(name: str) -> Path:
    for base in (PKG / "configs", R3 / "configs"):
        p = base / f"{name}.json"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def load_cfg(name: str) -> dict:
    return json.loads(cfg_path(name).read_text())


def env_for(cfg: dict, audit_dir: Path) -> dict:
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(R3))
    from e014_arms import apply_arm_env
    from apply_llava import overlay
    path = cfg_path(cfg["id"])
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


def ensure_qattn_baseline():
    """Symlink/copy R3 V5 Q_ATTN metrics into this campaign root when complete."""
    dest = OUT_ROOT / "R3_W2048_V5_Q_ATTN"
    dest.mkdir(parents=True, exist_ok=True)
    missing = []
    for task in TASKS:
        src = R3_QATTN_ROOT / task / "metrics.json"
        out = dest / task
        out.mkdir(parents=True, exist_ok=True)
        target = out / "metrics.json"
        if target.exists():
            continue
        if src.exists():
            if not (out / "metrics.json").exists():
                os.symlink(src.resolve(), target)
            # also link audit if present
            for name in ("audit.json", "trace.jsonl", "stage1.jsonl"):
                s2 = R3_QATTN_ROOT / task / name
                t2 = out / name
                if s2.exists() and not t2.exists():
                    os.symlink(s2.resolve(), t2)
        else:
            missing.append(task)
    return missing


def run_acc():
    qattn_missing = ensure_qattn_baseline()
    jobs = []
    for name in CONFIGS:
        cfg = load_cfg(name)
        out_id = "R3_W2048_V5_Q_ATTN" if name.startswith("R3_W2048_V5") else cfg["id"]
        if out_id == "R3_W2048_V5_Q_ATTN" and not qattn_missing:
            print("SKIP Q_ATTN baseline (reuse R3 V5 metrics)", flush=True)
            continue
        for task in TASKS:
            if out_id == "R3_W2048_V5_Q_ATTN" and task not in qattn_missing:
                continue
            out = OUT_ROOT / out_id / task
            if (out / "metrics.json").exists():
                continue
            jobs.append((cfg, task, out, out_id))
    print(f"ACC pending {len(jobs)}", flush=True)
    slots = list(ACC_GPUS)
    running = []
    while jobs or running:
        running = [(p, gpu, name) for p, gpu, name in running if p.poll() is None]
        busy = {gpu for _, gpu, _ in running}
        while jobs and len(slots) > len(busy):
            gpu = next(g for g in slots if g not in busy)
            cfg, task, out, out_id = jobs.pop(0)
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
            running.append((proc, gpu, f"{out_id}:{task}"))
            busy.add(gpu)
            print(f"START gpu={gpu} {out_id} {task} pid={proc.pid}", flush=True)
        time.sleep(20)
    print("ACC queue drained", flush=True)


def main():
    run_acc()


if __name__ == "__main__":
    main()
