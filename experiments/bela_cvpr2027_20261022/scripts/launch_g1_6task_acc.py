"""CVPR2027 G1: F2 ENTRY160 + F3 ENTRY128, 6-task ACC only.

Tasks: R4 four + scienceqa_img + textvqa_val_ocr.
GPUs 0/2/4/7. Reuses Gate1 metrics for F0/F1 into OUT_ROOT.
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

TASKS = (
    "gqa",
    "pope",
    "mme",
    "mmbench_en_dev",
    "scienceqa_img",
    "textvqa_val_ocr",
)

PKG = REPO / "experiments/bela_cvpr2027_20261022"
HANDOFF = PKG / "BELA_ELVA_CVPR_Cursor_Handoff_20261022"
R3 = REPO / "experiments/elva_r3"
GATE1 = REPO / "logs/e014_aee/full/bela_gate1_8task"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
ACC_GPUS = (0, 2, 4, 7)
BASE_ARM = "E018_COVER_TWO32"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_cvpr2027_g1_6task"

ARMS = (
    (
        PKG / "configs/F2_ENTRY160_RDRIFT_W2048.json",
        "F2_ENTRY160_RDRIFT",
        None,
    ),
    (
        PKG / "configs/F3_ENTRY128_RDRIFT_W2048.json",
        "F3_ENTRY128_RDRIFT",
        None,
    ),
)

# P0 anchors: symlink prior Gate1 metrics (same 6 tasks already complete).
REUSE_P0 = (
    ("F0_COVER_QATTN", GATE1 / "W2048_V5_COVER_QATTN"),
    ("F1_COVER_RDRIFT", GATE1 / "W2048_V5_COVER_RDRIFT"),
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


def seed_p0() -> None:
    for out_id, src in REUSE_P0:
        root = OUT_ROOT / out_id
        root.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            if link_reuse(src, root, task):
                print(f"HAVE {out_id}/{task}", flush=True)


def run_acc(limit: int = 0) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_p0()
    jobs = []
    for cfg_path, out_id, reuse in ARMS:
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        cfg = json.loads(cfg_path.read_text())
        root = OUT_ROOT / out_id
        root.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            if limit == 0 and link_reuse(reuse, root, task):
                print(f"HAVE {out_id}/{task}", flush=True)
                continue
            out = root / task
            if limit == 0 and (out / "metrics.json").exists():
                print(f"HAVE {out_id}/{task}", flush=True)
                continue
            jobs.append((cfg_path, cfg, out_id, task, out))
    print(
        f"ACC pending {len(jobs)} / arms×tasks={len(ARMS) * len(TASKS)} limit={limit}",
        flush=True,
    )
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
            if limit:
                cmd.extend(["--limit", str(limit)])
            child_env = os.environ.copy()
            child_env.update(env_for(cfg_path, out))
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            child_env["PYTHONPATH"] = (
                f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:"
                f"{REPO / 'e020_lib'}:{HANDOFF}"
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
    fails = []
    for p, gpu, name in running:
        rc = p.wait()
        if rc != 0:
            fails.append((name, rc))
    if fails:
        raise SystemExit(f"ACC failures: {fails}")
    print("ACC queue drained", flush=True)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_acc(limit=lim)
