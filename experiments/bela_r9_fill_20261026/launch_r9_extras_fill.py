"""Fill missing R9 extras (mmmu_val, ai2d, vizwiz) for competitive arms.

Skip arms that already have metrics.json. Official 9-task set from elva_acc_tasks.
GPUs {0,2,4,7} via gpu_exclusive. Do not regenerate completed 6-task metrics.

Arms:
  BELA: E3, E1, E5 (W2048), L2 (W1024) — prior strong / comparable schedules
  STAR: T32 only (T64 already full R9)
  CDPruner: K64, K32
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
sys.path.insert(0, str(REPO / "experiments"))
from gpu_exclusive import PROJECT_GPUS, release, try_acquire  # noqa: E402

R3 = REPO / "experiments/elva_r3"
PKG = REPO / "experiments/bela_r9_fill_20261026"
CFG_DIR = REPO / "experiments/bela_cvpr27_final_handoff_20261025/configs/p1_arms"
LOG_DIR = PKG / "logs"
PY_BELA = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
PY_STAR = "/home/xiangyue/envs/starpro/bin/python"
BASE_ARM = "E018_COVER_TWO32"

EXTRAS = ("mmmu_val", "ai2d", "vizwiz_vqa_val")

BELA_ARMS = (
    ("E3_RET24_TO16", REPO / "logs/e014_aee/full/bela_final_p1_20261025/E3_RET24_TO16"),
    ("E1_RET32_TO12", REPO / "logs/e014_aee/full/bela_final_p1_20261025/E1_RET32_TO12"),
    ("E5_DELAY_128_12_20", REPO / "logs/e014_aee/full/bela_final_p1_20261025/E5_DELAY_128_12_20"),
    ("L2_LATE_64_17_10", REPO / "logs/e014_aee/full/bela_final_p1_20261025/L2_LATE_64_17_10"),
)

STAR_ARMS = (
    ("STARPRO_T32", 32, REPO / "logs/starpro_x0_t64/full/STARPRO_T32"),
)

CD_ARMS = (
    (64, REPO / "logs/cdpruner_x0/full/CDPRUNER_K64"),
    (32, REPO / "logs/cdpruner_x0/full/CDPRUNER_K32"),
)


def bela_env(cfg_path: Path, audit_dir: Path) -> dict:
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


def build_jobs() -> list[tuple]:
    """Each job: (kind, name, task, out_dir, payload)."""
    jobs = []
    for arm_id, root in BELA_ARMS:
        cfg = CFG_DIR / f"{arm_id}.json"
        if not cfg.exists():
            raise FileNotFoundError(cfg)
        for task in EXTRAS:
            out = root / task
            if (out / "metrics.json").exists():
                print(f"HAVE BELA {arm_id}/{task}", flush=True)
                continue
            jobs.append(("bela", arm_id, task, out, cfg))
    for arm_id, T, root in STAR_ARMS:
        for task in EXTRAS:
            out = root / task
            if (out / "metrics.json").exists():
                print(f"HAVE STAR {arm_id}/{task}", flush=True)
                continue
            jobs.append(("star", arm_id, task, out, T))
    for k, root in CD_ARMS:
        for task in EXTRAS:
            out = root / task
            if (out / "metrics.json").exists():
                print(f"HAVE CD K{k}/{task}", flush=True)
                continue
            jobs.append(("cd", f"CDPRUNER_K{k}", task, out, k))
    return jobs


def start_job(kind: str, name: str, task: str, out: Path, payload, gpu: int) -> subprocess.Popen:
    out.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"acc_{name}_{task}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("HF_DATASETS_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)

    if kind == "bela":
        cfg_path: Path = payload
        env.update(bela_env(cfg_path, out))
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONPATH"] = (
            f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
        )
        cmd = [
            PY_BELA,
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
        py = PY_BELA
    elif kind == "star":
        T = int(payload)
        env["PYTHONPATH"] = (
            f"{REPO / 'third_party/LLaVA-starpro'}:"
            f"{REPO / 'experiments/starpro_x0_t64'}:{REPO / 'lmms-eval'}"
        )
        cmd = [
            PY_STAR,
            str(REPO / "experiments/starpro_x0_t64/run_one_task.py"),
            "--task",
            task,
            "--gpu-index",
            str(gpu),
            "--T",
            str(T),
        ]
        py = PY_STAR
    elif kind == "cd":
        k = int(payload)
        env["PYTHONPATH"] = (
            f"{REPO / 'third_party/CDPruner'}:"
            f"{REPO / 'experiments/cdpruner_x0'}:{REPO / 'lmms-eval'}"
        )
        cmd = [
            PY_STAR,
            str(REPO / "experiments/cdpruner_x0/run_one_task.py"),
            "--task",
            task,
            "--gpu-index",
            str(gpu),
            "--visual-token-num",
            str(k),
        ]
        py = PY_STAR
    else:
        raise ValueError(kind)

    handle = log.open("w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd="/tmp",
    )
    print(f"START gpu={gpu} {kind}:{name}:{task} pid={proc.pid} py={py}", flush=True)
    return proc


def main() -> None:
    jobs = build_jobs()
    print(f"R9 extras pending {len(jobs)}", flush=True)
    print(json.dumps([f"{k}:{n}:{t}" for k, n, t, _, _ in jobs], indent=2), flush=True)
    running: list[tuple[subprocess.Popen, object, str]] = []
    while jobs or running:
        still = []
        for proc, lease, label in running:
            if proc.poll() is None:
                still.append((proc, lease, label))
            else:
                release(lease)  # type: ignore[arg-type]
                print(f"DONE {label} rc={proc.returncode}", flush=True)
        running = still
        while jobs:
            leased = None
            for gpu in PROJECT_GPUS:
                if any(g.gpu == gpu for _, g, _ in running):  # type: ignore[attr-defined]
                    continue
                leased = try_acquire(
                    gpu, f"r9:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS
                )
                if leased is not None:
                    break
            if leased is None:
                break
            kind, name, task, out, payload = jobs.pop(0)
            proc = start_job(kind, name, task, out, payload, leased.gpu)
            running.append((proc, leased, f"{kind}:{name}:{task}"))
        time.sleep(20)
    print("R9 extras queue drained", flush=True)
    missing = []
    for kind, name, task, out, _ in build_jobs():
        # rebuild only reports still-missing after drain via fresh scan
        pass
    # fresh scan
    for arm_id, root in BELA_ARMS:
        for task in EXTRAS:
            if not (root / task / "metrics.json").exists():
                missing.append(f"BELA:{arm_id}:{task}")
    for arm_id, _T, root in STAR_ARMS:
        for task in EXTRAS:
            if not (root / task / "metrics.json").exists():
                missing.append(f"STAR:{arm_id}:{task}")
    for k, root in CD_ARMS:
        for task in EXTRAS:
            if not (root / task / "metrics.json").exists():
                missing.append(f"CD:K{k}:{task}")
    if missing:
        print("MISSING after drain:", missing, flush=True)
        raise SystemExit(1)
    print("ALL R9 extras present for queued arms", flush=True)


if __name__ == "__main__":
    main()
