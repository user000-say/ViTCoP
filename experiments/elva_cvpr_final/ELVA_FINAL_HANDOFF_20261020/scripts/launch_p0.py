"""CVPR P0: W2048 V4/V5+RDRIFT and W1024 K80 equal-W schedules.

ACC on physical GPUs 0, 2, 4. TTFT on GPU 7. Never use 1/3/5/6.
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
PYBENCH = str(REPO / "experiments/e014_latency_fair_256/bin/python")
BENCH = REPO / "experiments/e014_latency_fair_256/scripts/benchmark_llava_e2e_anygpu.py"
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
ACC_GPUS = (0, 2, 4)
TTFT_GPU = 7
TASKS = ("gqa", "pope", "mme", "mmbench_en_dev")
BASE_ARM = "E018_COVER_TWO32"
CONFIGS = (
    "H1_W2048_V4_COVER_RDRIFT",
    "H2_W2048_V5_COVER_RDRIFT",
    "E94_K80_48_16_PEN_PCOV_B12",
    "E94_K80_64_16_PEN_PCOV_B12",
)


def load_cfg(name: str) -> dict:
    return json.loads((PKG / "configs" / f"{name}.json").read_text())


def env_for(cfg: dict, audit_dir: Path) -> dict:
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(R3))
    from e014_arms import apply_arm_env
    from apply_llava import overlay
    cfg_path = PKG / "configs" / f"{cfg['id']}.json"
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


def run_acc():
    jobs = []
    for name in CONFIGS:
        cfg = load_cfg(name)
        for task in TASKS:
            out = REPO / "logs/e014_aee/full/elva_cvpr_p0" / cfg["id"] / task
            if not (out / "metrics.json").exists():
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


def run_ttft():
    for name in CONFIGS:
        cfg = load_cfg(name)
        out = PKG / "latency" / cfg["id"]
        if (out / "latency_summary.json").exists():
            print("SKIP", cfg["id"], flush=True)
            continue
        if out.exists():
            for child in out.iterdir():
                child.unlink()
        out.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        child_env.update(env_for(cfg, out))
        child_env["CUDA_VISIBLE_DEVICES"] = str(TTFT_GPU)
        child_env["PYTHONPATH"] = (
            f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
        )
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        child_env["HF_DATASETS_OFFLINE"] = "1"
        cmd = [
            PYBENCH, str(BENCH),
            "--repo-root", str(REPO), "--panel", str(PANEL),
            "--model-path", str(MODEL), "--conv-mode", "llava_v1",
            "--out-dir", str(out), "--warmups", "8", "--limit", "0",
        ]
        log = PKG / "logs" / f"ttft_{cfg['id']}.log"
        print(f"TTFT gpu={TTFT_GPU} {cfg['id']}", flush=True)
        with log.open("w") as handle:
            subprocess.check_call(cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
        print("TTFT done", cfg["id"], flush=True)


def main():
    mode = os.environ.get("ELVA_LAUNCH", "acc")
    if mode == "acc":
        run_acc()
    elif mode == "ttft":
        run_ttft()
    else:
        raise SystemExit(mode)


if __name__ == "__main__":
    main()
