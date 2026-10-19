"""Run R3 V1–V5 through the existing LLaVA scorer and first-token bench.

ACC uses GPUs 0, 2 and 4. TTFT keeps GPU 7. Physical ids 1, 3, 5 and 6 are never used.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
PYBENCH = str(REPO / "experiments/e014_latency_fair_256/bin/python")
BENCH = REPO / "experiments/e014_latency_fair_256/scripts/benchmark_llava_e2e_anygpu.py"
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
ACC_GPUS = (0, 2, 4)
TTFT_GPU = 7
TASKS = ("gqa", "pope", "mme", "mmbench_en_dev")
BASE_ARM = "E018_COVER_TWO32"


def configs():
    rows = []
    for path in sorted((PKG / "configs").glob("R3_*.json")):
        cfg = json.loads(path.read_text())
        if cfg.get("phase") == "stage1_core":
            rows.append(cfg)
    if len(rows) != 10:
        raise SystemExit(f"expected 10 core configs, found {len(rows)}")
    return rows


def legacy_layer(_cfg):
    # E014 still requires the frozen S=D guard value. The real prune layers are E018_EVENTS.
    return 8


def acc_jobs():
    jobs = []
    for cfg in configs():
        for task in TASKS:
            out = REPO / "logs/e014_aee/full/elva_r3_stage1" / cfg["id"] / task
            jobs.append((cfg, task, out))
    return jobs


def env_for(cfg, audit_dir: Path) -> dict:
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(PKG))
    from e014_arms import apply_arm_env
    from apply_llava import overlay
    os.environ["ELVA_R3_CONFIG"] = str(PKG / "configs" / f"{cfg['id']}.json")
    os.environ["E014_ARM"] = BASE_ARM
    meta = overlay(apply_arm_env(BASE_ARM))
    keep = {}
    for key, value in os.environ.items():
        if key.startswith(("E0", "ELVA_", "USE_", "STAGE", "COMPUTE")) or key in ("E014_ARM",):
            keep[key] = value
    keep["ELVA_R3_CONFIG"] = os.environ["ELVA_R3_CONFIG"]
    keep["ELVA_R3_AUDIT_DIR"] = str(audit_dir)
    keep["E014_ARM"] = meta["arm"]
    return keep


def main():
    mode = os.environ.get("ELVA_LAUNCH", "acc")
    if mode == "acc":
        run_acc()
    elif mode == "ttft":
        run_ttft()
    else:
        raise SystemExit(mode)


def run_acc():
    jobs = [j for j in acc_jobs() if not (j[2] / "metrics.json").exists()]
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
                "--layer", str(legacy_layer(cfg)),
            ]
            child_env = os.environ.copy()
            child_env.update(env_for(cfg, out))
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            child_env["PYTHONPATH"] = f"{REPO}:{REPO / 'experiments'}:{PKG}:{REPO / 'LLaVA'}"
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["HF_HUB_OFFLINE"] = "1"
            child_env["HF_DATASETS_OFFLINE"] = "1"
            child_env["TRANSFORMERS_OFFLINE"] = "1"
            handle = log.open("w")
            proc = subprocess.Popen(cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            running.append((proc, gpu, f"{cfg['id']}:{task}"))
            busy.add(gpu)
            print(f"START gpu={gpu} {cfg['id']} {task} pid={proc.pid}", flush=True)
        time.sleep(20)
    print("ACC queue drained", flush=True)


def run_ttft():
    for cfg in configs():
        out = PKG / "latency" / cfg["id"]
        if (out / "latency_summary.json").exists():
            print("SKIP", cfg["id"], flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        child_env.update(env_for(cfg, out))
        child_env["CUDA_VISIBLE_DEVICES"] = str(TTFT_GPU)
        child_env["PYTHONPATH"] = f"{REPO}:{REPO / 'experiments'}:{PKG}:{REPO / 'LLaVA'}"
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


if __name__ == "__main__":
    main()
