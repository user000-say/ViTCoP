"""Wait for CDPruner K64+K32 6-task ACC to finish, then TTFT/Fixed32 for both K.

Uses GPUs {0,2,4,7} via gpu_exclusive; runs K64 then K32 TTFT sequentially
on one free card once ACC drained (avoids fighting remaining ACC jobs).
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

PKG = REPO / "experiments/cdpruner_x0"
PY = "/home/xiangyue/envs/starpro/bin/python"
OUT_ROOT = REPO / "logs/cdpruner_x0/full"
TTFT_ROOT = REPO / "logs/cdpruner_x0/ttft"
LOG_DIR = PKG / "logs"
TASKS6 = (
    "gqa",
    "pope",
    "mme",
    "mmbench_en_dev",
    "scienceqa_img",
    "textvqa_val_ocr",
)
KS = (64, 32)
DENSE = {
    "gqa": 61.9,
    "pope": 85.8,
    "mme": 1852.0,
    "mmbench_en_dev": 64.1,
    "scienceqa_img": 69.51,
    "textvqa_val_ocr": 58.20,
}


def abs_score(task: str, m: dict) -> float:
    if task == "pope":
        return 100.0 * float(m["pope_f1_score"])
    if task == "mme":
        return float(m["mme_perception_score"]) + float(m["mme_cognition_score"])
    if task == "mmbench_en_dev":
        return float(m["gpt_eval_score"])
    v = m["exact_match"]
    return 100.0 * float(v) if float(v) <= 1.5 else float(v)


def acc_complete() -> bool:
    for k in KS:
        for t in TASKS6:
            if not (OUT_ROOT / f"CDPRUNER_K{k}" / t / "metrics.json").exists():
                return False
    return True


def print_acc_table() -> None:
    for k in KS:
        scores = {}
        for t in TASKS6:
            p = OUT_ROOT / f"CDPRUNER_K{k}" / t / "metrics.json"
            scores[t] = abs_score(t, json.loads(p.read_text()))
        r4 = sum(100 * scores[t] / DENSE[t] for t in TASKS6[:4]) / 4
        m6 = sum(100 * scores[t] / DENSE[t] for t in TASKS6) / 6
        print(
            f"CDPRUNER_K{k} R4={r4:.2f} mean6={m6:.2f} { {t: round(scores[t], 2) for t in TASKS6} }",
            flush=True,
        )


def run_ttft(k: int, gpu: int) -> int:
    log = LOG_DIR / f"ttft_CDPRUNER_K{k}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = (
        f"{REPO / 'third_party/CDPruner'}:"
        f"{PKG}:{REPO / 'lmms-eval'}"
    )
    cmd = [
        PY,
        str(PKG / "bench_ttft.py"),
        "--visual-token-num",
        str(k),
        "--gpu-index",
        str(gpu),
    ]
    print(f"START TTFT K{k} gpu={gpu} log={log}", flush=True)
    with log.open("w") as handle:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd="/tmp",
        )
    print(f"DONE TTFT K{k} rc={proc.returncode}", flush=True)
    return int(proc.returncode)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("watch_acc_then_ttft: waiting for K64+K32 ACC metrics…", flush=True)
    while not acc_complete():
        have = sum(
            1
            for k in KS
            for t in TASKS6
            if (OUT_ROOT / f"CDPRUNER_K{k}" / t / "metrics.json").exists()
        )
        print(f"  ACC progress {have}/12", flush=True)
        time.sleep(60)
    print("ACC complete:", flush=True)
    print_acc_table()

    for k in KS:
        summary = TTFT_ROOT / f"CDPRUNER_K{k}" / "latency_summary.json"
        if summary.exists():
            print(f"HAVE TTFT K{k} {summary}", flush=True)
            continue
        while True:
            leased = None
            for gpu in PROJECT_GPUS:
                leased = try_acquire(gpu, f"cdp:ttft:K{k}", allowed=PROJECT_GPUS)
                if leased is not None:
                    break
            if leased is None:
                print("  no free GPU for TTFT; sleep 30s", flush=True)
                time.sleep(30)
                continue
            try:
                rc = run_ttft(k, leased.gpu)
            finally:
                release(leased)
            if rc != 0:
                raise SystemExit(f"TTFT K{k} failed rc={rc}")
            break

    print("=== TTFT / Fixed32 ===", flush=True)
    for k in KS:
        p = TTFT_ROOT / f"CDPRUNER_K{k}" / "latency_summary.json"
        if not p.exists():
            print(f"MISSING {p}", flush=True)
            continue
        d = json.loads(p.read_text())
        print(
            f"K{k}: TTFT={d['ttft_ms']['mean']:.1f} ms  Fixed32={d['fixed32_ms']['mean']:.1f} ms",
            flush=True,
        )
    print("watch_acc_then_ttft: drained", flush=True)


if __name__ == "__main__":
    main()
