"""QA-COVER on native E3 / W2048, after the W1024 arm has finished.

Does not call try_acquire until S1_QANCHOR_L2 has all six metrics and the
W1024 protect launcher is gone. Same anchor thresholds as S1_QANCHOR_L2.
Schedule is native E3: 172/120/24/16, lengths 4/8/10/10, budget 2048.
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
sys.path.insert(0, str(REPO / "experiments/bela_query_anchor"))
from gpu_exclusive import PROJECT_GPUS, release, try_acquire  # noqa: E402
from launch_qacover_pilot import LOG_DIR, OUT_ROOT, PKG, PY, R3, env_for, write_run_meta  # noqa: E402

ARM = "S1_QANCHOR_E3"
L2_ARM = "S1_QANCHOR_L2"
L2_TASKS = (
    "gqa",
    "pope",
    "mme",
    "mmbench_en_dev",
    "scienceqa_img",
    "textvqa_val_ocr",
)
TASKS = L2_TASKS
EXPECTED_ENTRY = "172"
EXPECTED_FINAL = "16"
EXPECTED_EVENTS = "3:120,11:24,21:16"
SMOKE = REPO / "logs/e014_aee/smoke/bela_qacover_e3_20261028" / ARM


def _metrics(arm: str, task: str) -> Path:
    return OUT_ROOT / arm / task / "metrics.json"


def _boot_alive(out: Path) -> bool:
    needle = str(out)
    try:
        listing = subprocess.check_output(["ps", "-u", "xiangyue", "-o", "cmd"], text=True)
    except subprocess.CalledProcessError:
        return False
    return any("e014_boot.py" in line and needle in line for line in listing.splitlines())


def _protect_alive() -> bool:
    try:
        out = subprocess.check_output(["ps", "-u", "xiangyue", "-o", "cmd"], text=True)
    except subprocess.CalledProcessError:
        return False
    for line in out.splitlines():
        if "launch_qacover_protect.py" in line:
            return True
        if "e014_boot.py" in line and f"/{L2_ARM}/" in line:
            return True
    return False


def wait_for_w1024() -> None:
    while True:
        missing = [task for task in L2_TASKS if not _metrics(L2_ARM, task).exists()]
        alive = _protect_alive()
        if not missing and not alive:
            print("W1024 S1_QANCHOR_L2 six metrics present and protect launcher idle", flush=True)
            return
        print(
            f"WAIT w1024 missing={missing or '-'} protect_alive={alive}; not acquiring GPUs",
            flush=True,
        )
        time.sleep(60)


def _checked_env(out: Path) -> dict:
    cfg = PKG / "configs" / f"{ARM}.json"
    child_env = os.environ.copy()
    child_env.update(env_for(cfg, out))
    if child_env.get("ELVA_R3_ENTRY_K") != EXPECTED_ENTRY:
        raise SystemExit(f"entry K {child_env.get('ELVA_R3_ENTRY_K')} != {EXPECTED_ENTRY}")
    if child_env.get("ELVA_R3_FINAL_K") != EXPECTED_FINAL:
        raise SystemExit(f"final K {child_env.get('ELVA_R3_FINAL_K')} != {EXPECTED_FINAL}")
    if child_env.get("E018_EVENTS") != EXPECTED_EVENTS:
        raise SystemExit(f"events {child_env.get('E018_EVENTS')} != {EXPECTED_EVENTS}")
    if child_env.get("BELA_QA_MAX_SWAPS") != "8" or child_env.get("BELA_QA_ENTRY_MODE") != "s1_anchor":
        raise SystemExit("QA-COVER hyperparameters drifted from the W1024 arm")
    child_env["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{PKG}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
    )
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["HF_DATASETS_OFFLINE"] = "1"
    child_env["TRANSFORMERS_OFFLINE"] = "1"
    return child_env


def _spawn(task: str, out: Path, log: Path, leased, limit: int | None) -> subprocess.Popen:
    out.mkdir(parents=True, exist_ok=True)
    cfg = PKG / "configs" / f"{ARM}.json"
    child_env = _checked_env(out)
    child_env["CUDA_VISIBLE_DEVICES"] = str(leased.gpu)
    write_run_meta(out, ARM, cfg, child_env)
    meta_path = out / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["note"] = "W2048 E3 QA-COVER. Starts only after W1024 six-task metrics exist."
    meta["budget"] = 2048
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    cmd = [
        PY, str(REPO / "e014_boot.py"),
        "--profile", "ANCHOR_A", "--task", task,
        "--output", str(out), "--gpu-index", str(leased.gpu), "--layer", "8",
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd, env=child_env, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True,
    )
    print(f"START gpu={leased.gpu} {ARM} {task} pid={proc.pid} limit={limit or 'full'}", flush=True)
    return proc


def _smoke_ok() -> None:
    trace = SMOKE / "entry_trace.jsonl"
    if not trace.exists():
        raise SystemExit(f"smoke produced no entry trace at {trace}")
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    if len(rows) < 1:
        raise SystemExit("smoke entry trace is empty")
    for row in rows:
        n = len(row.get("final_ids") or [])
        if n != 172:
            raise SystemExit(f"smoke entry keep is {n}, expected 172")
    print(f"SMOKE ok rows={len(rows)} entry_k=172", flush=True)


def _run_smoke() -> None:
    if (SMOKE / "entry_trace.jsonl").exists():
        _smoke_ok()
        return
    while True:
        leased = None
        for gpu in PROJECT_GPUS:
            leased = try_acquire(gpu, f"qacover:{ARM}:smoke", allowed=PROJECT_GPUS)
            if leased is not None:
                break
        if leased is not None:
            break
        print(f"WAIT gpu for {ARM} smoke; cards in {PROJECT_GPUS} are busy", flush=True)
        time.sleep(30)
    proc = _spawn("gqa", SMOKE, LOG_DIR / "smoke_S1_QANCHOR_E3.log", leased, limit=2)
    rc = proc.wait()
    release(leased)
    print(f"DONE {ARM}:smoke rc={rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)
    _smoke_ok()


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"QA-COVER E3 queued. Will not touch {PROJECT_GPUS} until W1024 finishes.", flush=True)
    wait_for_w1024()
    _run_smoke()
    root = OUT_ROOT / ARM
    jobs = []
    for task in TASKS:
        out = root / task
        if (out / "metrics.json").exists():
            print(f"HAVE {ARM}/{task}", flush=True)
            continue
        if _boot_alive(out):
            print(f"ALREADY {ARM}/{task}; not starting a second copy", flush=True)
            continue
        jobs.append((task, out))
    running: list[tuple[subprocess.Popen, object, str]] = []
    while jobs or running:
        still = []
        for proc, lease, name in running:
            if proc.poll() is None:
                still.append((proc, lease, name))
            else:
                release(lease)  # type: ignore[arg-type]
                print(f"DONE {name} rc={proc.returncode}", flush=True)
        running = still
        while jobs:
            leased = None
            for gpu in PROJECT_GPUS:
                if any(g.gpu == gpu for _, g, _ in running):  # type: ignore[attr-defined]
                    continue
                leased = try_acquire(gpu, f"qacover:{ARM}:{jobs[0][0]}", allowed=PROJECT_GPUS)
                if leased is not None:
                    break
            if leased is None:
                break
            task, out = jobs.pop(0)
            proc = _spawn(task, out, LOG_DIR / f"acc_{ARM}_{task}.log", leased, limit=None)
            running.append((proc, leased, f"{ARM}:{task}"))
        time.sleep(20)
    pending = [task for task in TASKS if not _metrics(ARM, task).exists()]
    while pending and any(_boot_alive(root / task) for task in pending):
        print(f"WAIT in-flight {pending}", flush=True)
        time.sleep(30)
        pending = [task for task in TASKS if not _metrics(ARM, task).exists()]
    missing = [task for task in TASKS if not _metrics(ARM, task).exists()]
    if missing:
        print("MISSING", missing, flush=True)
        raise SystemExit(1)
    print("QA-COVER E3 six-task drained", flush=True)


if __name__ == "__main__":
    main()
