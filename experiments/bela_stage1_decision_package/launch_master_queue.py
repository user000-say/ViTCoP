#!/usr/bin/env python3
"""Single master queue: keep project GPUs busy, run pending ACC in priority order.

Priority (handoff order):
  1) P2 structure six-task gates (gqa/pope) then extras
  2) Top-B L2 fill-three (mmmu/ai2d/vizwiz) for B6/B8/B10
  3) Top-B E3 W2048 (hist E3-B8 six reused; B6/B10 full nine; B8 fill-three)

Behavior:
  - Rebuilds pending list every loop from disk + live e014 outs
  - try_acquire flock before spawn; releases on exit
  - After MAX_FAIL consecutive failures for the same arm:task, skip it
    (writes skip marker) so one broken job cannot pin an idle GPU forever
  - Polls every IDLE_SLEEP_S when work remains (default 8s)
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
from elva_acc_tasks import ACC_TASKS  # noqa: E402
from gpu_exclusive import PROJECT_GPUS as _DEFAULT_GPUS, release, try_acquire  # noqa: E402
from launch_qacover_pilot import PY, R3, PKG as QA_PKG, env_for, write_run_meta  # noqa: E402

DEC = REPO / "experiments/bela_stage1_decision_package"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_stage1_decision_20261029"
HIST = REPO / "logs/e014_aee/full/bela_stage1_queryanchor_20261027"
LOG_DIR = DEC / "logs"
CFG_DIR = DEC / "configs"
FAIL_DIR = LOG_DIR / "fail_counts"
SKIP_DIR = LOG_DIR / "skipped"

_env_gpus = os.environ.get("VITCOP_DECISION_GPUS", "").strip()
PROJECT_GPUS = (
    tuple(int(x) for x in _env_gpus.split(",") if x.strip() != "")
    if _env_gpus
    else _DEFAULT_GPUS
)
MAX_FAIL = int(os.environ.get("VITCOP_MASTER_MAX_FAIL", "3"))
IDLE_SLEEP_S = float(os.environ.get("VITCOP_MASTER_SLEEP", "8"))
# Comma-separated arm ids to run ASAP (still after LIVE skips). Example: P3_QA_QATTN
_PRIORITY_ARMS = {
    x.strip()
    for x in os.environ.get("VITCOP_PRIORITY_ARMS", "").split(",")
    if x.strip()
}

SIX = ("gqa", "pope", "mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")
GATE = ("gqa", "pope")
FILL3 = ("mmmu_val", "ai2d", "vizwiz_vqa_val")
TOP_B = (6, 8, 10)

P2_ARMS: list[tuple[str, Path, str | None]] = [
    ("L2_NATIVE", QA_PKG / "configs/L2_NATIVE.json", "L2_NATIVE"),  # S0 Native+RDRIFT
    ("S1_QANCHOR_L2", QA_PKG / "configs/S1_QANCHOR_L2.json", "S1_QANCHOR_L2"),  # S1 QA+RDRIFT
    ("S2_SALIENT_QA_B8", CFG_DIR / "S2_SALIENT_QA_B8.json", None),
    ("S3_JOINT", CFG_DIR / "S3_JOINT.json", None),
    ("S4_JOINT_CAP8", CFG_DIR / "S4_JOINT_CAP8.json", None),
    ("S5_QUERY_TOPK", CFG_DIR / "S5_QUERY_TOPK.json", None),
    ("S1_MATCHED_RANDOM", QA_PKG / "configs/S1_MATCHED_RANDOM.json", "S1_MATCHED_RANDOM"),  # S6 seed17
    ("S1_QUERY_SHUFFLE", QA_PKG / "configs/S1_QUERY_SHUFFLE.json", "S1_QUERY_SHUFFLE"),  # S7
    ("S8_NO_NOVELTY", CFG_DIR / "S8_NO_NOVELTY.json", None),
    ("S8_NO_COVERAGE", CFG_DIR / "S8_NO_COVERAGE.json", None),
    ("S8_QUERY_ONLY_GAIN", CFG_DIR / "S8_QUERY_ONLY_GAIN.json", None),
    # Handoff §E.1 pass schedules (cumulative B=8)
    ("S1_QA_PASSES_4_4", CFG_DIR / "S1_QA_PASSES_4_4.json", None),
    ("S1_QA_PASSES_3_3_2", CFG_DIR / "S1_QA_PASSES_3_3_2.json", None),
    # Handoff §F P3: Stage2 selector swap (Q_ATTN vs RDRIFT already in S0/S1)
    ("P3_NATIVE_QATTN", CFG_DIR / "P3_NATIVE_QATTN.json", None),
    ("P3_QA_QATTN", CFG_DIR / "P3_QA_QATTN.json", None),
    # Handoff S6 ≥5 seeds (17 covered by S1_MATCHED_RANDOM)
    ("S6_MATCHED_RANDOM_S23", CFG_DIR / "S6_MATCHED_RANDOM_S23.json", None),
    ("S6_MATCHED_RANDOM_S29", CFG_DIR / "S6_MATCHED_RANDOM_S29.json", None),
    ("S6_MATCHED_RANDOM_S31", CFG_DIR / "S6_MATCHED_RANDOM_S31.json", None),
    ("S6_MATCHED_RANDOM_S37", CFG_DIR / "S6_MATCHED_RANDOM_S37.json", None),
]


def _have(root: Path, task: str) -> bool:
    return (root / task / "metrics.json").exists()


def _metric(write: Path, hist_alias: str | None, task: str) -> Path | None:
    if _have(write, task):
        return write / task / "metrics.json"
    if hist_alias and _have(HIST / hist_alias, task):
        return HIST / hist_alias / task / "metrics.json"
    return None


def _live_outputs() -> set[str]:
    live: set[str] = set()
    try:
        out = subprocess.check_output(["ps", "-eo", "args="], text=True)
    except Exception:
        return live
    needle = str(OUT_ROOT) + "/"
    for line in out.splitlines():
        if "e014_boot.py" not in line or "--output" not in line:
            continue
        parts = line.split()
        for i, p in enumerate(parts):
            if p == "--output" and i + 1 < len(parts):
                path = parts[i + 1]
                if path.startswith(needle):
                    live.add(path.rstrip("/"))
                break
    return live


def _fail_path(arm: str, task: str) -> Path:
    return FAIL_DIR / f"{arm}__{task}.count"


def _skip_path(arm: str, task: str) -> Path:
    return SKIP_DIR / f"{arm}__{task}.skip"


def _is_skipped(arm: str, task: str) -> bool:
    return _skip_path(arm, task).exists()


def _bump_fail(arm: str, task: str, rc: int) -> int:
    FAIL_DIR.mkdir(parents=True, exist_ok=True)
    SKIP_DIR.mkdir(parents=True, exist_ok=True)
    # Never auto-skip known interface/identity errors — hold for human fix.
    log = LOG_DIR / f"acc_{arm}_{task}.log"
    if log.exists():
        try:
            tail = log.read_text(errors="ignore")[-8000:]
        except Exception:
            tail = ""
        if "duplicate sample_id" in tail:
            sp = _skip_path(arm, task)
            sp.write_text(
                json.dumps(
                    {
                        "arm": arm,
                        "task": task,
                        "kind": "HOLD_INTERFACE",
                        "reason": "duplicate sample_id — identity/eval interface bug, not effect failure",
                        "last_rc": rc,
                        "ts": time.time(),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"HOLD_INTERFACE {arm}:{task} (duplicate sample_id) — paused, not counted as ACC fail", flush=True)
            return 0
    p = _fail_path(arm, task)
    n = int(p.read_text().strip() or "0") if p.exists() else 0
    n += 1
    p.write_text(str(n) + "\n")
    if n >= MAX_FAIL:
        sp = _skip_path(arm, task)
        sp.write_text(
            json.dumps(
                {"arm": arm, "task": task, "kind": "SKIP_FAIL", "fails": n, "last_rc": rc, "ts": time.time()},
                indent=2,
            )
            + "\n"
        )
        print(f"SKIP_FAIL {arm}:{task} after {n} fails rc={rc}", flush=True)
    return n


def _clear_fail(arm: str, task: str) -> None:
    p = _fail_path(arm, task)
    if p.exists():
        p.unlink()


def _jobs(running_names: set[str]) -> list[tuple[Path, str, str, Path, int]]:
    """Return (cfg, arm, task, out, priority). Lower priority first."""
    live = _live_outputs()
    jobs: list[tuple[Path, str, str, Path, int]] = []

    def add(cfg: Path, arm: str, task: str, out: Path, hist: str | None, prio: int) -> None:
        key = f"{arm}:{task}"
        if key in running_names:
            return
        if _is_skipped(arm, task):
            print(f"SKIPPED {arm}/{task}", flush=True)
            return
        write = out.parent
        if _metric(write, hist, task) is not None:
            return
        if str(out) in live:
            print(f"LIVE {arm}/{task}", flush=True)
            return
        if not cfg.exists():
            print(f"MISSING_CFG {cfg}", flush=True)
            return
        jobs.append((cfg, arm, task, out, prio))

    # --- 1) P2 structure ---
    for arm, cfg, hist in P2_ARMS:
        write = OUT_ROOT / arm
        for task in SIX:
            # gates before extras: prio 10 + gate_boost
            gate_boost = 0 if task in GATE else 100
            prio = 10 + gate_boost
            # User-requested priority arms (e.g. finish P3_QA_QATTN six-task now)
            if arm in _PRIORITY_ARMS:
                prio = 1 if task in GATE else 2
            add(cfg, arm, task, write / task, hist, prio)

    # --- 2) Top-B L2 fill-three ---
    for b in TOP_B:
        arm = f"S1_QANCHOR_L2_B{b}"
        cfg = CFG_DIR / f"{arm}.json"
        write = OUT_ROOT / arm
        for task in FILL3:
            add(cfg, arm, task, write / task, None, 200)

    # --- 3) Top-B E3 W2048 ---
    for b in TOP_B:
        arm = f"S1_QANCHOR_E3_B{b}"
        cfg = CFG_DIR / f"{arm}.json"
        write = OUT_ROOT / arm
        hist = "S1_QANCHOR_E3" if b == 8 else None
        tasks = FILL3 if b == 8 else ACC_TASKS
        for task in tasks:
            gate_boost = 0 if task in GATE else 50
            add(cfg, arm, task, write / task, hist, 300 + gate_boost)

    jobs.sort(key=lambda j: (j[4], j[1], j[2]))
    return jobs


def _child_env(cfg_path: Path, out: Path) -> dict:
    child = os.environ.copy()
    child.update(env_for(cfg_path, out))
    cfg = json.loads(cfg_path.read_text())
    init = cfg.get("method", {}).get("stage1_init", "cover")
    child["BELA_STAGE1_INIT"] = init
    qa = (cfg.get("method") or {}).get("qa_cover") or {}
    caps = qa.get("pass_caps")
    if caps:
        child["BELA_QA_PASS_CAPS"] = ",".join(str(int(x)) for x in caps)
    else:
        child.pop("BELA_QA_PASS_CAPS", None)
    child["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{QA_PKG}:{DEC}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
    )
    child["PYTHONUNBUFFERED"] = "1"
    child["HF_HUB_OFFLINE"] = "1"
    child["HF_DATASETS_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    return child


def _free_gpus(held: set[int]) -> list[int]:
    out = []
    for g in PROJECT_GPUS:
        if g in held:
            continue
        try:
            used = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={g}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            )
        except Exception:
            used = 99999
        if used < 800:
            out.append(g)
    return out


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FAIL_DIR.mkdir(parents=True, exist_ok=True)
    SKIP_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"MASTER_QUEUE gpus={PROJECT_GPUS} max_fail={MAX_FAIL} sleep={IDLE_SLEEP_S}s",
        flush=True,
    )

    running: list[tuple[subprocess.Popen, object, str, str, str]] = []
    # (proc, lease, name, arm, task)
    while True:
        still = []
        for proc, lease, name, arm, task in running:
            if proc.poll() is None:
                still.append((proc, lease, name, arm, task))
            else:
                rc = proc.returncode
                release(lease)  # type: ignore[arg-type]
                if rc == 0:
                    _clear_fail(arm, task)
                    print(f"DONE {name} rc=0", flush=True)
                else:
                    n = _bump_fail(arm, task, rc if rc is not None else -1)
                    print(f"FAIL {name} rc={rc} fails={n}/{MAX_FAIL}", flush=True)
        running = still
        running_names = {name for _, _, name, _, _ in running}
        held = {lease.gpu for _, lease, _, _, _ in running}  # type: ignore[attr-defined]
        jobs = _jobs(running_names)
        print(
            f"PENDING {len(jobs)} running={len(running)} free_cand={_free_gpus(held)}",
            flush=True,
        )
        if not jobs and not running:
            print("MASTER_QUEUE_DRAINED", flush=True)
            break

        # Fill every free GPU immediately.
        while jobs:
            free = _free_gpus(held)
            if not free:
                break
            leased = None
            gpu_used = None
            for gpu in free:
                leased = try_acquire(gpu, f"master:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS)
                if leased is not None:
                    gpu_used = gpu
                    break
            if leased is None:
                break
            cfg_path, arm_id, task, out, prio = jobs.pop(0)
            # Clean partial out from prior failed attempts (no metrics).
            if out.exists() and not (out / "metrics.json").exists():
                # keep logs elsewhere; wipe work dir to avoid duplicate sample_id residue
                import shutil

                shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True, exist_ok=True)
            log = LOG_DIR / f"acc_{arm_id}_{task}.log"
            child_env = _child_env(cfg_path, out)
            child_env["CUDA_VISIBLE_DEVICES"] = str(leased.gpu)
            write_run_meta(out, arm_id, cfg_path, child_env)
            if "_E3_" in arm_id:
                assert child_env.get("ELVA_R3_ENTRY_K") == "172", child_env.get("ELVA_R3_ENTRY_K")
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
                str(leased.gpu),
                "--layer",
                "8",
            ]
            # append logs across retries
            logf = log.open("a")
            logf.write(f"\n===== START {time.strftime('%F %T')} gpu={leased.gpu} prio={prio} =====\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                env=child_env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            name = f"{arm_id}:{task}"
            running.append((proc, leased, name, arm_id, task))
            held.add(leased.gpu)
            print(
                f"START gpu={leased.gpu} {name} prio={prio} init={child_env.get('BELA_STAGE1_INIT')} "
                f"qa={child_env.get('BELA_QA_ENTRY_MODE')} swaps={child_env.get('BELA_QA_MAX_SWAPS')} pid={proc.pid}",
                flush=True,
            )

        time.sleep(IDLE_SLEEP_S)


if __name__ == "__main__":
    main()
