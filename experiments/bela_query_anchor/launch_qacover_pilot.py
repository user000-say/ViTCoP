"""G1 pilot: GQA + POPE for L2 native, QA-COVER, query-shuffle, matched-random.

GPUs 0/2/4/7. Does not start MMB, TextVQA, MME, or SciQA.
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

PKG = REPO / "experiments/bela_query_anchor"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_stage1_queryanchor_20261027"
LOG_DIR = PKG / "logs"
BASE_ARM = "E018_COVER_TWO32"
TASKS = ("gqa", "pope")
ARMS = ("L2_NATIVE", "S1_QANCHOR_L2", "S1_QUERY_SHUFFLE", "S1_MATCHED_RANDOM")


def env_for(cfg_path: Path, audit_dir: Path) -> dict:
    sys.path.insert(0, str(R3))
    from e014_arms import apply_arm_env
    from apply_llava import overlay

    os.environ["ELVA_R3_CONFIG"] = str(cfg_path)
    os.environ["E014_ARM"] = BASE_ARM
    meta = overlay(apply_arm_env(BASE_ARM))
    cfg = json.loads(cfg_path.read_text())
    qa = cfg["method"]["qa_cover"]
    anchor = qa["anchor"]
    keep = {}
    for key, value in os.environ.items():
        if key.startswith(("E0", "ELVA_", "USE_", "STAGE", "COMPUTE", "BELA_")) or key == "E014_ARM":
            keep[key] = value
    keep.update({
        "ELVA_R3_CONFIG": str(cfg_path),
        "ELVA_R3_AUDIT_DIR": str(audit_dir),
        "E014_ARM": meta["arm"],
        "BELA_QA_ENTRY_MODE": qa["entry_mode"],
        "BELA_QA_QUESTION_SOURCE": qa["question_source"],
        "BELA_QA_MAX_SWAPS": str(anchor["max_swaps"]),
        "BELA_QA_DONOR_POOL": str(anchor["donor_pool"]),
        "BELA_QA_DROP_POOL": str(anchor["drop_pool"]),
        "BELA_QA_MIN_RAW_STD": str(anchor["min_raw_std"]),
        "BELA_QA_MIN_QUERY_GAIN": str(anchor["min_query_gain"]),
        "BELA_QA_MIN_NOVELTY": str(anchor["min_novelty"]),
        "BELA_QA_MAX_COVERAGE_LOSS": str(anchor["max_coverage_loss"]),
        "BELA_QA_COVERAGE_WEIGHT": str(anchor["coverage_weight"]),
        "BELA_QA_SALIENCY_GUARD": str(anchor["saliency_guard_slots"]),
        "BELA_QA_SEED": str(qa.get("seed") or 17),
        "BELA_QA_CONTENT_MASK": "1",
        "BELA_QA_TRACE": "True",
        "BELA_QCP_MODE": "off",
        "BELA_ICER_MODE": "off",
    })
    return keep


def write_run_meta(out: Path, arm_id: str, cfg_path: Path, child: dict) -> None:
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    meta = {
        "arm": arm_id,
        "git": git,
        "config": str(cfg_path),
        "entry_k": child.get("ELVA_R3_ENTRY_K"),
        "final_k": child.get("ELVA_R3_FINAL_K"),
        "events": child.get("E018_EVENTS"),
        "qa_mode": child.get("BELA_QA_ENTRY_MODE"),
        "question_source": child.get("BELA_QA_QUESTION_SOURCE"),
        "content_mask": child.get("BELA_QA_CONTENT_MASK"),
        "max_swaps": child.get("BELA_QA_MAX_SWAPS"),
        "qcp_mode": child.get("BELA_QCP_MODE"),
        "note": "G1 pilot only. No accuracy claim until metrics.json exists.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm_id in ARMS:
        cfg = PKG / "configs" / f"{arm_id}.json"
        root = OUT_ROOT / arm_id
        for task in TASKS:
            if (root / task / "metrics.json").exists():
                print(f"HAVE {arm_id}/{task}", flush=True)
                continue
            jobs.append((cfg, arm_id, task, root / task))
    print(f"QA-COVER G1 pending {len(jobs)}", flush=True)
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
                leased = try_acquire(gpu, f"qacover:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS)
                if leased is not None:
                    break
            if leased is None:
                break
            cfg_path, arm_id, task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = LOG_DIR / f"acc_{arm_id}_{task}.log"
            child_env = os.environ.copy()
            child_env.update(env_for(cfg_path, out))
            child_env["CUDA_VISIBLE_DEVICES"] = str(leased.gpu)
            child_env["PYTHONPATH"] = (
                f"{REPO}:{REPO / 'experiments'}:{R3}:{PKG}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
            )
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["HF_HUB_OFFLINE"] = "1"
            child_env["HF_DATASETS_OFFLINE"] = "1"
            child_env["TRANSFORMERS_OFFLINE"] = "1"
            write_run_meta(out, arm_id, cfg_path, child_env)
            cmd = [
                PY, str(REPO / "e014_boot.py"),
                "--profile", "ANCHOR_A", "--task", task,
                "--output", str(out), "--gpu-index", str(leased.gpu), "--layer", "8",
            ]
            proc = subprocess.Popen(
                cmd, env=child_env, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True,
            )
            running.append((proc, leased, f"{arm_id}:{task}"))
            print(f"START gpu={leased.gpu} {arm_id} {task} pid={proc.pid}", flush=True)
        time.sleep(20)
    print("QA-COVER G1 drained", flush=True)
    missing = [
        f"{arm}:{task}" for arm in ARMS for task in TASKS
        if not (OUT_ROOT / arm / task / "metrics.json").exists()
    ]
    if missing:
        print("MISSING", missing, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
