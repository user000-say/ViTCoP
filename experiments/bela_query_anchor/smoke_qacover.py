"""One-GPU smoke: off regression, then the real QA-COVER arm. limit=2 GQA."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/bela_query_anchor"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
SMOKE = REPO / "logs/e014_aee/smoke/bela_qacover_20261027"
BASE_ARM = "E018_COVER_TWO32"
GPU = 0


def env_for(cfg_path: Path, audit_dir: Path) -> dict:
    sys.path.insert(0, str(REPO))
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


def run_one(arm_id: str) -> dict:
    cfg = PKG / "configs" / f"{arm_id}.json"
    out = SMOKE / arm_id
    out.mkdir(parents=True, exist_ok=True)
    log = PKG / "logs" / f"smoke_{arm_id}.log"
    child = os.environ.copy()
    child.update(env_for(cfg, out))
    child["CUDA_VISIBLE_DEVICES"] = str(GPU)
    child["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{PKG}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
    )
    child["PYTHONUNBUFFERED"] = "1"
    child["HF_HUB_OFFLINE"] = child["HF_DATASETS_OFFLINE"] = child["TRANSFORMERS_OFFLINE"] = "1"
    cmd = [
        PY, str(REPO / "e014_boot.py"),
        "--profile", "ANCHOR_A", "--task", "gqa",
        "--output", str(out), "--gpu-index", str(GPU),
        "--layer", "8", "--limit", "2",
    ]
    print(
        f"SMOKE {arm_id} gpu={GPU} mode={child['BELA_QA_ENTRY_MODE']} "
        f"events={child.get('E018_EVENTS')} entry={child.get('ELVA_R3_ENTRY_K')}",
        flush=True,
    )
    proc = subprocess.run(cmd, env=child, stdout=log.open("w"), stderr=subprocess.STDOUT)
    report = {"arm": arm_id, "returncode": proc.returncode, "log": str(log)}
    audit_path = out / "audit.json"
    if audit_path.exists():
        report["audit"] = json.loads(audit_path.read_text())
    trace_path = out / "entry_trace.jsonl"
    if trace_path.exists():
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        report["trace_rows"] = len(rows)
        report["swaps"] = [row.get("swaps") for row in rows]
        report["reasons"] = [row.get("reason") for row in rows]
        report["ks"] = [len(row.get("final_ids") or []) for row in rows]
    else:
        report["trace_rows"] = 0
    survival = out / "qa_survival.jsonl"
    if survival.exists():
        rows = [json.loads(line) for line in survival.read_text().splitlines() if line.strip()]
        report["survival_final_k"] = [row.get("final_k") for row in rows]
    if proc.returncode != 0:
        report["log_tail"] = log.read_text(encoding="utf-8", errors="replace")[-2000:]
    return report


def main() -> None:
    (PKG / "logs").mkdir(parents=True, exist_ok=True)
    SMOKE.mkdir(parents=True, exist_ok=True)
    reports = [run_one("L2_NATIVE"), run_one("S1_QANCHOR_L2")]
    path = SMOKE / "QA_COVER_SMOKE.json"
    path.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2), flush=True)
    failed = []
    for report in reports:
        audit = report.get("audit") or {}
        stage1 = audit.get("stage1_tokens")
        final = audit.get("final_visual_tokens")
        events = audit.get("e018_events")
        if report["returncode"] != 0 or stage1 != 64 or final != 10:
            failed.append(report["arm"])
        if report["arm"] == "L2_NATIVE" and report.get("trace_rows", 0) != 0:
            failed.append("L2_NATIVE_unexpected_trace")
        if report["arm"] == "S1_QANCHOR_L2":
            if report.get("trace_rows") != 2 or any(k != 64 for k in report.get("ks") or []):
                failed.append("S1_trace")
            if report.get("survival_final_k") != [10, 10]:
                failed.append("S1_survival")
        ev = (report.get("audit") or {}).get("e018_events")
        if ev != [[11, 17], [19, 10]]:
            failed.append(f"events:{report['arm']}:{ev}")
    if failed:
        raise SystemExit(f"smoke failed: {failed}")
    print("ALL_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
