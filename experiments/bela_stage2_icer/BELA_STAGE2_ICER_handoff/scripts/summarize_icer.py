"""Summarize ICER 9-task ACC with R4/R8/R9 vs Gate1 baselines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
sys.path.insert(0, str(REPO))
from elva_acc_tasks import ACC_TASKS, EXTRA_TASKS, R4_TASKS
from elva_scores import format_relative_table, score_task_from_metrics, summarize_arm

PKG = REPO / "experiments/bela_stage2_icer/BELA_STAGE2_ICER_handoff"
ACC = REPO / "logs/e014_aee/full/bela_icer_g1"
TIMING = PKG / "server_results" / "timing"
ARMS = (
    "S2_W2048_QATTN_ICER",
    "S3_W2048_QATTN_ICER_DRIFT",
    "S4_W2048_RDRIFT_ICER",
    "S5_W2048_SINGLETON_PROXY",
    "S7_W2048_RANDOM_SWAP",
)


def main():
    out = {"status": "PARTIAL", "arms": {}, "score_protocol": "elva_scores_20261021"}
    complete = True
    for arm in ARMS:
        row = {"tasks": {}, "missing": []}
        for t in ACC_TASKS:
            p = ACC / arm / t / "metrics.json"
            if not p.exists():
                row["missing"].append(t)
                complete = False
                continue
            m = json.loads(p.read_text())
            row["tasks"][t] = score_task_from_metrics(m, t)
        if all(t in row["tasks"] for t in R4_TASKS + EXTRA_TASKS):
            row.update(summarize_arm(row["tasks"]))
        elif all(t in row["tasks"] for t in R4_TASKS):
            from elva_scores import compute_r4
            row["R4"] = compute_r4(row["tasks"])
        lat = TIMING / arm / "latency_summary.json"
        flp = TIMING / arm / "flops_report.json"
        if lat.exists():
            L = json.loads(lat.read_text())
            row["ttft_wall_median_ms"] = L["ttft_wall_ms"]["median"]
            row["ttft_cuda_median_ms"] = L["ttft_cuda_ms"]["median"]
            row["fixed32_wall_median_ms"] = L["fixed32_wall_ms"]["median"]
        else:
            row["missing"].append("latency_summary")
            complete = False
        if flp.exists():
            row["flops"] = json.loads(flp.read_text())
        out["arms"][arm] = row
    out["status"] = "COMPLETE" if complete else "PARTIAL"
    dest = PKG / "server_results" / "official_scores.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(format_relative_table(out["arms"], ARMS))
    for arm, row in out["arms"].items():
        if row.get("missing"):
            print(f"MISSING {arm}: {row['missing']}")
    md = PKG / "server_results" / "relative_acc_table.md"
    md.write_text(
        "# ICER relative ACC (100×abs/Dense)\n\n"
        + format_relative_table(out["arms"], ARMS)
        + "\n"
    )


if __name__ == "__main__":
    main()
