"""Summarize Gate1 9-task ACC + TTFT/FLOPs with R4/R8/R9 retention."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
sys.path.insert(0, str(REPO))
from elva_acc_tasks import ACC_TASKS, EXTRA_TASKS, R4_TASKS
from elva_scores import format_relative_table, score_task_from_metrics, summarize_arm

PKG = REPO / "experiments/bela_20261020/BELA_CURSOR_20261020"
ACC = REPO / "logs/e014_aee/full/bela_gate1_8task"
TIMING = PKG / "server_results" / "timing"
ARMS = ("W2048_V5_COVER_QATTN", "W2048_V5_COVER_RDRIFT", "W1024_V4_PCOV_QATTN")


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
            a = m.get("audit") or {}
            row["entry_k"] = a.get("stage1_tokens")
            row["final_k"] = a.get("final_visual_tokens")
            row["e018_events"] = a.get("e018_events")
            row["W_trace"] = (
                sum(a["e018_pre_block_counts"]) if a.get("e018_pre_block_counts") else None
            )
        if all(t in row["tasks"] for t in R4_TASKS + EXTRA_TASKS):
            row.update(summarize_arm(row["tasks"]))
            # keep flat R4 for backward compat
            row["R4"] = row["R4"]
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
            row["peak_gpu_mb_median"] = L["peak_gpu_allocated_mb"]["median"]
        else:
            complete = False
            row["missing"].append("latency_summary")
        if flp.exists():
            row["flops"] = json.loads(flp.read_text())
        out["arms"][arm] = row
    out["status"] = "COMPLETE" if complete else "PARTIAL"
    dest = PKG / "server_results" / "official_scores.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(format_relative_table(out["arms"], ARMS))
    md = PKG / "server_results" / "relative_acc_table.md"
    md.write_text(
        "# Gate1 relative ACC (100×abs/Dense)\n\n"
        + format_relative_table(out["arms"], ARMS)
        + "\n"
    )


if __name__ == "__main__":
    main()
