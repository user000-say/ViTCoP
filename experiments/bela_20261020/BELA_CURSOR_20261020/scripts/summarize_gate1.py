"""Summarize Gate1 8-task ACC + TTFT/FLOPs into server_results/official_scores.json."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/bela_20261020/BELA_CURSOR_20261020"
ACC = REPO / "logs/e014_aee/full/bela_gate1_8task"
TIMING = PKG / "server_results" / "timing"
R4_TASKS = ("gqa", "pope", "mme", "mmbench_en_dev")
EXTRA = ("scienceqa_img", "mmmu_val", "ai2d", "textvqa_val_ocr")
ARMS = ("W2048_V5_COVER_QATTN", "W2048_V5_COVER_RDRIFT", "W1024_V4_PCOV_QATTN")


def score_task(m: dict, task: str):
    if task == "gqa":
        v = float(m["exact_match"]); return v * (100 if v <= 1.5 else 1)
    if task == "pope":
        v = float(m["pope_f1_score"]); return v * (100 if v <= 1.5 else 1)
    if task == "mme":
        return float(m["mme_perception_score"]) + float(m["mme_cognition_score"])
    if task == "mmbench_en_dev":
        v = m.get("gpt_eval_score"); v = float(v if v is not None else m["exact_match"])
        return v * (100 if v <= 1.5 else 1)
    if task == "mmmu_val":
        return float(m["mmmu_acc"]) * (100 if float(m["mmmu_acc"]) <= 1.5 else 1)
    v = float(m["exact_match"]); return v * (100 if v <= 1.5 else 1)


def main():
    out = {"status": "PARTIAL", "arms": {}}
    complete = True
    for arm in ARMS:
        row = {"tasks": {}, "missing": []}
        for t in R4_TASKS + EXTRA:
            p = ACC / arm / t / "metrics.json"
            if not p.exists():
                row["missing"].append(t); complete = False; continue
            m = json.loads(p.read_text())
            row["tasks"][t] = score_task(m, t)
            a = m.get("audit") or {}
            row["entry_k"] = a.get("stage1_tokens")
            row["final_k"] = a.get("final_visual_tokens")
            row["e018_events"] = a.get("e018_events")
            row["W_trace"] = sum(a["e018_pre_block_counts"]) if a.get("e018_pre_block_counts") else None
        if all(t in row["tasks"] for t in R4_TASKS):
            s = row["tasks"]
            row["R4"] = 25 * (
                s["gqa"] / 61.9 + s["pope"] / 85.8 + s["mme"] / 1852 + s["mmbench_en_dev"] / 64.1
            )
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
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
