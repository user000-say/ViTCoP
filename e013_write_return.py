#!/usr/bin/env python3
"""Assemble E013_RETURN.md from live artifacts (handoff section K)."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
ROOT = REPO / "logs/e013_p1"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"


def sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    smoke = ROOT / "SMOKE_STATUS"
    compare = read_json(ROOT / "smoke" / "COMPARE.json")
    full = read_json(ROOT / "FULL_ACCURACY_TABLE.json")
    gate = read_json(ROOT / "latency" / "CROSSOVER_GATE.json")
    panel = read_json(ROOT / "PANEL_SUMMARY.json")
    master = (ROOT / "MASTER_STATUS").read_text().strip() if (ROOT / "MASTER_STATUS").exists() else "RUNNING"

    llama = REPO / "LLaVA/llava/model/language_model/vitcop_modeling_llama.py"
    clip = REPO / "LLaVA/llava/model/multimodal_encoder/clip_encoder.py"
    support = REPO / "LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py"

    versions = subprocess.check_output(
        [PY, "-c", "import torch,transformers; print(torch.__version__, transformers.__version__, torch.version.cuda)"],
        text=True,
    ).strip()
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        text=True,
    ).strip()

    lat_dirs = sorted(p.name for p in (ROOT / "latency").glob("*") if p.is_dir())
    stage1_arms = ["BASE_S8", "REPLACE_ATTN16", "REPLACE_COVER16", "EXTRA_ATTN16", "EXTRA_COVER16"]
    stage1_done = [a for a in stage1_arms if (ROOT / "latency" / a / "latency.jsonl").exists()]

    # H1/H2 decision from FULL metrics only
    decision = "inconclusive / pending full metrics"
    if full and full.get("complete"):
        arms = full["arms"]
        def gqa(a):
            cell = arms[a]["gqa"] or {}
            return cell.get("exact_match")
        def f1(a):
            cell = arms[a]["pope"] or {}
            return cell.get("pope_f1_score")
        def mme(a):
            cell = arms[a]["mme"] or {}
            return cell.get("mme_total")
        def mmb(a):
            cell = arms[a]["mmbench_en_dev"] or {}
            return cell.get("gpt_eval_score")

        def beats(cand, controls, metric):
            cv = metric(cand)
            if cv is None:
                return False
            return all(metric(c) is not None and cv > metric(c) for c in controls)

        h1 = beats("REPLACE_COVER16", ["BASE_S8", "REPLACE_RANDOM16", "REPLACE_ATTN16"], gqa) and \
             beats("REPLACE_COVER16", ["BASE_S8", "REPLACE_RANDOM16", "REPLACE_ATTN16"], f1)
        h2 = beats("EXTRA_COVER16", ["EXTRA_RANDOM16", "EXTRA_ATTN16", "EXTRA_NULL16", "EXTRA_DUPLICATE16"], gqa) and \
             beats("EXTRA_COVER16", ["EXTRA_RANDOM16", "EXTRA_ATTN16", "EXTRA_NULL16", "EXTRA_DUPLICATE16"], f1)
        decision = (
            f"H1={'accept_candidate' if h1 else 'reject/inconclusive'} "
            f"(COVER vs BASE/RANDOM/ATTN on full GQA+POPE F1; also inspect MME/MMBench in table). "
            f"H2={'accept_candidate' if h2 else 'reject/inconclusive'} "
            f"(COVER vs RANDOM/ATTN/NULL/DUPLICATE). "
            "COVER is a test heuristic — no algorithmic novelty claim."
        )

    lines = []
    lines.append("# E013_RETURN")
    lines.append(f"generated_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## 1) Source")
    lines.append(f"- git: `{git}`")
    lines.append(f"- live llama SHA: `{sha(llama)}`")
    lines.append(f"- live clip SHA: `{sha(clip)}`")
    lines.append(f"- live e013_stage1_support SHA: `{sha(support)}`")
    lines.append(f"- torch/transformers/CUDA: `{versions}`")
    lines.append("- GPUs:")
    lines.append("```")
    lines.append(gpus)
    lines.append("```")
    lines.append("- runner: `e013_boot.py` → `e010_eval_job.py` / lmms_eval; latency `e013_latency.py` timed region=`raw.generate`; input_policy=`model_inference`")
    lines.append("- ACC policy: full four-dataset only (user). Latency gate: Fixed32 |rel Δ| ≤ 15% (user).")
    lines.append(f"- backup: `{(ROOT / 'ACTIVE_BACKUP_PATH.txt').read_text().strip() if (ROOT / 'ACTIVE_BACKUP_PATH.txt').exists() else 'n/a'}`")
    lines.append("")
    lines.append("## 2) Install & smoke")
    lines.append(f"- SMOKE_STATUS: `{(smoke.read_text().strip() if smoke.exists() else 'missing')}`")
    lines.append(f"- COMPARE: `{json.dumps(compare)}`")
    lines.append("- logs: `logs/e013_p1/smoke_driver.log`, `logs/e013_p1/smoke/*/smoke.json`")
    lines.append("")
    lines.append("## 3) Panel (diagnostic only — NOT ACC evidence)")
    lines.append(f"- PANEL_SUMMARY present: `{panel is not None}`")
    lines.append("- screen1/screen2 metrics under `logs/e013_p1/panel/`; N=512 per task; do not use for H1/H2.")
    lines.append("")
    lines.append("## 4) Full official metrics")
    lines.append(f"- FULL_ACCURACY_TABLE: `{ROOT / 'FULL_ACCURACY_TABLE.json'}`")
    lines.append(f"- complete: `{bool(full and full.get('complete'))}`")
    if full:
        lines.append(f"- missing: {full.get('missing')}")
        lines.append(f"- bad_n: {full.get('bad_n')}")
        lines.append("```json")
        lines.append(json.dumps(full.get("arms"), indent=2)[:120000])
        lines.append("```")
    lines.append("")
    lines.append("## 5) ABBA crossover")
    lines.append(f"- gate: `{json.dumps(gate, indent=2) if gate else 'pending'}`")
    lines.append(f"- latency dirs: {lat_dirs}")
    lines.append("- rule: user 15% relative Fixed32 vs LEGACY")
    lines.append("")
    lines.append("## 6) Stage1 latency")
    lines.append(f"- done arms: {stage1_done}")
    lines.append("- required after crossover PASS: BASE_S8 REPLACE_ATTN16 REPLACE_COVER16 EXTRA_ATTN16 EXTRA_COVER16")
    lines.append("- threshold vs BASE: user ≤15% Fixed32 relative overhead for acceptability notes")
    lines.append("")
    lines.append("## 7) Decision")
    lines.append(f"- MASTER_STATUS: `{master}`")
    lines.append(f"- {decision}")
    lines.append("")
    lines.append("## 8) Rollback / archive")
    lines.append("- rollback: `bash e013_rollback.sh $(cat logs/e013_p1/ACTIVE_BACKUP_PATH.txt)`")
    lines.append(f"- return path: `{ROOT / 'E013_RETURN.md'}`")
    lines.append("- next: only if H2 accepts on full metrics → E014 counterfactual utility predictor; else report soft negative.")

    text = "\n".join(lines) + "\n"
    out = ROOT / "E013_RETURN.md"
    out.write_text(text)
    (REPO / "E013_RETURN.md").write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
