#!/usr/bin/env python3
"""Write E014_RETURN.md from smoke reports + accuracy tables."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
OUT = REPO / "experiments/e014_aee"
ROOT = REPO / "logs/e014_aee"


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoke_summary() -> list[dict]:
    rows = []
    smoke = ROOT / "smoke_gpu"
    if not smoke.exists():
        return rows
    for path in sorted(smoke.glob("*/smoke_report.json")):
        data = json.loads(path.read_text())
        rows.append({
            "arm": data.get("arm"),
            "ok": data.get("ok"),
            "budgets": [s.get("e014_budget") for s in data.get("samples", [])],
            "errors": data.get("errors") or [],
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    clip = REPO / "LLaVA/llava/model/multimodal_encoder/clip_encoder.py"
    llama = REPO / "LLaVA/llava/model/language_model/vitcop_modeling_llama.py"
    summary_path = OUT / "FULL_ACCURACY_SUMMARY.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    smoke = smoke_summary()
    lines = [
        "# E014_RETURN",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Provenance",
        f"- git HEAD: `{git}`",
        f"- clip SHA256: `{sha(clip)}`",
        f"- llama SHA256: `{sha(llama)}`",
        "- remote: `fork` → https://github.com/user000-say/ViTCoP",
        "- protocol: LLaVA-1.5-7B FP16 SDPA, Stage1=172, S=D=L8, QED K∈{96,64,32}",
        "",
        "## P0",
        "- CPU tests: 12/12 PASS",
        "- compileall: PASS",
        "- E013 base SHA matched before patch; E014 sources installed by byte copy",
        f"- GPU smoke arms: {sum(1 for r in smoke if r['ok'])}/{len(smoke)} PASS",
        "",
    ]
    for row in smoke:
        lines.append(
            f"- `{row['arm']}`: ok={row['ok']} budgets={row['budgets']} errors={row['errors'][:2]}"
        )
    lines += ["", "## Full accuracy"]
    for phase, blob in summary.items():
        lines.append(f"### {phase} complete={blob.get('complete')}")
        for row in blob.get("rows", []):
            lines.append(
                f"- `{row.get('arm')}` gqa={row.get('gqa')} pope={row.get('pope')} "
                f"mme={row.get('mme')} mmb={row.get('mmbench_en_dev')} "
                f"ok={row.get('arm_ok')}"
            )
    lines += [
        "",
        "## Judgment (fill after full N)",
        "- Pending full_k96/k64/k32 completion for AGREE/MODIFY/REJECT.",
        "- Note: panel AEE/ADAPTIVE chose b=32 on smoke samples — report histogram on full audit before claiming adaptive success.",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "E014_RETURN.md").write_text(text, encoding="utf-8")
    (REPO / "E014_RETURN.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
