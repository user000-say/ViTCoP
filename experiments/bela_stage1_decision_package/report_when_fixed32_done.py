#!/usr/bin/env python3
"""Poll Fixed32 pair; write REPORT md and print to log as soon as both summaries exist."""
from __future__ import annotations

import json
import time
from pathlib import Path

DEC = Path("/home/xiangyue/ViTCoP/experiments/bela_stage1_decision_package")
LAT = DEC / "latency"
OUT = DEC / "FIXED32_S1_VS_P3QA_REPORT.md"
LOG = DEC / "logs" / "fixed32_report_watch.log"


def _summ(arm: str) -> dict | None:
    p = LAT / arm / "latency_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _fmt(s: dict, key: str) -> str:
    x = s[key]
    return f"mean={x['mean']:.2f}  med={x['median']:.2f}  p90={x['p90']:.2f}"


def write_report(s1: dict, p3: dict) -> None:
    lines = [
        "# Fixed32 report — S1 RDRIFT vs P3 QA+Q_ATTN (V100 fair-256)",
        "",
        f"- Panel: `{s1.get('panel_path')}` n={s1.get('n')}",
        f"- Protocol: {s1.get('definition')}",
        f"- Scope: {s1.get('scope')}",
        "",
        "## Absolute (wall ms)",
        "",
        "| Arm | Stage2 | TTFT mean/med/p90 | Fixed32 mean/med/p90 | Peak MB med |",
        "|---|---|---|---|---|",
    ]
    for name, lab, d in [
        ("S1_QANCHOR_L2_RDRIFT", "first RDRIFT + later Q_ATTN", s1),
        ("P3_QA_QATTN", "all-cut Q_ATTN", p3),
    ]:
        lines.append(
            f"| {name} | {lab} | {_fmt(d,'ttft_wall_ms')} | {_fmt(d,'fixed32_wall_ms')} | "
            f"{d['peak_gpu_allocated_mb']['median']:.1f} |"
        )
    d_ttft = p3["ttft_wall_ms"]["mean"] - s1["ttft_wall_ms"]["mean"]
    d_f32 = p3["fixed32_wall_ms"]["mean"] - s1["fixed32_wall_ms"]["mean"]
    lines += [
        "",
        "## Delta (P3 − S1), mean wall ms",
        f"- TTFT: **{d_ttft:+.2f}** ms ({100*d_ttft/s1['ttft_wall_ms']['mean']:+.2f}%)",
        f"- Fixed32: **{d_f32:+.2f}** ms ({100*d_f32/s1['fixed32_wall_ms']['mean']:+.2f}%)",
        "",
        "Same Stage1 Dual+QA B8 / L2 W1024; only Stage2 first-cut selector differs.",
        "V100 engineering panel — not A800/STAR paper protocol.",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    msg = (
        f"FIXED32_REPORT_READY TTFT_delta={d_ttft:+.2f}ms "
        f"Fixed32_delta={d_f32:+.2f}ms path={OUT}"
    )
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")
        f.write(OUT.read_text())


def main() -> None:
    print("WATCH_FIXED32_REPORT", flush=True)
    while True:
        s1 = _summ("S1_QANCHOR_L2_RDRIFT")
        p3 = _summ("P3_QA_QATTN")
        if s1 and p3:
            write_report(s1, p3)
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
