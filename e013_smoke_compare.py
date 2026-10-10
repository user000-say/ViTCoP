#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path("/home/xiangyue/ViTCoP/logs/e013_p1/smoke")


def load(arm: str) -> dict:
    path = ROOT / arm / "smoke.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())


def main() -> None:
    fails = []
    base = load("BASE_S8")
    b0 = load("B0_S8")
    for i, (a, b) in enumerate(zip(base["samples"], b0["samples"])):
        if a["selected_local"] != b["selected_local"]:
            fails.append(f"BASE/B0 selected diverge sample{i}")
        if a["token_ids"] != b["token_ids"] or a["text"] != b["text"]:
            fails.append(f"BASE/B0 output diverge sample{i}")
        if a["persistent_ids"] is not None and b["persistent_ids"] is not None:
            if a["persistent_ids"] != b["persistent_ids"]:
                fails.append(f"BASE/B0 patch ids diverge sample{i}")
    for arm in ("REPLACE_COVER16", "EXTRA_ATTN16", "EXTRA_COVER16", "EXTRA_NULL16", "EXTRA_DUPLICATE16"):
        rec = load(arm)
        if rec.get("status") != "PASS":
            fails.append(f"{arm} status {rec.get('status')}")
    report = {"status": "FAIL" if fails else "PASS", "fails": fails}
    (ROOT / "COMPARE.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
