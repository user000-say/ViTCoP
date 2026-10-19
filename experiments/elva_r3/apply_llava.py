"""Overlay one R3 config onto the existing E014/E018 environment.

Stage2 stays the original Q_ATTN prune. The two schedules are the R3
pre-block counts, expressed as the 0-based layer index E018 already prunes after.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def e018_events(config: dict) -> str:
    counts = config["method"]["schedule"]["counts"]
    lengths = config["method"]["schedule"]["lengths"]
    done = 0
    parts = []
    for length, keep in zip(lengths[:-1], counts[1:]):
        done += int(length)
        parts.append(f"{done - 1}:{int(keep)}")
    return ",".join(parts)


def overlay(meta: dict | None = None) -> dict:
    path = Path(os.environ["ELVA_R3_CONFIG"])
    config = json.loads(path.read_text())
    counts = [int(k) for k in config["method"]["schedule"]["counts"]]
    stage = config["method"]["stage1"]
    slots = 0
    if stage["cover_family"] == "legacy_native":
        slots = 32
    elif stage["cover_family"] == "protected_coverage":
        slots = int(stage["coverage_slots"])
    events = e018_events(config)
    os.environ.update(
        ELVA_R3_ENABLE="True",
        ELVA_R3_ENTRY_K=str(counts[0]),
        ELVA_R3_FINAL_K=str(counts[-1]),
        ELVA_R3_SLOTS=str(slots),
        ELVA_R3_ID=config["id"],
        E018_ENABLE="True",
        E018_MODE="STATIC",
        E018_EVENTS=events,
        E018_SELECTOR="Q_ATTN",
        E018_CACHE_POLICY="layer_local",
        E020_ENABLE="False",
        E020_3_ENABLE="False",
        E006_TARGET_K=str(counts[-1]),
        E009_EXPECT_VISUAL=str(counts[0]),
        E007_EXPECT_STAGE1=str(counts[0]),
    )
    meta = dict(meta or {})
    meta["visual"] = counts[0]
    meta["target_k"] = counts[-1]
    meta["elva_id"] = config["id"]
    meta["e018_events"] = events
    return meta
