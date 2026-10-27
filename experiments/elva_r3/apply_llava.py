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
    first = config["method"]["stage2"].get("first_selector", "Q_ATTN")
    later = config["method"]["stage2"].get("later_selector", "Q_ATTN")
    # E018 uses one score family for every progressive event.
    # QED is a native E018 selector; RDRIFT refines the first event on top of Q_ATTN.
    if first == "QED" or later == "QED":
        e018_selector = "QED"
    elif first == "ENTRY_ORDER" or later == "ENTRY_ORDER":
        # Sprint B2: both cuts must use entry COVER order (no Q_ATTN/RDRIFT mix).
        if first != "ENTRY_ORDER" or later != "ENTRY_ORDER":
            raise ValueError("ENTRY_ORDER requires first_selector=later_selector=ENTRY_ORDER")
        e018_selector = "ENTRY_ORDER"
    elif first in {"Q_ATTN", "RANDOM"}:
        e018_selector = first
    else:
        e018_selector = "Q_ATTN"
    if e018_selector not in {"Q_ATTN", "QED", "RANDOM", "ENTRY_ORDER"}:
        raise ValueError(f"unsupported E018_SELECTOR from stage2: {first}/{later}")
    # Flat schedules (single segment) yield empty E018_EVENTS — entry-only, no Stage2 cuts.
    last_layer = int(events.split(",")[-1].split(":")[0]) if events else -1
    os.environ.update(
        ELVA_R3_ENABLE="True",
        ELVA_R3_ENTRY_K=str(counts[0]),
        ELVA_R3_FINAL_K=str(counts[-1]),
        ELVA_R3_SLOTS=str(slots),
        ELVA_R3_ID=config["id"],
        E018_ENABLE="True" if events else "False",
        E018_MODE="STATIC",
        E018_EVENTS=events,
        E018_SELECTOR=e018_selector,
        E018_CACHE_POLICY="layer_local",
        E020_3_ENABLE="False",
        E006_TARGET_K=str(counts[-1]),
        E009_EXPECT_VISUAL=str(counts[0]),
        E007_EXPECT_STAGE1=str(counts[0]),
    )
    if first == "RDRIFT" and events:
        # Original E020 rank_drift on FIRST only; FINAL stays Q_ATTN. No merge.
        if e018_selector != "Q_ATTN":
            raise ValueError("RDRIFT first_selector requires E018_SELECTOR=Q_ATTN baseline")
        os.environ.update(
            E020_ENABLE="True",
            E020_KIND="rank_drift",
            E020_CURRENT_WEIGHT="0.6",
            E020_MODEL="current",
            E020_MODEL_PATH="",
            E020_GROUP_SIZE="1",
            E020_ROUNDS="1",
            E020_SHORTLIST="16",
            E020_CANDIDATE_LIMIT="16",
            E020_THRESHOLD="0",
            E020_ACCEPT_ENABLED="False",
            E020_NEXT_AFTER=str(last_layer + 1),
            E020_DEBUG=os.environ.get("E020_DEBUG", "True"),
        )
    else:
        os.environ["E020_ENABLE"] = "False"
        os.environ["E020_KIND"] = "reference"
    meta = dict(meta or {})
    meta["visual"] = counts[0]
    meta["target_k"] = counts[-1]
    meta["elva_id"] = config["id"]
    meta["e018_events"] = events
    meta["first_selector"] = first
    meta["later_selector"] = later
    meta["e018_selector"] = e018_selector
    return meta
