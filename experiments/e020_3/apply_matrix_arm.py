"""Map E020.3 matrix configs onto E014/E018/E020 env for e014_boot."""
from __future__ import annotations

import json
import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
MATRIX = json.loads((PKG / "matrix.json").read_text())
BY_ID = {c["id"]: c for c in MATRIX["configs"]}

# Match live FIRST_SHIFT / NEXT conventions (0-based last completed layer).
PROFILES = {
    "SHIFT": "2:116,12:32",
    "NEXT": "3:109,12:32",
    "SHIFT_K108": "2:108,12:32",
    "SHIFT_K100": "2:100,12:32",
    "SHIFT_K132": "2:132,12:32",
    "FINAL11": "2:116,10:32",
    "FINAL15_WMATCH": "2:102,14:32",
}


def arm_name(cfg: dict) -> str:
    """Stable short arm id for the queue."""
    return f"E0203_{cfg['id']}"


def apply_config(cfg: dict) -> dict:
    """Set os.environ for one matrix config. Call after base apply_arm_env REF."""
    import e014_arms as m

    entry = cfg["entry"]
    base_arm = (
        "E020_COVER_FIRST_SHIFT_REF32"
        if entry == "COVER"
        else "E020_BASE_FIRST_SHIFT_REF32"
    )
    meta = dict(m.apply_arm_env(base_arm))
    events = PROFILES[cfg["profile_alias"]]
    name = arm_name(cfg)
    os.environ["E018_EVENTS"] = events
    os.environ["E014_ARM"] = name
    os.environ["E020_3_ENABLE"] = "True"
    os.environ["E020_3_OPERATOR"] = json.dumps(cfg["operator"])
    os.environ["E020_3_SEED"] = str(int(cfg.get("seed", 0)))
    os.environ["E020_3_SOURCE_ID"] = cfg["id"]
    os.environ["E020_3_CONFIG_ID"] = cfg["id"]
    os.environ["E020_3_BLOCK"] = cfg["block"]
    # First-event selector: REF vs historical RDRIFT control.
    if cfg["first_selector"] == "RDRIFT":
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
        )
    else:
        os.environ["E020_ENABLE"] = "False"
        os.environ["E020_KIND"] = "reference"
    # Final selector extras (E/G). ORIGINAL leaves E018 Q_ATTN path.
    final = cfg.get("final_selector", "ORIGINAL")
    os.environ["E020_3_FINAL_SELECTOR"] = final
    if final in ("POOLED_QK", "KEY_ENERGY", "RANDOM"):
        os.environ["E020_3_FINAL_K"] = events.split(",")[-1].split(":")[1]
    meta["arm"] = name
    meta["e0203_id"] = cfg["id"]
    meta["e0203_block"] = cfg["block"]
    return meta


def apply_arm(arm: str) -> dict:
    if not arm.startswith("E0203_"):
        raise ValueError(arm)
    cfg_id = arm[len("E0203_") :]
    if cfg_id not in BY_ID:
        raise KeyError(cfg_id)
    return apply_config(BY_ID[cfg_id])


def wave1_configs():
    """Wiring + discovery: A anchors → B same-K → D frontier (ORIGINAL final only)."""
    buckets = {"A": [], "B": [], "D": []}
    for c in MATRIX["configs"]:
        if c["block"] not in buckets:
            continue
        if c.get("final_selector", "ORIGINAL") != "ORIGINAL":
            continue
        buckets[c["block"]].append(c)

    def rank_a(c):
        # COVER×SHIFT×RDRIFT historical 96.79 first among anchors.
        return (
            0 if c["entry"] == "COVER" and c["profile_alias"] == "SHIFT" and c["first_selector"] == "RDRIFT" else
            1 if c["entry"] == "COVER" and c["profile_alias"] == "SHIFT" else
            2 if c["entry"] == "COVER" else
            3
        )

    def rank_b(c):
        # Prefer RDRIFT first-selector; both-event alpha=0.5 next.
        ev = tuple(c["operator"].get("events") or [])
        return (
            0 if c["first_selector"] == "RDRIFT" and ev == ("first", "final") else
            1 if c["first_selector"] == "RDRIFT" else
            2
        )

    buckets["A"].sort(key=rank_a)
    buckets["B"].sort(key=rank_b)
    return buckets["A"] + buckets["B"] + buckets["D"]


def wave1_arms():
    return [arm_name(c) for c in wave1_configs()]
