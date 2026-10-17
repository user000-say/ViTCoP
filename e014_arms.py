"""E014 arm definitions. Env wiring only — no selector math."""
from __future__ import annotations

from pathlib import Path

# name -> (variant, b, s, d, target_k)
# Stage2 freeze: S=D primary L8; K32 additionally sweeps L8/L10/L12 full ACC.
ARMS: dict[str, tuple[str, int, int, int, int]] = {
    # K96 primary matrix (L8)
    "BASE_K96": ("BASE", 0, 8, 8, 96),
    "FIXED_COVER_B0_K96": ("FIXED_COVER", 0, 8, 8, 96),
    "FIXED_COVER_B8_K96": ("FIXED_COVER", 8, 8, 8, 96),
    "FIXED_COVER_B16_K96": ("FIXED_COVER", 16, 8, 8, 96),
    "FIXED_COVER_B24_K96": ("FIXED_COVER", 24, 8, 8, 96),
    "FIXED_COVER_B32_K96": ("FIXED_COVER", 32, 8, 8, 96),
    "FIXED_RISK_B16_K96": ("FIXED_RISK", 16, 8, 8, 96),
    "ADAPTIVE_B_ONLY_K96": ("ADAPTIVE_B_ONLY", 16, 8, 8, 96),
    "AEE_K96": ("AEE", 16, 8, 8, 96),
    # K64 stress (L8)
    "BASE_K64": ("BASE", 0, 8, 8, 64),
    "FIXED_COVER_B16_K64": ("FIXED_COVER", 16, 8, 8, 64),
    "FIXED_COVER_B32_K64": ("FIXED_COVER", 32, 8, 8, 64),
    "FIXED_RISK_B16_K64": ("FIXED_RISK", 16, 8, 8, 64),
    "ADAPTIVE_B_ONLY_K64": ("ADAPTIVE_B_ONLY", 16, 8, 8, 64),
    "AEE_K64": ("AEE", 16, 8, 8, 64),
    # K32 L8 (original)
    "BASE_K32": ("BASE", 0, 8, 8, 32),
    "FIXED_COVER_B16_K32": ("FIXED_COVER", 16, 8, 8, 32),
    "FIXED_RISK_B16_K32": ("FIXED_RISK", 16, 8, 8, 32),
    "ADAPTIVE_B_ONLY_K32": ("ADAPTIVE_B_ONLY", 16, 8, 8, 32),
    "AEE_K32": ("AEE", 16, 8, 8, 32),
    # K32 layer sweep L10 / L12 (and aliases L8 for explicit naming)
    "BASE_K32_L8": ("BASE", 0, 8, 8, 32),
    "FIXED_COVER_B16_K32_L8": ("FIXED_COVER", 16, 8, 8, 32),
    "FIXED_RISK_B16_K32_L8": ("FIXED_RISK", 16, 8, 8, 32),
    "ADAPTIVE_B_ONLY_K32_L8": ("ADAPTIVE_B_ONLY", 16, 8, 8, 32),
    "AEE_K32_L8": ("AEE", 16, 8, 8, 32),
    "BASE_K32_L10": ("BASE", 0, 10, 10, 32),
    "FIXED_COVER_B16_K32_L10": ("FIXED_COVER", 16, 10, 10, 32),
    "FIXED_RISK_B16_K32_L10": ("FIXED_RISK", 16, 10, 10, 32),
    "ADAPTIVE_B_ONLY_K32_L10": ("ADAPTIVE_B_ONLY", 16, 10, 10, 32),
    "AEE_K32_L10": ("AEE", 16, 10, 10, 32),
    "BASE_K32_L12": ("BASE", 0, 12, 12, 32),
    "FIXED_COVER_B16_K32_L12": ("FIXED_COVER", 16, 12, 12, 32),
    "FIXED_RISK_B16_K32_L12": ("FIXED_RISK", 16, 12, 12, 32),
    "ADAPTIVE_B_ONLY_K32_L12": ("ADAPTIVE_B_ONLY", 16, 12, 12, 32),
    "AEE_K32_L12": ("AEE", 16, 12, 12, 32),
    # E015 regional core full ACC (K64/L8): BASE → GLOBAL_RETAINED → REGION_MEAN → SALIENT
    "GLOBAL_RETAINED_MEAN_B16_K64": ("GLOBAL_RETAINED_MEAN", 16, 8, 8, 64),
    "REGION_MEAN_B16_K64": ("REGION_MEAN", 16, 8, 8, 64),
    "REGION_DUAL_SALIENT_B16_K64": ("REGION_DUAL_SALIENT", 16, 8, 8, 64),
    # E016 progressive cascade (QED rank frozen at L8): 96→64→32→16
    "CASCADE_L8_96_L10_64_L16_32_L20_16": ("BASE", 0, 8, 8, 16),
    # E015 dynamic COVER / regional AEE (K96 / S=D=L8, MEAN)
    "E015_COVER_FIXED_B16_K96": ("E015_COVER_FIXED", 16, 8, 8, 96),
    "E015_COVER_FIXED_B32_K96": ("E015_COVER_FIXED", 32, 8, 8, 96),
    "E015_AEE_FIXED_B16_K96": ("E015_AEE_FIXED", 16, 8, 8, 96),
    "E015_AEE_FIXED_B32_K96": ("E015_AEE_FIXED", 32, 8, 8, 96),
    "E015_GLOBAL_COVER_FIXED_B16_K96": ("E015_GLOBAL_COVER_FIXED", 16, 8, 8, 96),
    "E015_GLOBAL_AEE_FIXED_B16_K96": ("E015_GLOBAL_AEE_FIXED", 16, 8, 8, 96),
    "E015_COVER_DYNAMIC_K96": ("E015_COVER_DYNAMIC", 16, 8, 8, 96),
    "E015_AEE_DYNAMIC_K96": ("E015_AEE_DYNAMIC", 16, 8, 8, 96),
    "E015_GLOBAL_COVER_DYNAMIC_K96": ("E015_GLOBAL_COVER_DYNAMIC", 16, 8, 8, 96),
    "E015_GLOBAL_AEE_DYNAMIC_K96": ("E015_GLOBAL_AEE_DYNAMIC", 16, 8, 8, 96),
    # E016 Stage1 fixed-b=32 closure (K96 / S=D=L8)
    "E016_MARGINAL_B32_K96": ("E016_MARGINAL", 32, 8, 8, 96),
    "E016_PROTECTED_B32_K96": ("E016_PROTECTED", 32, 8, 8, 96),
    "E016_SPATIAL_B32_K96": ("E016_SPATIAL", 32, 8, 8, 96),
    "E016_HIER_B32_K96": ("E016_HIER", 32, 8, 8, 96),
    "E016_CENTERED_B32_K96": ("E016_CENTERED", 32, 8, 8, 96),
    # E016_HIER follow-up: K64 × QED/delete layer {8,10,12}
    "E016_HIER_B32_K64_L8": ("E016_HIER", 32, 8, 8, 64),
    "E016_HIER_B32_K64_L10": ("E016_HIER", 32, 10, 10, 64),
    "E016_HIER_B32_K64_L12": ("E016_HIER", 32, 12, 12, 64),
}

# Layer sweep requested 2026-10-14:
#   K96 → L{8,10};  K64/K32 → L{8,10,12,14,16}
# Methods: HIER b32 + COVER32 + AEE (matched Stage1 comparators).
def _add_layer_sweep_arms() -> None:
    for k, layers in ((96, (8, 10)), (64, (8, 10, 12, 14, 16)), (32, (8, 10, 12, 14, 16))):
        for layer in layers:
            ARMS[f"E016_HIER_B32_K{k}_L{layer}"] = ("E016_HIER", 32, layer, layer, k)
            ARMS[f"FIXED_COVER_B32_K{k}_L{layer}"] = ("FIXED_COVER", 32, layer, layer, k)
            ARMS[f"AEE_K{k}_L{layer}"] = ("AEE", 16, layer, layer, k)


_add_layer_sweep_arms()

SMOKE_K96 = [
    "BASE_K96",
    "FIXED_COVER_B16_K96",
    "FIXED_RISK_B16_K96",
    "ADAPTIVE_B_ONLY_K96",
    "AEE_K96",
]
SMOKE_K64 = ["BASE_K64", "AEE_K64", "FIXED_COVER_B16_K64"]
SMOKE_K32 = ["BASE_K32", "AEE_K32", "FIXED_COVER_B16_K32"]
SMOKE_K32_L10 = ["BASE_K32_L10", "AEE_K32_L10", "FIXED_COVER_B16_K32_L10"]
SMOKE_K32_L12 = ["BASE_K32_L12", "AEE_K32_L12", "FIXED_COVER_B16_K32_L12"]

FULL_K96 = [
    "BASE_K96",
    "FIXED_COVER_B0_K96",
    "FIXED_COVER_B8_K96",
    "FIXED_COVER_B16_K96",
    "FIXED_COVER_B24_K96",
    "FIXED_COVER_B32_K96",
    "FIXED_RISK_B16_K96",
    "ADAPTIVE_B_ONLY_K96",
    "AEE_K96",
]
FULL_K64 = [
    "BASE_K64",
    "FIXED_COVER_B16_K64",
    "FIXED_COVER_B32_K64",
    "FIXED_RISK_B16_K64",
    "ADAPTIVE_B_ONLY_K64",
    "AEE_K64",
]
# Original L8 K32 names (already launched); keep for continuity.
FULL_K32 = [
    "BASE_K32",
    "FIXED_COVER_B16_K32",
    "FIXED_RISK_B16_K32",
    "ADAPTIVE_B_ONLY_K32",
    "AEE_K32",
]
# Extra full-ACC layer sweep requested for QED32: L10 + L12 (L8 covered by FULL_K32).
FULL_K32_L10 = [
    "BASE_K32_L10",
    "FIXED_COVER_B16_K32_L10",
    "FIXED_RISK_B16_K32_L10",
    "ADAPTIVE_B_ONLY_K32_L10",
    "AEE_K32_L10",
]
FULL_K32_L12 = [
    "BASE_K32_L12",
    "FIXED_COVER_B16_K32_L12",
    "FIXED_RISK_B16_K32_L12",
    "ADAPTIVE_B_ONLY_K32_L12",
    "AEE_K32_L12",
]
# E015 handoff core-16: 4 methods × 4 tasks (BASE already in FULL_K64; listed explicitly)
FULL_E015_K64 = [
    "BASE_K64",
    "GLOBAL_RETAINED_MEAN_B16_K64",
    "REGION_MEAN_B16_K64",
    "REGION_DUAL_SALIENT_B16_K64",
]
FULL_E016_CASCADE = [
    "CASCADE_L8_96_L10_64_L16_32_L20_16",
]
# Handoff H1/H2 core: FIXED + DYNAMIC regional vs GLOBAL controls (K96).
FULL_E015_DYN_COVER_K96 = [
    "E015_COVER_FIXED_B16_K96",
    "E015_COVER_FIXED_B32_K96",
    "E015_AEE_FIXED_B16_K96",
    "E015_AEE_FIXED_B32_K96",
    "E015_GLOBAL_COVER_FIXED_B16_K96",
    "E015_GLOBAL_AEE_FIXED_B16_K96",
    "E015_COVER_DYNAMIC_K96",
    "E015_AEE_DYNAMIC_K96",
    "E015_GLOBAL_COVER_DYNAMIC_K96",
    "E015_GLOBAL_AEE_DYNAMIC_K96",
]
FULL_E016_STAGE1_K96 = [
    "E016_MARGINAL_B32_K96",
    "E016_PROTECTED_B32_K96",
    "E016_SPATIAL_B32_K96",
    "E016_HIER_B32_K96",
    "E016_CENTERED_B32_K96",
]
FULL_E016_HIER_K64_LAYER = [
    "E016_HIER_B32_K64_L8",
    "E016_HIER_B32_K64_L10",
    "E016_HIER_B32_K64_L12",
]
# K64 matched Stage1 set for extra-benchmark sweep (COVER/AEE + HIER layers).
FULL_E016_K64_EXTRA = [
    "BASE_K64",
    "FIXED_COVER_B16_K64",
    "FIXED_COVER_B32_K64",
    "AEE_K64",
    "E016_HIER_B32_K64_L8",
    "E016_HIER_B32_K64_L10",
    "E016_HIER_B32_K64_L12",
]
FULL_E016_LAYER_SWEEP = [
    # K96: L8/L10 only
    *[f"{m}_K96_L{L}" for L in (8, 10)
      for m in ("E016_HIER_B32", "FIXED_COVER_B32", "AEE")],
    # K64 / K32: L8..L16
    *[f"{m}_K{k}_L{L}" for k in (64, 32) for L in (8, 10, 12, 14, 16)
      for m in ("E016_HIER_B32", "FIXED_COVER_B32", "AEE")],
]

# E018 Stage2 continuation: fixed same-W K32 schedules (W=2284).
# Keep S=D=8 so legacy E014 guards stay green; real prune schedule is E018_EVENTS only.
ARMS.update({
    "E018_COVER_ONE32": ("FIXED_COVER", 32, 8, 8, 32),  # E018_EVENTS=8:32
    "E018_COVER_TWO32": ("FIXED_COVER", 32, 8, 8, 32),  # E018_EVENTS=4:102,12:32
    "E018_BASE_ONE32": ("BASE", 0, 8, 8, 32),
    "E018_BASE_TWO32": ("BASE", 0, 8, 8, 32),
})
FULL_E018_K32_STATIC = [
    "E018_COVER_ONE32",
    "E018_COVER_TWO32",
    "E018_BASE_ONE32",
    "E018_BASE_TWO32",
]

# E018 extended: early/late final-delete on TWO backbone (Δd=4).
# EARLY ≡ TWO (4:102,12:32, W=2284); LATE keeps 102 until post-16 →32 (W=2564).
ARMS.update({
    "E018_COVER_EARLY32": ("FIXED_COVER", 32, 8, 8, 32),
    "E018_COVER_LATE32": ("FIXED_COVER", 32, 8, 8, 32),
    "E018_BASE_EARLY32": ("BASE", 0, 8, 8, 32),
    "E018_BASE_LATE32": ("BASE", 0, 8, 8, 32),
})
FULL_E018_EARLY_LATE = [
    "E018_COVER_EARLY32",
    "E018_COVER_LATE32",
    "E018_BASE_EARLY32",
    "E018_BASE_LATE32",
]
FULL_E018_LATE_ONLY = [
    "E018_COVER_LATE32",
    "E018_BASE_LATE32",
]

# E019 progressive-evidence: up to 48 configs = A schedules + B selectors + C joint.
# Arm name: E019_{COVER|BASE}_{SCHEDULE}_{SELECTOR}32
_E019_EVENTS = {
    "ONE": "8:32",
    "EARLY": "4:102,12:32",
    "LATE": "4:102,16:32",
    "LATE_WMATCH": "4:78,16:32",  # near-W; integer mid cannot hit 2284 exactly
    "FIRST_SHIFT_WMATCH": "2:116,12:32",
    "THREE_WMATCH": "4:103,8:101,12:32",
}
_E019_SELECTORS = {
    "REF": {"mode": "reference", "alpha": 0.0, "lambda": 0.0, "swaps": 0},
    "CURREX": {"mode": "exchange", "alpha": 0.0, "lambda": 0.15, "swaps": 4},
    "TEMPA25": {"mode": "exchange", "alpha": 0.25, "lambda": 0.15, "swaps": 4},
    "TEMPA50": {"mode": "exchange", "alpha": 0.5, "lambda": 0.15, "swaps": 4},
    "TEMPA75": {"mode": "exchange", "alpha": 0.75, "lambda": 0.15, "swaps": 4},
    "PREV": {"mode": "exchange", "alpha": 1.0, "lambda": 0.15, "swaps": 4},
    "TEMPL05": {"mode": "exchange", "alpha": 0.5, "lambda": 0.05, "swaps": 4},
    "TEMPL30": {"mode": "exchange", "alpha": 0.5, "lambda": 0.30, "swaps": 4},
    "TEMPB2": {"mode": "exchange", "alpha": 0.5, "lambda": 0.15, "swaps": 2},
    "SCEMA": {"mode": "mean", "alpha": 0.5, "lambda": 0.0, "swaps": 0},
    "CMEAN": {"mode": "mean", "alpha": 0.0, "lambda": 0.0, "swaps": 0},
    "STOPK": {"mode": "score", "alpha": 0.0, "lambda": 0.0, "swaps": 0},
    "RSWAP": {"mode": "random", "alpha": 0.0, "lambda": 0.0, "swaps": 4},
}


def _e019_register():
    arms = {}
    # A: 2×6 schedule references
    for stage, variant, b in (("COVER", "FIXED_COVER", 32), ("BASE", "BASE", 0)):
        for sched in _E019_EVENTS:
            name = f"E019_{stage}_{sched}_REF32"
            arms[name] = (variant, b, 8, 8, 32)
    # B: 2×12 selectors on EARLY (exclude REF already in A)
    for stage, variant, b in (("COVER", "FIXED_COVER", 32), ("BASE", "BASE", 0)):
        for sel in _E019_SELECTORS:
            if sel == "REF":
                continue
            name = f"E019_{stage}_EARLY_{sel}32"
            arms[name] = (variant, b, 8, 8, 32)
    # C: 2×3 near-work × {CURREX,TEMPA50}
    for stage, variant, b in (("COVER", "FIXED_COVER", 32), ("BASE", "BASE", 0)):
        for sched in ("LATE_WMATCH", "FIRST_SHIFT_WMATCH", "THREE_WMATCH"):
            for sel in ("CURREX", "TEMPA50"):
                name = f"E019_{stage}_{sched}_{sel}32"
                arms[name] = (variant, b, 8, 8, 32)
    return arms


ARMS.update(_e019_register())
# Back-compat short names used by the first schedule-ref campaign.
for _stage, _var, _b in (("COVER", "FIXED_COVER", 32), ("BASE", "BASE", 0)):
    for _sched in ("LATE_WMATCH", "FIRST_SHIFT_WMATCH", "THREE_WMATCH"):
        ARMS[f"E019_{_stage}_{_sched}32"] = ARMS[f"E019_{_stage}_{_sched}_REF32"]

FULL_E019_SCHEDULE_REF = [
    "E019_COVER_LATE_WMATCH32",
    "E019_COVER_FIRST_SHIFT_WMATCH32",
    "E019_COVER_THREE_WMATCH32",
    "E019_BASE_LATE_WMATCH32",
    "E019_BASE_FIRST_SHIFT_WMATCH32",
    "E019_BASE_THREE_WMATCH32",
]
FULL_E019_SELECTOR_B = sorted(
    a for a in ARMS if a.startswith("E019_") and "_EARLY_" in a and not a.endswith("_REF32")
    and a.split("_")[2] == "EARLY"
)
FULL_E019_JOINT_C = sorted(
    a for a in ARMS
    if a.startswith("E019_") and any(s in a for s in ("LATE_WMATCH_", "FIRST_SHIFT_WMATCH_", "THREE_WMATCH_"))
    and any(a.endswith(f"_{s}32") for s in ("CURREX", "TEMPA50"))
)
FULL_E019_REMAINING = FULL_E019_SELECTOR_B + FULL_E019_JOINT_C
FULL_E019_ALL48 = sorted(
    a for a in ARMS
    if a.startswith("E019_") and a.endswith("32") and "_REF32" in a or (
        a.startswith("E019_") and a.endswith("32") and any(
            a.endswith(f"_{s}32") for s in _E019_SELECTORS if s != "REF"
        )
    )
)
# Cleaner ALL48: every registered canonical name ending _REF32 or selector tags.
FULL_E019_ALL48 = sorted({
    a for a in ARMS
    if a.startswith("E019_") and (
        a.endswith("_REF32")
        or any(a.endswith(f"_{s}32") for s in _E019_SELECTORS if s != "REF")
    )
})

# E020: FIRST_SHIFT neighbourhood (A) + no-weight controls (D) first; learned B later.
_E020_EVENTS = {
    "ONE": "8:32",
    "EARLY": "4:102,12:32",
    "FIRST_SHIFT": "2:116,12:32",
    "THREE_WMATCH": "4:103,8:101,12:32",
    "LATE": "4:102,16:32",
    "LATE_WMATCH": "4:78,16:32",
    "FIRST_PREV_WMATCH": "1:121,12:32",
    "FIRST_NEXT_WMATCH": "3:109,12:32",
    "FIRST_KMINUS8": "2:108,12:32",
    "FIRST_KPLUS8": "2:124,12:32",
}
_E020_KINDS = {
    "REF": {"kind": "reference"},
    "CSCORE": {"kind": "current_score"},
    "RDRIFT": {"kind": "rank_drift", "current_weight": 0.6},
    "E019CUR": {"kind": "coverage_exchange", "alpha": 0.0, "lambda": 0.15, "rounds": 4},
    "E019TMP": {"kind": "coverage_exchange", "alpha": 0.5, "lambda": 0.15, "rounds": 4},
    "RANDR2": {"kind": "random", "group_size": 1, "rounds": 2, "shortlist": 16, "seed": 20},
    "RANDR4": {"kind": "random", "group_size": 1, "rounds": 4, "shortlist": 16, "seed": 20},
    "RANDG4R1": {"kind": "random", "group_size": 4, "rounds": 1, "shortlist": 16, "seed": 20},
    "RANDG4R2": {"kind": "random", "group_size": 4, "rounds": 2, "shortlist": 16, "seed": 20},
    # R2 one-step catalogue only (DEFAULT_POLICY rounds=1, group_size=1).
    # Do not deploy G1R4/G4R2 against g1r1-calibrated checkpoints.
    "CFFULL_G1R1": {
        "kind": "continuation", "model": "full",
        "group_size": 1, "rounds": 1, "shortlist": 16, "candidate_limit": 16,
    },
    "CFCUR_G1R1": {
        "kind": "continuation", "model": "current",
        "group_size": 1, "rounds": 1, "shortlist": 16, "candidate_limit": 16,
    },
    "CFNOCTX_G1R1": {
        "kind": "continuation", "model": "no_context",
        "group_size": 1, "rounds": 1, "shortlist": 16, "candidate_limit": 16,
    },
    "CFSHUF_G1R1": {
        "kind": "continuation", "model": "shuffled",
        "group_size": 1, "rounds": 1, "shortlist": 16, "candidate_limit": 16,
    },
}


def _e020_register():
    arms = {}
    # A neighbourhood references on BASE/COVER
    for stage, variant, b in (("COVER", "FIXED_COVER", 32), ("BASE", "BASE", 0)):
        for sched in _E020_EVENTS:
            arms[f"E020_{stage}_{sched}_REF32"] = (variant, b, 8, 8, 32)
        # D controls + B continuation heads on FIRST_SHIFT
        for kind, spec in _E020_KINDS.items():
            if kind == "REF":
                continue
            if spec.get("kind") == "continuation":
                continue  # added below
            arms[f"E020_{stage}_FIRST_SHIFT_{kind}32"] = (variant, b, 8, 8, 32)
        for kind, spec in _E020_KINDS.items():
            if spec.get("kind") != "continuation":
                continue
            arms[f"E020_{stage}_FIRST_SHIFT_{kind}32"] = (variant, b, 8, 8, 32)
        # C transfer: full head on EARLY/THREE/K±8 (same one-step policy)
        for sched in ("EARLY", "THREE_WMATCH", "FIRST_KMINUS8", "FIRST_KPLUS8"):
            arms[f"E020_{stage}_{sched}_CFFULL_G1R132"] = (variant, b, 8, 8, 32)
    return arms



ARMS.update(_e020_register())
FULL_E020_A_ALL = sorted(a for a in ARMS if a.startswith("E020_") and a.endswith("_REF32"))
FULL_E020_A_NEIGHBOR = sorted(
    a for a in FULL_E020_A_ALL
    if any(s in a for s in ("FIRST_PREV", "FIRST_NEXT", "FIRST_KMINUS8", "FIRST_KPLUS8"))
)
FULL_E020_D_CONTROLS = sorted(
    a for a in ARMS if a.startswith("E020_") and "FIRST_SHIFT_" in a
    and not a.endswith("_REF32")
    and not any(x in a for x in ("CFFULL", "CFCUR", "CFNOCTX", "CFSHUF"))
)
FULL_E020_B_LEARNED = sorted(
    a for a in ARMS if a.startswith("E020_") and "FIRST_SHIFT_" in a
    and any(x in a for x in ("CFFULL_G1R1", "CFCUR_G1R1", "CFNOCTX_G1R1", "CFSHUF_G1R1"))
)
FULL_E020_C_TRANSFER = sorted(
    a for a in ARMS if a.startswith("E020_")
    and any(
        s in a
        for s in (
            "_EARLY_CFFULL_G1R1",
            "_THREE_WMATCH_CFFULL_G1R1",
            "_FIRST_KMINUS8_CFFULL_G1R1",
            "_FIRST_KPLUS8_CFFULL_G1R1",
        )
    )
)
FULL_E020_NOLEARN = sorted(set(FULL_E020_A_ALL) | set(FULL_E020_D_CONTROLS))
FULL_E020_PRIORITY = FULL_E020_A_NEIGHBOR + [
    a for a in FULL_E020_A_ALL if a.endswith("_FIRST_SHIFT_REF32")
] + FULL_E020_D_CONTROLS
FULL_E020_LEARNED = sorted(set(FULL_E020_B_LEARNED) | set(FULL_E020_C_TRANSFER))
# Models dir used by continuation arms (populated by fit_models after pairdata).
E020_MODELS_DIR = Path(__file__).resolve().parent / "experiments" / "e020_continuation" / "models"


def apply_arm_env(arm: str) -> dict:
    import os

    variant, b, s, d, target_k = ARMS[arm]
    os.environ.update(
        USE_VITCOP="True",
        E006_ENABLE="True",
        E006_SELECTOR="QED",
        E006_LAYER="12",
        E006_TARGET_K=str(target_k),
        E006_REQUIRE_EXPLICIT_QUESTION_SPAN="True",
        E007_ENABLE="True",
        E007_STAGE1_MODE="ORIGINAL_172",
        E007_SKIP_UNUSED_VIC="True",
        E008_ENABLE="False",
        E009_ENABLE="False",
        E010_ENABLE="True",
        E010_QED_LAYER=str(s),
        E010_CONTEXT_EXIT_LAYER="-1",
        E010_KEEP_CONTEXT_AFTER_QED="False",
        E010_QED_SIDE_ONLY="False",
        E010_ANALYSIS_FULL576="False",
        E012_ENABLE="True",
        E012_DELETE_LAYER=str(d),
        E012_DIAG="False",
        E012_KV_AUDIT="False",
        E013_ENABLE="False",
        STAGE3_POLICY="skip",
        USE_T3GUIDE="False",
        COMPUTE_EFFICIENCY="False",
        E006_PROFILE="False",
        E006_DEBUG="False",
        USE_STAGE2_SRE="False",
        USE_STAGE2_LITE_ROUTER="False",
        USE_STAGE2_WEIGHTED_MERGE="False",
        USE_DYNAMIC_RATIO="False",
        USE_ADAPTIVE_SELECT="False",
        USE_DEBIAS="False",
        USE_ZONE_ROUTING="False",
        USE_COORD_TOKENS="False",
        USE_SPATIAL_AWARE="False",
        E014_ENABLE="True",
        E014_VARIANT=variant,
        E014_B=str(b),
        E014_LAYER=str(s),
        E014_LAMBDA="1.0",
        E014_PENALTY="0.0",
        E014_SEED="20261013",
        E014_AUDIT="False",
        E015_GRID="4",
        E015_JOINT_PROPOSAL="REGION_DUAL_SALIENT",
        E015_JOINT_MEAN_WEIGHT="0.5",
        E015_JOINT_PENALTY="0.0",
        E015_CELLS="4",
        E015_REF_MODE="MEAN",
        E015_DUAL_WEIGHT="0.25",
        E015_ALPHA="0.8",
        E016_BLOCK="8",
        E016_PROTECT="128",
        E016_SPATIAL_WEIGHT="0.50",
        E016_SIGMA="4.0",
        E016_MID_LAYER="8",
        E016_MID_MIX="0.30",
        E016_AUDIT_COVERAGE="False",
    )
    # Dynamic arms require frozen thresholds from unlabeled probe calibration.
    thr_path = os.environ.get(
        "E015_THRESHOLDS_FILE",
        "/home/xiangyue/ViTCoP/experiments/e015_dyn_cover/state/e015_thresholds.json",
    )
    if variant.startswith("E015_") and variant.endswith("_DYNAMIC"):
        if "E015_THRESHOLDS" not in os.environ or not os.environ["E015_THRESHOLDS"].strip():
            import json
            from pathlib import Path
            p = Path(thr_path)
            if not p.is_file():
                raise RuntimeError(
                    f"[E015] missing thresholds for {arm}; expected {p}"
                )
            os.environ["E015_THRESHOLDS"] = json.loads(p.read_text())["E015_THRESHOLDS"]
    elif "E015_THRESHOLDS" in os.environ and not variant.endswith("_DYNAMIC"):
        # Avoid leaking dynamic thresholds into fixed arms in a reused shell.
        os.environ.pop("E015_THRESHOLDS", None)
    if arm.startswith("CASCADE_"):
        os.environ["E016_CASCADE"] = "8:96,10:64,16:32,20:16"
    else:
        # Prevent accidental inheritance from a prior cascade job in the same shell.
        os.environ["E016_CASCADE"] = ""
    if arm.startswith("E018_") or arm.startswith("E019_") or arm.startswith("E020_"):
        # Isolated Stage2/E019/E020 path; keep Stage1 via E014_* above.
        events = None
        e019_sel = {"mode": "reference", "alpha": 0.0, "lambda": 0.0, "swaps": 0}
        e020_kind = {"kind": "reference"}
        if arm.startswith("E020_"):
            body = arm[len("E020_"):]
            if body.endswith("32"):
                body = body[:-2]
            parts = body.split("_")
            # Kind keys may contain underscores (e.g. CFFULL_G1R4). Match longest
            # suffix in _E020_KINDS; do not trust parts[-1] alone.
            stage = parts[0]
            rest = "_".join(parts[1:])
            kind_key = "REF"
            sched = rest
            for cand in sorted(_E020_KINDS.keys(), key=len, reverse=True):
                if rest == cand:
                    kind_key = cand
                    sched = ""
                    break
                suffix = "_" + cand
                if rest.endswith(suffix):
                    kind_key = cand
                    sched = rest[: -len(suffix)]
                    break
            if sched not in _E020_EVENTS:
                raise ValueError(f"Unknown E020 schedule in arm {arm}: {sched}")
            events = _E020_EVENTS[sched]
            e020_kind = dict(_E020_KINDS[kind_key])
            _ = stage  # stage already encoded in ARMS[arm] variant/b
        elif arm.startswith("E019_"):
            # Parse E019_{COVER|BASE}_{SCHEDULE}_{SELECTOR}32 or legacy ..._{SCHEDULE}32
            body = arm[len("E019_"):]
            if body.endswith("32"):
                body = body[:-2]
            parts = body.split("_")
            stage = parts[0]
            # schedule may be multi-token; selector is last known tag or REF
            sel_key = "REF"
            if parts[-1] in _E019_SELECTORS:
                sel_key = parts[-1]
                sched_parts = parts[1:-1]
            else:
                sched_parts = parts[1:]
            sched = "_".join(sched_parts)
            # legacy short arms omit REF
            if sched not in _E019_EVENTS and sel_key == "REF":
                # e.g. COVER_LATE_WMATCH
                sched = "_".join(parts[1:])
            if sched not in _E019_EVENTS:
                raise ValueError(f"Unknown E019 schedule in arm {arm}: {sched}")
            events = _E019_EVENTS[sched]
            e019_sel = dict(_E019_SELECTORS[sel_key])
        elif arm.endswith("_ONE32"):
            events = "8:32"
        elif arm.endswith(("_TWO32", "_EARLY32")):
            events = "4:102,12:32"
        elif arm.endswith("_LATE32") and "WMATCH" not in arm:
            events = "4:102,16:32"
        else:
            raise ValueError(f"Unknown E018/E019 arm schedule: {arm}")
        os.environ.update(
            E018_ENABLE="True",
            E018_MODE="STATIC",
            E018_EVENTS=events,
            E018_SELECTOR="Q_ATTN",
            E018_CACHE_POLICY="layer_local",
            E018_LAMBDA="0",
            E018_MAX_WORK="0",
            E018_TRACE="",
            E018_CONTROLLER="",
            E007_TOKEN_AUDIT="False",
            E014_AUDIT="False",
            E012_DIAG="False",
            E012_KV_AUDIT="False",
            E006_PROFILE="False",
            COMPUTE_EFFICIENCY="False",
        )
        if arm.startswith("E019_"):
            enable = e019_sel["mode"] != "reference"
            os.environ.update(
                E019_ENABLE="True" if enable else "False",
                E019_MODE=str(e019_sel["mode"]),
                E019_ALPHA=str(e019_sel["alpha"]),
                E019_LAMBDA=str(e019_sel["lambda"]),
                E019_SWAPS=str(e019_sel["swaps"]),
                E019_QUERY_GROUPS="4",
                E019_DEBUG=os.environ.get("E019_DEBUG", "True"),
                E020_ENABLE="False",
            )
        elif arm.startswith("E020_"):
            kind = e020_kind.get("kind", "reference")
            enable = kind != "reference"
            # completed-block horizon of final event (zero-based last layer + 1)
            last_layer = int(events.split(",")[-1].split(":")[0])
            model_name = str(e020_kind.get("model", "current"))
            model_path = ""
            accept_enabled = e020_kind.get("accept_enabled", True)
            candidate_limit = e020_kind.get("candidate_limit", 16)
            threshold = e020_kind.get("threshold", 0)
            if kind == "continuation":
                import json
                import numpy as np

                cand = E020_MODELS_DIR / f"{model_name}.npz"
                if not cand.is_file():
                    raise FileNotFoundError(
                        f"E020 continuation arm {arm} needs fitted model at {cand}")
                model_path = str(cand)
                # Freeze deployment + abstention from the fitted checkpoint
                # (R2 rejects g1r1 weights run with a different rounds/group).
                with np.load(cand, allow_pickle=False) as data:
                    meta = json.loads(str(data["metadata"]))
                pol = meta.get("deployment_policy") or {}
                for key in ("rounds", "group_size", "shortlist", "candidate_limit"):
                    if key in pol:
                        e020_kind[key] = pol[key]
                threshold = meta["decision_threshold"]
                accept_enabled = bool(meta["accept_enabled"])
                candidate_limit = int(e020_kind.get("candidate_limit", 16))
            os.environ.update(
                E019_ENABLE="False",
                E020_ENABLE="True" if enable else "False",
                E020_KIND=str(kind),
                E020_MODEL=model_name,
                E020_MODEL_PATH=model_path,
                E020_GROUP_SIZE=str(e020_kind.get("group_size", 1)),
                E020_ROUNDS=str(e020_kind.get("rounds", e020_kind.get("swaps", 1))),
                E020_SHORTLIST=str(e020_kind.get("shortlist", 16)),
                E020_CANDIDATE_LIMIT=str(candidate_limit),
                E020_THRESHOLD=str(threshold),
                E020_ACCEPT_ENABLED="True" if accept_enabled else "False",
                E020_ALPHA=str(e020_kind.get("alpha", 0)),
                E020_LAMBDA=str(e020_kind.get("lambda", 0.15)),
                E020_CURRENT_WEIGHT=str(e020_kind.get("current_weight", 0.6)),
                E020_SEED=str(e020_kind.get("seed", 20)),
                E020_NEXT_AFTER=str(last_layer + 1),
                E020_DEBUG=os.environ.get("E020_DEBUG", "True"),
            )
        else:
            os.environ["E019_ENABLE"] = "False"
            os.environ["E020_ENABLE"] = "False"
    else:
        os.environ["E018_ENABLE"] = "False"
        os.environ["E019_ENABLE"] = "False"
        os.environ["E020_ENABLE"] = "False"
    return {
        "arm": arm,
        "variant": variant,
        "b": b,
        "s": s,
        "d": d,
        "target_k": target_k,
        "visual": 172,
    }
