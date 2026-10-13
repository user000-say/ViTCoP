"""E014 arm definitions. Env wiring only — no selector math."""
from __future__ import annotations

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
}

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
    )
    if arm.startswith("CASCADE_"):
        os.environ["E016_CASCADE"] = "8:96,10:64,16:32,20:16"
    else:
        # Prevent accidental inheritance from a prior cascade job in the same shell.
        os.environ["E016_CASCADE"] = ""
    return {
        "arm": arm,
        "variant": variant,
        "b": b,
        "s": s,
        "d": d,
        "target_k": target_k,
        "visual": 172,
    }
