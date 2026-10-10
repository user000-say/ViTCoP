"""E014 arm definitions. Env wiring only — no selector math."""
from __future__ import annotations

# name -> (variant, b, s, d, target_k)
# Stage2 freeze: S=D=8 primary; QED final-K set per arm (96/64/32).
ARMS: dict[str, tuple[str, int, int, int, int]] = {
    # K96 primary matrix
    "BASE_K96": ("BASE", 0, 8, 8, 96),
    "FIXED_COVER_B0_K96": ("FIXED_COVER", 0, 8, 8, 96),
    "FIXED_COVER_B8_K96": ("FIXED_COVER", 8, 8, 8, 96),
    "FIXED_COVER_B16_K96": ("FIXED_COVER", 16, 8, 8, 96),
    "FIXED_COVER_B24_K96": ("FIXED_COVER", 24, 8, 8, 96),
    "FIXED_COVER_B32_K96": ("FIXED_COVER", 32, 8, 8, 96),
    "FIXED_RISK_B16_K96": ("FIXED_RISK", 16, 8, 8, 96),
    "ADAPTIVE_B_ONLY_K96": ("ADAPTIVE_B_ONLY", 16, 8, 8, 96),
    "AEE_K96": ("AEE", 16, 8, 8, 96),
    # K64 stress (after K96 proof)
    "BASE_K64": ("BASE", 0, 8, 8, 64),
    "FIXED_COVER_B16_K64": ("FIXED_COVER", 16, 8, 8, 64),
    "FIXED_RISK_B16_K64": ("FIXED_RISK", 16, 8, 8, 64),
    "ADAPTIVE_B_ONLY_K64": ("ADAPTIVE_B_ONLY", 16, 8, 8, 64),
    "AEE_K64": ("AEE", 16, 8, 8, 64),
    # K32 stress
    "BASE_K32": ("BASE", 0, 8, 8, 32),
    "FIXED_COVER_B16_K32": ("FIXED_COVER", 16, 8, 8, 32),
    "FIXED_RISK_B16_K32": ("FIXED_RISK", 16, 8, 8, 32),
    "ADAPTIVE_B_ONLY_K32": ("ADAPTIVE_B_ONLY", 16, 8, 8, 32),
    "AEE_K32": ("AEE", 16, 8, 8, 32),
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
    "FIXED_RISK_B16_K64",
    "ADAPTIVE_B_ONLY_K64",
    "AEE_K64",
]
FULL_K32 = [
    "BASE_K32",
    "FIXED_COVER_B16_K32",
    "FIXED_RISK_B16_K32",
    "ADAPTIVE_B_ONLY_K32",
    "AEE_K32",
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
        E014_LAMBDA="1.0",
        E014_PENALTY="0.0",
        E014_SEED="20261009",
        E014_AUDIT="False",
    )
    return {
        "arm": arm,
        "variant": variant,
        "b": b,
        "s": s,
        "d": d,
        "target_k": target_k,
        "visual": 172,
    }
