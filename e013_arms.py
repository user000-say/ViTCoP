"""E013 arm definitions. No selector math — only env wiring for frozen Stage2."""
from __future__ import annotations

# name -> (e013_enable, variant, pick, b, s, d)
# Stage2 freeze: S=D=8, QED96 unless bridge12 uses S=D=12.
ARMS = {
    "BASE_S8": (False, "EXTRA", "ATTN", 0, 8, 8),
    "B0_S8": (True, "EXTRA", "ATTN", 0, 8, 8),
    "REPLACE_RANDOM16": (True, "REPLACE", "RANDOM", 16, 8, 8),
    "REPLACE_ATTN16": (True, "REPLACE", "ATTN", 16, 8, 8),
    "REPLACE_COVER16": (True, "REPLACE", "COVER", 16, 8, 8),
    "EXTRA_RANDOM16": (True, "EXTRA", "RANDOM", 16, 8, 8),
    "EXTRA_ATTN16": (True, "EXTRA", "ATTN", 16, 8, 8),
    "EXTRA_COVER16": (True, "EXTRA", "COVER", 16, 8, 8),
    "EXTRA_NULL16": (True, "EXTRA", "NULL", 16, 8, 8),
    "EXTRA_DUPLICATE16": (True, "EXTRA", "DUPLICATE", 16, 8, 8),
    "EXTRA_ATTN8": (True, "EXTRA", "ATTN", 8, 8, 8),
    "EXTRA_ATTN32": (True, "EXTRA", "ATTN", 32, 8, 8),
    "EXTRA_RANDOM8": (True, "EXTRA", "RANDOM", 8, 8, 8),
    "EXTRA_RANDOM32": (True, "EXTRA", "RANDOM", 32, 8, 8),
    "EXTRA_COVER8": (True, "EXTRA", "COVER", 8, 8, 8),
    "EXTRA_COVER32": (True, "EXTRA", "COVER", 32, 8, 8),
    "BASE_S12": (False, "EXTRA", "ATTN", 0, 12, 12),
    "EXTRA_RANDOM16_S12": (True, "EXTRA", "RANDOM", 16, 12, 12),
    "EXTRA_ATTN16_S12": (True, "EXTRA", "ATTN", 16, 12, 12),
    "EXTRA_COVER16_S12": (True, "EXTRA", "COVER", 16, 12, 12),
    # Latency crossover arms use E012-only paths (E013 off).
    "LEGACY_L12": (False, "EXTRA", "ATTN", 0, 12, 12),
    "E012_S12D12": (False, "EXTRA", "ATTN", 0, 12, 12),
}

SMOKE = [
    "BASE_S8", "B0_S8", "REPLACE_COVER16", "EXTRA_ATTN16",
    "EXTRA_COVER16", "EXTRA_NULL16", "EXTRA_DUPLICATE16",
]
SCREEN1 = [
    "BASE_S8", "B0_S8", "REPLACE_RANDOM16", "REPLACE_ATTN16",
    "REPLACE_COVER16", "EXTRA_RANDOM16", "EXTRA_ATTN16", "EXTRA_COVER16",
]
SCREEN2 = [
    "EXTRA_NULL16", "EXTRA_DUPLICATE16",
    "EXTRA_ATTN8", "EXTRA_ATTN32",
    "EXTRA_RANDOM8", "EXTRA_RANDOM32",
    "EXTRA_COVER8", "EXTRA_COVER32",
]
# Addendum 2026-10-09: bridge mandatory; Δ12 vs BASE_S12 only (never vs BASE_S8).
# COVER_S12 optional extra; ATTN+RANDOM are the required depth controls.
BRIDGE12 = ["BASE_S12", "EXTRA_ATTN16_S12", "EXTRA_RANDOM16_S12", "EXTRA_COVER16_S12"]
CROSSOVER = ["LEGACY_L12", "E012_S12D12", "E012_S12D12", "LEGACY_L12"]
LATENCY = ["BASE_S8", "REPLACE_ATTN16", "REPLACE_COVER16", "EXTRA_ATTN16", "EXTRA_COVER16"]


def visual_count(arm: str) -> int:
    enable, variant, _pick, b, _s, _d = ARMS[arm]
    if not enable:
        return 172
    if variant == "EXTRA" and b > 0:
        return 172 + b
    return 172


def apply_arm_env(arm: str) -> dict:
    import os

    enable, variant, pick, b, s, d = ARMS[arm]
    # Force frozen Stage2 + E012 S=D.
    os.environ.update(
        USE_VITCOP="True",
        E006_ENABLE="True",
        E006_SELECTOR="QED",
        E006_LAYER="12",
        E006_TARGET_K="96",
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
        E012_ENABLE="True",
        E012_DELETE_LAYER=str(d),
        E012_DIAG="False",
        E012_KV_AUDIT="False",
        STAGE3_POLICY="skip",
        USE_T3GUIDE="False",
        COMPUTE_EFFICIENCY="False",
        E006_PROFILE="False",
        E006_DEBUG="False",
        E013_SEED="20261009",
    )
    if enable:
        os.environ.update(
            E013_ENABLE="True",
            E013_VARIANT=variant,
            E013_PICK=pick,
            E013_B=str(b),
        )
    else:
        os.environ["E013_ENABLE"] = "False"
        os.environ.pop("E013_VARIANT", None)
        os.environ.pop("E013_PICK", None)
        os.environ.pop("E013_B", None)
    n = visual_count(arm)
    os.environ["E009_EXPECT_VISUAL"] = str(n)
    os.environ["E007_EXPECT_STAGE1"] = "172"
    return {
        "arm": arm,
        "enable": enable,
        "variant": variant,
        "pick": pick,
        "b": b,
        "S": s,
        "D": d,
        "visual": n,
    }
