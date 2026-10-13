"""E014: controls for fixed-budget adaptive visual evidence exchange.

Only imported when E014_ENABLE=True.  No learned parameters or global state.
"""
from dataclasses import dataclass
import math
import os

E014_BUDGETS = (0, 8, 16, 24, 32)
E014_VARIANTS = (
    "BASE",
    "FIXED_COVER",
    "FIXED_COVER_PROXY",
    "FIXED_RISK",
    "ADAPTIVE_B_ONLY",
    "AEE",
    # E015 regional v1 (prior install)
    "GLOBAL_RETAINED_MEAN",
    "REGION_MEAN",
    "REGION_DUAL_SALIENT",
    "REGION_DUAL_NEAREST",
    "REGION_DUAL_CENTER",
    "REGION_JOINT",
    # E015 dynamic COVER / regional AEE (2026-10-13)
    "E015_PROBE",
    "E015_COVER_FIXED",
    "E015_COVER_DYNAMIC",
    "E015_AEE_FIXED",
    "E015_AEE_DYNAMIC",
    "E015_GLOBAL_COVER_FIXED",
    "E015_GLOBAL_COVER_DYNAMIC",
    "E015_GLOBAL_AEE_FIXED",
    "E015_GLOBAL_AEE_DYNAMIC",
    "E015_SCORE_RECON",
)

@dataclass(frozen=True)
class E014Config:
    enabled: bool = False
    variant: str = "AEE"
    b: int = 16
    lam: float = 1.0
    penalty: float = 0.0
    audit: bool = False
    seed: int = 20261009
    # E015-only. Ignored by every old E014 variant.
    cells: int = 4
    ref_mode: str = "MEAN"
    dual_weight: float = 0.25
    alpha: float = 0.8
    thresholds: tuple = ()

    @classmethod
    def from_env(cls):
        c = cls(
            enabled=os.environ.get("E014_ENABLE", "False") == "True",
            variant=os.environ.get("E014_VARIANT", "AEE").upper(),
            b=int(os.environ.get("E014_B", "16")),
            lam=float(os.environ.get("E014_LAMBDA", "1.0")),
            penalty=float(os.environ.get("E014_PENALTY", "0.0")),
            audit=os.environ.get("E014_AUDIT", "False") == "True",
            seed=int(os.environ.get("E014_SEED", "20261009")),
            cells=int(os.environ.get("E015_CELLS", "4")),
            ref_mode=os.environ.get("E015_REF_MODE", "MEAN").upper(),
            dual_weight=float(os.environ.get("E015_DUAL_WEIGHT", "0.25")),
            alpha=float(os.environ.get("E015_ALPHA", "0.8")),
            thresholds=tuple(
                float(v.strip())
                for v in os.environ.get("E015_THRESHOLDS", "").split(",")
                if v.strip()
            ),
        )
        if c.variant not in E014_VARIANTS:
            raise ValueError("[E014] invalid E014_VARIANT: " + c.variant)
        if c.b not in E014_BUDGETS:
            raise ValueError("[E014] E014_B must be one of 0,8,16,24,32")
        if not math.isfinite(c.lam) or c.lam < 0:
            raise ValueError("[E014] E014_LAMBDA must be finite and >=0")
        if not math.isfinite(c.penalty) or c.penalty < 0:
            raise ValueError("[E014] E014_PENALTY must be finite and >=0")
        if c.cells not in (2, 4, 6, 8, 12):
            raise ValueError("[E015] cells must be 2,4,6,8,12 per dimension")
        if c.ref_mode not in ("MEAN", "MEDOID", "SALIENT", "DUAL"):
            raise ValueError("[E015] invalid reference; use MEAN/MEDOID/SALIENT/DUAL")
        if not (math.isfinite(c.dual_weight) and 0 <= c.dual_weight <= 1):
            raise ValueError("[E015] E015_DUAL_WEIGHT must be within [0,1]")
        if not (math.isfinite(c.alpha) and c.alpha >= 0):
            raise ValueError("[E015] E015_ALPHA must be finite and >=0")
        if c.variant.endswith("_DYNAMIC") and c.variant.startswith("E015_"):
            if (
                len(c.thresholds) != 4
                or any(not math.isfinite(t) or not 0 < t < 1 for t in c.thresholds)
                or any(c.thresholds[i] >= c.thresholds[i + 1] for i in range(3))
            ):
                raise ValueError(
                    "[E015] dynamic swap requires 4 strictly increasing "
                    "E015_THRESHOLDS in (0,1), calibrated on separate images"
                )
        return c
