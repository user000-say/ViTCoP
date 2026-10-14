"""E014: controls for fixed-budget adaptive visual evidence exchange.

Only imported when E014_ENABLE=True.  No learned parameters or global state.
"""
from dataclasses import dataclass
import math
import os

E014_BUDGETS = (0, 8, 16, 24, 32)
E014_VARIANTS = (
    "BASE", "FIXED_COVER", "FIXED_COVER_PROXY",
    "FIXED_RISK", "ADAPTIVE_B_ONLY", "AEE",
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
        )
        if c.variant not in E014_VARIANTS:
            raise ValueError("[E014] invalid E014_VARIANT: " + c.variant)
        if c.b not in E014_BUDGETS:
            raise ValueError("[E014] E014_B must be one of 0,8,16,24,32")
        if not math.isfinite(c.lam) or c.lam < 0:
            raise ValueError("[E014] E014_LAMBDA must be finite and >=0")
        if not math.isfinite(c.penalty) or c.penalty < 0:
            raise ValueError("[E014] E014_PENALTY must be finite and >=0")
        return c
