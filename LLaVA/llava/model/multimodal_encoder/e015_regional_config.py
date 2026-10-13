"""Image-only regional evidence exchange, scoped to E014's 172 real patches."""
from dataclasses import dataclass
import math
import os

FIXED_REGIONAL_VARIANTS = (
    "REGION_MEAN", "REGION_DUAL_SALIENT", "REGION_DUAL_NEAREST",
    "REGION_DUAL_CENTER",
)
REGIONAL_VARIANTS = ("GLOBAL_RETAINED_MEAN",) + FIXED_REGIONAL_VARIANTS + ("REGION_JOINT",)
BUDGETS = (0, 8, 16, 24, 32)


@dataclass(frozen=True)
class RegionalConfig:
    variant: str = "REGION_DUAL_SALIENT"
    b: int = 16
    grid: int = 4
    joint_proposal: str = "REGION_DUAL_SALIENT"
    joint_mean_weight: float = 0.5
    joint_penalty: float = 0.0
    audit: bool = False
    eps: float = 1e-8
    dump_dir: str = ""
    dump_limit: int = 64

    def validate(self):
        if self.variant not in REGIONAL_VARIANTS:
            raise ValueError("[E015] unknown regional variant: " + self.variant)
        if self.joint_proposal not in FIXED_REGIONAL_VARIANTS:
            raise ValueError("[E015] joint proposal must be a fixed regional variant")
        if self.b not in BUDGETS:
            raise ValueError("[E015] b must be 0,8,16,24,32")
        if self.grid not in (3, 4, 6):
            raise ValueError("[E015] grid must be 3,4,6 and divide the 24x24 grid")
        if not math.isfinite(self.joint_mean_weight) or not 0 <= self.joint_mean_weight <= 1:
            raise ValueError("[E015] joint mean weight must be finite and within [0,1]")
        if not math.isfinite(self.joint_penalty) or self.joint_penalty < 0:
            raise ValueError("[E015] joint penalty must be finite and nonnegative")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("[E015] eps must be finite and positive")
        if self.dump_limit < 1:
            raise ValueError("[E015] dump limit must be positive")
        if self.dump_dir and not self.audit:
            raise ValueError("[E015] feature dumping requires E014_AUDIT=True")
        return self

    @classmethod
    def from_e014(cls, legacy):
        return cls(
            variant=legacy.variant, b=legacy.b, audit=legacy.audit,
            grid=int(os.environ.get("E015_GRID", "4")),
            joint_proposal=os.environ.get("E015_JOINT_PROPOSAL", "REGION_DUAL_SALIENT").upper(),
            joint_mean_weight=float(os.environ.get("E015_JOINT_MEAN_WEIGHT", "0.5")),
            joint_penalty=float(os.environ.get("E015_JOINT_PENALTY", "0.0")),
            dump_dir=os.environ.get("E015_DUMP_DIR", ""),
            dump_limit=int(os.environ.get("E015_DUMP_LIMIT", "64")),
        ).validate()
