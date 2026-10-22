"""Research-only, training-free BELA/ELVA budget policy components.

Nothing here alters Hugging Face KV caches or decoder masks. The live-model
integration is OFF by default and must pass the server's own validation gates.
"""
from .budget import Schedule, PolicyProfile, PROFILES, schedule_for_cut
from .signals import coverage_difficulty, alignment_drift
from .policy import QuantilePolicy, plan_for_image
__all__ = [
    "Schedule", "PolicyProfile", "PROFILES", "schedule_for_cut",
    "coverage_difficulty", "alignment_drift",
    "QuantilePolicy", "plan_for_image",
]
