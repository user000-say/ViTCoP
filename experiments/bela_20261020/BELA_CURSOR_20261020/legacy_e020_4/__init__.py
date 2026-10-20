"""E020.3 reference operators. GPU modules are imported explicitly."""
from .recycle import recycle, cosine_cache, AffinityCache
from .runtime import RecyclingState
from .fast_selectors import pooled_qk, key_energy, topk

__all__ = ["recycle", "cosine_cache", "AffinityCache", "RecyclingState",
           "pooled_qk", "key_energy", "topk"]
