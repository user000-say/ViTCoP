"""Capture SigLIP mid + penultimate 729-token maps for Dual-COVER adaptation."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

_ORIG_FORWARD = None
MID_LAYER = 9  # match image Layer9 index into remaining encoder stack


def dual_capture_enabled() -> bool:
    m = os.environ.get("BELA_VIDEO_MODE", "").strip().lower()
    return m.startswith("dual") or os.environ.get("BELA_VIDEO_DUAL", "0") in ("1", "true", "yes")


def extract_dual_729(vision_tower, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mid_729, pen_729, last_729) each [B,729,D]."""
    vt = vision_tower.vision_tower if hasattr(vision_tower, "vision_tower") else vision_tower
    out = vt(images.to(device=vision_tower.device, dtype=vision_tower.dtype), output_hidden_states=True)
    hs = out.hidden_states
    if hs is None or len(hs) < 3:
        raise RuntimeError("SigLIP did not return enough hidden_states")
    # hidden_states[0]=embeddings; encoder layers follow
    # After deleting last layer, len varies; use index 9 into full tuple carefully.
    # Prefer: last=hs[-1], pen=hs[-2], mid=hs[min(9+1, len-3)] if hs[0] is embed.
    last = hs[-1]
    pen = hs[-2]
    # Layer9 of encoder ≈ index 10 if hs[0] is embedding (1+9)
    mid_idx = 1 + MID_LAYER if len(hs) > MID_LAYER + 2 else max(1, len(hs) // 3)
    mid = hs[mid_idx]
    for name, t in ("mid", mid), ("pen", pen), ("last", last):
        if t.shape[-2] != 729:
            raise RuntimeError(f"{name} tokens {t.shape} != 729")
    return mid, pen, last


def install_vision_capture(model) -> bool:
    """Monkey-patch vision tower forward to stash dual maps on the Llava model."""
    global _ORIG_FORWARD
    tower = model.get_vision_tower()
    if getattr(tower, "_bela_dual_capture", False):
        return False

    if _ORIG_FORWARD is None:
        _ORIG_FORWARD = tower.forward

    parent = model

    def wrapped(images):
        if not dual_capture_enabled():
            return _ORIG_FORWARD(images)
        # list path unused for video tensor batches
        if isinstance(images, list):
            return _ORIG_FORWARD(images)
        mid, pen, last = extract_dual_729(tower, images)
        parent._bela_dual_vit = {
            "mid_729": mid.detach(),
            "pen_729": pen.detach(),
            "last_729": last.detach(),
            "mid_layer_index": MID_LAYER,
        }
        # Match stock tower: return last hidden (payload path unchanged)
        return last.to(images.dtype)

    tower.forward = wrapped
    tower._bela_dual_capture = True
    return True
