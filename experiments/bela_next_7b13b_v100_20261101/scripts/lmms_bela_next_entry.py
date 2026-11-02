#!/usr/bin/env python3
"""lmms_eval entry that applies BELA NeXT Dual COVER + Stage2 after LlavaHf load."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PKG = Path("/home/xiangyue/ViTCoP/experiments/bela_next_7b13b_v100_20261101")
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "BELA_NEXT_7B13B_V100_CURSOR_20261101/src"))
sys.path.insert(0, "/home/xiangyue/ViTCoP/lmms-eval")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Must set before model load
if os.environ.get("BELA_NEXT_V2_ENABLE", "0") not in ("1", "true", "TRUE", "yes"):
    raise SystemExit("BELA_NEXT_V2_ENABLE=1 required for this entry (refuse Dense-as-pruned)")
if not os.environ.get("BELA_NEXT_ARM", "").strip():
    raise SystemExit("BELA_NEXT_ARM required")

from lmms_eval.models.llava_hf import LlavaHf
from bela_next_forward.apply_llava_next import apply_patch

_orig_init = LlavaHf.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    apply_patch(self._model)
    # NATIVE FIX KIT: do NOT force [[672,672]] — match 2026-10-28 Dense processor.
    pin = None
    try:
        pin = getattr(getattr(self._image_processor, "image_processor", self._image_processor), "image_grid_pinpoints", None)
    except Exception:
        pass
    print(
        f"[BELA_NEXT] patched arm={os.environ['BELA_NEXT_ARM']} "
        f"enable={os.environ.get('BELA_NEXT_V2_ENABLE')} "
        f"model_patched={getattr(self._model, '_bela_next_v2_patched', False)} "
        f"pinpoints={pin} (native, not forced)",
        flush=True,
    )


LlavaHf.__init__ = _patched_init

from lmms_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
