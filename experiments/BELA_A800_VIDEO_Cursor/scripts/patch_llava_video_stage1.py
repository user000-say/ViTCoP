#!/usr/bin/env python3
"""Insert Stage1 video prune hook into live llava_arch.py. Idempotent."""
from __future__ import annotations

import sys
from pathlib import Path

ARCH = Path("/data1/xiangyue/src/LLaVA-NeXT/llava/model/llava_arch.py")
MARKER = "BELA_VIDEO_STAGE1_HOOK"

HOOK_METHOD = f'''
    def _bela_video_stage1_maybe_flat(self, pooled, input_ids):
        """Return flattened pruned video tokens, or None to keep the official path."""
        # {MARKER}
        import os
        mode = os.environ.get("BELA_VIDEO_MODE", "off").strip().lower() or "off"
        if mode in ("off", "dense", "0", "false", "no", ""):
            return None
        sys_path = "/data1/xiangyue/ViTCoP/experiments/BELA_A800_VIDEO_Cursor"
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        from bela_video_a800.live_stage1 import select_and_gather
        gathered, trace = select_and_gather(model=self, pooled=pooled, input_ids=input_ids)
        # Keep one structural newline per frame; do not delete time/question tokens.
        framed = self.add_token_per_frame(gathered)
        flat = framed.flatten(0, 1)
        self._bela_last_video_stage1 = {{
            "mode": trace.mode,
            "tier": trace.tier,
            "frames": trace.frames,
            "patches": trace.patches,
            "keep_per_frame": trace.keep_per_frame,
            "swaps": list(trace.swaps),
            "selector": trace.selector,
            "reason": trace.reason,
            "ids": trace.ids.tolist(),
            "flat_tokens": int(flat.shape[0]),
        }}
        return flat
'''

OLD = """                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)"""

NEW = """                if idx in video_idx_in_batch:
                    pooled = self.get_2dPool(image_feat)
                    bela_flat = self._bela_video_stage1_maybe_flat(pooled, input_ids)
                    if bela_flat is not None:
                        image_features.append(bela_flat)
                    else:
                        image_features.append(pooled)
                else:
                    image_features.append(image_feat)"""

# When features are already flattened by BELA, skip spatial grid packing.
OLD_SPATIAL = """                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":"""

NEW_SPATIAL = """                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if isinstance(image_feature, torch.Tensor) and image_feature.ndim == 2:
                            # Already flattened by BELA Stage1 (visual + per-frame newline).
                            new_image_features.append(image_feature)
                            continue
                        if mm_newline_position == "grid":"""


def main() -> int:
    text = ARCH.read_text()
    if MARKER in text and OLD not in text and "bela_flat = self._bela_video_stage1_maybe_flat" in text:
        print("ALREADY_PATCHED", ARCH)
        return 0
    if OLD not in text:
        print("HOOK_SITE_MISSING", file=sys.stderr)
        return 2
    if OLD_SPATIAL not in text:
        print("SPATIAL_SITE_MISSING", file=sys.stderr)
        return 3
    if MARKER not in text:
        # Insert method before prepare_inputs_labels_for_multimodal
        anchor = "    def prepare_inputs_labels_for_multimodal("
        if anchor not in text:
            print("ANCHOR_MISSING", file=sys.stderr)
            return 4
        text = text.replace(anchor, HOOK_METHOD + "\n" + anchor, 1)
    text = text.replace(OLD, NEW, 1)
    text = text.replace(OLD_SPATIAL, NEW_SPATIAL, 1)
    backup = ARCH.with_suffix(".py.pre_bela_stage1")
    if not backup.exists():
        backup.write_text(ARCH.read_text())
    ARCH.write_text(text)
    print("PATCHED", ARCH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
