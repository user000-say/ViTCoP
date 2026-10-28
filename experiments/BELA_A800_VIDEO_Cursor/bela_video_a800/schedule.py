"""LLaVA-Video-7B-Qwen2 28-layer STAR-Pro-matched layer-average schedules.

These schedules are *public comparison protocols*, not new BELA inventions.
BELA's original QA-COVER selection formula is kept unchanged.
Cuts are pruning AFTER 1-based layers 12 and 21, i.e. zero-based
decoder forward boundaries after index 11 and index 20.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json

VIDEO_LAYERS = 28
FRAME_PATCHES = 169
FRAME_COUNT = 64
CUTS_AFTER_LAYERS_1BASED = (12, 21)

# From STAR-Pro 2026 arXiv:2609.05916 Table 5.
# (initial per-frame, after layer 12, after layer 21)
SCHEDULES = {
    "T32": (64, 11, 4),
    "T16": (32, 5, 3),
}


@dataclass(frozen=True)
class VideoBudget:
    tier: str
    frames: int
    patches_per_frame: int
    decoder_layers: int
    cut_layers_after: tuple[int, int]
    framewise_counts: tuple[int, int, int]
    layerwise_frame_k: tuple[int, ...]
    entry_total_k: int
    final_total_k: int
    visual_token_layer_W: int
    dense_visual_token_layer_W: int
    normalized_keep_fraction: float
    nominal_reduction_fraction: float
    stage1_frame_token_reduction_fraction: float

    def to_json(self):
        return asdict(self)


def budget(tier: str, frames: int = FRAME_COUNT,
           patches_per_frame: int = FRAME_PATCHES) -> VideoBudget:
    if tier not in SCHEDULES:
        raise ValueError(f"Unknown video budget {tier}")
    if frames <= 0 or frames > FRAME_COUNT:
        raise ValueError("LLaVA-Video 7B official checkpoint supports 1-64 frames")
    if patches_per_frame <= 0:
        raise ValueError("Invalid pooled patch count")
    entry, second, final = SCHEDULES[tier]
    if entry > patches_per_frame:
        raise ValueError(
            f"Expected >= {entry} decoder-visible pooled patches/frame, "
            "never silently lower a paper-matched budget"
        )
    layers = (
        (entry,)*12 + (second,)*9 + (final,)*7
    )
    assert len(layers) == VIDEO_LAYERS
    W = frames * sum(layers)
    dense = VIDEO_LAYERS * frames * patches_per_frame
    return VideoBudget(
        tier=tier, frames=frames, patches_per_frame=patches_per_frame,
        decoder_layers=VIDEO_LAYERS,
        cut_layers_after=CUTS_AFTER_LAYERS_1BASED,
        framewise_counts=(entry, second, final),
        layerwise_frame_k=layers,
        entry_total_k=frames*entry,
        final_total_k=frames*final,
        visual_token_layer_W=W,
        dense_visual_token_layer_W=dense,
        normalized_keep_fraction=W/dense,
        nominal_reduction_fraction=1-W/dense,
        stage1_frame_token_reduction_fraction=1-entry/patches_per_frame,
    )


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--tier",choices=sorted(SCHEDULES),required=True)
    p.add_argument("--frames",type=int,default=64)
    p.add_argument("--patches-per-frame",type=int,default=169)
    args=p.parse_args()
    print(json.dumps(budget(args.tier,args.frames,args.patches_per_frame).to_json(),indent=2))


if __name__ == "__main__":
    main()
