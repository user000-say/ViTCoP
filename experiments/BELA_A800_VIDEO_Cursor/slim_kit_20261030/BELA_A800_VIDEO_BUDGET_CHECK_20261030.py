#!/usr/bin/env python3
"""Check LLaVA-Video budget schedules before using a STAR-Pro-like protocol.
This is NOT a GPU profiler or A800 model verification.
"""
from __future__ import annotations
import argparse
import json
from typing import Any

def budget(frames: int, patches_per_frame: int, layers: int,
           segment_lengths: tuple[int, int, int], keep: tuple[int, int, int],
           swaps_per_frame: int) -> dict[str, Any]:
    if not all(isinstance(x,int) and not isinstance(x,bool) for x in
               (frames,patches_per_frame,layers,swaps_per_frame,*segment_lengths,*keep)):
        raise ValueError("integer values required")
    if min(frames,patches_per_frame,layers,*segment_lengths,*keep) <= 0:
        raise ValueError("all sizes/segments must be positive")
    if sum(segment_lengths) != layers:
        raise ValueError("segment lengths must sum to decoder layers")
    if not (patches_per_frame >= keep[0] >= keep[1] >= keep[2]):
        raise ValueError("keep must be monotone and <= input patches")
    if not 0 <= swaps_per_frame <= keep[0]:
        raise ValueError("swap count outside candidate pool")
    wk = sum(n*k for n,k in zip(segment_lengths,keep))
    original_per_frame_work = layers*patches_per_frame
    return {
        "num_frames":frames,
        "input_patches_per_frame":patches_per_frame,
        "num_decoder_layers":layers,
        "keep_per_frame":keep,
        "segment_lengths":segment_lengths,
        "swaps_per_frame":swaps_per_frame,
        "original_visual_tokens_total":frames*patches_per_frame,
        "initial_visual_tokens_total":frames*keep[0],
        "first_checkpoint_visual_tokens_total":frames*keep[1],
        "final_visual_tokens_total":frames*keep[2],
        "visual_token_layers_total":frames*wk,
        "layer_average_tokens_per_frame":wk/layers,
        "layer_average_tokens_per_video":frames*wk/layers,
        "layer_average_retention_percent":100*wk/original_per_frame_work,
        "layer_average_reduction_percent":100*(1-wk/original_per_frame_work),
        "max_changed_fraction_percent":100*swaps_per_frame/keep[0],
    }

def self_test():
    a = budget(64,169,28,(12,9,7),(64,11,4),8)
    b = budget(64,169,28,(12,9,7),(32,5,3),4)
    assert a["visual_token_layers_total"] == 57280
    assert b["visual_token_layers_total"] == 28800
    assert abs(a["layer_average_tokens_per_frame"] - 895/28) < 1e-12
    assert abs(b["layer_average_tokens_per_frame"] - 450/28) < 1e-12
    assert a["max_changed_fraction_percent"] == b["max_changed_fraction_percent"] == 12.5
    # 1 frame and tiny per-frame budget; every segment must remain positive.
    tiny = budget(1,3,3,(1,1,1),(3,2,1),0)
    assert tiny["visual_token_layers_total"] == 6
    invalid = [
        (0,169,28,(12,9,7),(64,11,4),8),
        (64,169,28,(12,9,8),(64,11,4),8),
        (64,169,28,(12,9,7),(64,11,12),8),
        (64,169,28,(12,9,7),(64,11,4),65),
        (64,32,28,(12,9,7),(64,11,4),8),
        (64,169,28,(12,9,7),(64,11,4),-1),
        (64,169,28,(12,9,7),(64,11,4),True),
    ]
    for args in invalid:
        try:
            budget(*args)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {args!r}")
    print("PASS: 2 STAR schedules + tiny boundary + 7 invalid scenarios")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--patches", type=int, default=169)
    parser.add_argument("--profile", choices=["T32","T16"],default="T32")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    keep, b = ((64,11,4),8) if args.profile=="T32" else ((32,5,3),4)
    print(json.dumps(budget(args.frames,args.patches,28,(12,9,7),keep,b),
                     ensure_ascii=False,indent=2))
if __name__ == "__main__":
    main()
