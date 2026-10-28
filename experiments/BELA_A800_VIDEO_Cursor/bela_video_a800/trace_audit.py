"""Reject malformed video token traces before logging video Accuracy.

IDs are ORIGINAL (frame, pooled-patch) identities, not mutable decoder slots.
A passing trace validates only metadata, NOT real cache/position correctness.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .schedule import budget

def _check_stage(name, ids, f, n, k, allowed=None):
    if not isinstance(ids,list) or len(ids)!=f:
        raise ValueError(f"{name}: must have {f} per-frame rows")
    for frame,row in enumerate(ids):
        if not isinstance(row,list) or len(row)!=k:
            raise ValueError(f"{name}: frame {frame} expected {k} IDs")
        if any(type(i)!=int or i<0 or i>=n for i in row):
            raise ValueError(f"{name}: frame {frame} bad original patch ID")
        if len(set(row)) != len(row):
            raise ValueError(f"{name}: frame {frame} duplicate ID")
        if allowed is not None and not set(row).issubset(allowed[frame]):
            raise ValueError(f"{name}: frame {frame} resurrected a pruned patch")


def check_trace(row: dict, require_physical: bool = False):
    tier=row["tier"]
    f=int(row["num_frames"])
    n=int(row["pooled_patches_per_frame"])
    b=budget(tier,frames=f,patches_per_frame=n)
    entry, k1, k2=b.framewise_counts
    s0=row["entry_frame_patch_ids"]
    s1=row["after_layer12_frame_patch_ids"]
    s2=row["after_layer21_frame_patch_ids"]
    _check_stage("entry",s0,f,n,entry)
    _check_stage("after_layer12",s1,f,n,k1,[set(x) for x in s0])
    _check_stage("after_layer21",s2,f,n,k2,[set(x) for x in s1])
    if "frame_timestamps_sec" in row:
        times=row["frame_timestamps_sec"]
        if len(times)!=f or any(not isinstance(t,(int,float)) or t<0 for t in times):
            raise ValueError("invalid timestamp vector")
        if any(times[i]>times[i+1] for i in range(f-1)):
            raise ValueError("frame order was changed")
    if "original_llm_sequence_image_positions" in row:
        # A packed image-only frame×N map, in the FULL decoder prompt.
        positions=row["original_llm_sequence_image_positions"]
        if len(positions)!=f or any(len(x)!=n for x in positions):
            raise ValueError("LLM positions do not cover all image patches")
        all_pos=[int(j) for x in positions for j in x]
        if any(j<0 for j in all_pos) or len(all_pos)!=len(set(all_pos)):
            raise ValueError("invalid full-sequence visual positions")
    if "observed_visual_k_by_decoder_layer" in row:
        obs=row["observed_visual_k_by_decoder_layer"]
        exp=[f*k for k in b.layerwise_frame_k]
        if obs!=exp:
            raise ValueError(f"ACTUAL decoder visual K differs from target: {obs} vs {exp}")
    elif require_physical:
        raise ValueError("No observed layerwise K: cannot claim physical pruning")
    if require_physical and not row.get("position_and_cache_regression_pass",False):
        raise ValueError("RoPE and per-layer KV cache were not independently verified")
    return {
        "status":"TRACE_PASS",
        "tier":tier,"frames":f,"pooled_per_frame":n,
        "entry_total":f*entry,"after_12_total":f*k1,
        "after_21_total":f*k2,
        "W":b.visual_token_layer_W,
        "limitations":"Token count/identity only; no video QA quality was measured",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--jsonl",required=True)
    p.add_argument("--require-physical",action="store_true")
    args=p.parse_args()
    n=0
    with open(args.jsonl,encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                check_trace(json.loads(raw),require_physical=args.require_physical)
                n+=1
    print(json.dumps({"trace_count_passed":n,"scope":"metadata check only"}))


if __name__=="__main__":
    main()
