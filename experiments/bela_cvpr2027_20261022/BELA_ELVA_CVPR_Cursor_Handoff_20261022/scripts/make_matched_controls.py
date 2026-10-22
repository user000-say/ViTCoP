#!/usr/bin/env python3
"""Shuffle realized per-image profile assignments while preserving EXACT arm counts.

Use on heldout image IDs only; no labels. Avoid repeated image IDs here:
for repeated image+question samples, stage1 profile must be rejoined by image_id.
The shuffle is a negative control, NEVER the adaptive deployment policy.
"""
import argparse, json, random
from pathlib import Path
from bela_next.policy import QuantilePolicy, Plan, audit_plans
from bela_next.budget import PROFILES

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--policy",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--seed",type=int,default=20261022)
    args=ap.parse_args()
    rows=[json.loads(s) for s in Path(args.input).read_text(encoding="utf-8").splitlines() if s.strip()]
    ids=[str(r["image_id"]) for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError("Unique IMAGE-level IDs required: merge repeated questions first")
    if len(rows)<2:
        raise ValueError("Need >=2 image groups")
    policy=QuantilePolicy.load(args.policy)
    original=[policy.choose_profile(float(r["difficulty"])) for r in rows]
    shuffled=original.copy()
    rng=random.Random(args.seed)
    if len(set(original))<2:
        raise ValueError("No budget variation: shuffled control cannot test adaptivity")
    for _attempt in range(20):
        rng.shuffle(shuffled)
        if shuffled!=original:
            break
    if shuffled==original:
        raise RuntimeError("Could not build a non-identity matched shuffle")
    out=[]
    plans=[]
    for row,pr in zip(rows,shuffled):
        p=Plan(pr,PROFILES[pr].schedule(),float(row["difficulty"]),
               "fixed",None,"negative_control_permuted_profile")
        plans.append(p)
        out.append(json.dumps({"image_id":row["image_id"],**p.as_dict()},ensure_ascii=False))
    assert sorted(original)==sorted(shuffled)
    Path(args.output).write_text("\n".join(out)+"\n",encoding="utf-8")
    print(json.dumps({"label":"SHUFFLED_NO_ACCURACY_LABELS",**audit_plans(plans)},indent=2))
if __name__=="__main__":main()
