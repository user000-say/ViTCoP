#!/usr/bin/env python3
"""Unlabeled DEV calibration -> frozen JSON, or held-out inference planning.

For proof-of-concept only. Input JSONL: {image_id, difficulty, early_drift?}.
The difficulty must come from real COVER greedy evidence (not from labels).
"""
import argparse
import json
from pathlib import Path
from bela_next.policy import QuantilePolicy, plan_for_image, audit_plans

def load_rows(p):
    rows=[]
    for i,line in enumerate(Path(p).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        r=json.loads(line)
        if "image_id" not in r or "difficulty" not in r:
            raise ValueError(f"line {i}: image_id and difficulty needed")
        rows.append(r)
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["calibrate","plan"])
    parser.add_argument("--input",required=True)
    parser.add_argument("--policy",required=True)
    parser.add_argument("--output")
    parser.add_argument("--mode",choices=["fixed","adaptive"],default="fixed")
    parser.add_argument("--model-revision",default="unknown")
    args=parser.parse_args()
    rows=load_rows(args.input)
    if args.command=="calibrate":
        policy=QuantilePolicy.calibrate(
            [r["difficulty"] for r in rows],model_revision=args.model_revision)
        policy.save(args.policy)
        print(json.dumps(policy.__dict__,ensure_ascii=False,indent=2))
        return
    if not args.output: raise ValueError("--output required for planning")
    policy=QuantilePolicy.load(args.policy)
    plans=[]
    lines=[]
    for r in rows:
        plan=plan_for_image(policy,float(r["difficulty"]),args.mode,r.get("early_drift"))
        plans.append(plan)
        lines.append(json.dumps({"image_id":r["image_id"],**plan.as_dict()},ensure_ascii=False))
    Path(args.output).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(audit_plans(plans),ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
