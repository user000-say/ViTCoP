"""Build a deterministic *anchor-question-only* mismatch control.

The official model's evaluation question and answer key MUST remain unchanged.
Never use raw benchmark question_id as the unique key (can repeat).
"""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import defaultdict

def stable_key(rec):
    required=("dataset","row_index","video_id","question_stem")
    if any(k not in rec for k in required):
        raise ValueError("Required: dataset,row_index,video_id,question_stem")
    value="|".join(str(rec[k]) for k in required)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def build_mismatch(records:list[dict], seed:int=2027) -> list[dict]:
    groups=defaultdict(list)
    for record in records:
        groups[record["dataset"]].append(record)
    keys=[stable_key(r) for r in records]
    if len(set(keys))!=len(keys):
        raise ValueError("Duplicate composite dataset row key")
    out=[]
    rng=random.Random(seed)
    for dataset,items in sorted(groups.items()):
        # Sorted by video ID groups; rotating half a dataset often deranges correctly.
        items=sorted(items,key=lambda r:(str(r["video_id"]),str(r["row_index"])))
        n=len(items)
        if n<2 or len({str(r["video_id"]) for r in items})<2:
            raise ValueError(f"{dataset}: need multiple different videos")
        remaining=list(range(1,n))
        rng.shuffle(remaining)
        offsets=list(dict.fromkeys([n//2]+remaining))
        shift=next((
            shift for shift in offsets
            if all(
                str(items[i]["video_id"]) != str(items[(i+shift)%n]["video_id"])
                for i in range(n)
            )
        ),None)
        if shift is None:
            raise ValueError(f"{dataset}: impossible to derange video groups with rotation")
        for i,rec in enumerate(items):
            other=items[(i+shift)%n]
            out.append({
                "dataset":dataset,
                "row_index":rec["row_index"],
                "sample_key":stable_key(rec),
                "video_id":rec["video_id"],
                "eval_question_stem":rec["question_stem"],
                # This text is ONLY for Stage1 input: do not change generation prompt.
                "anchor_question_stem":other["question_stem"],
                "anchor_question_source_video_id":other["video_id"],
                "anchor_question_source_key":stable_key(other),
            })
    assert len(out)==len(records)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-jsonl",required=True)
    p.add_argument("--output-jsonl",required=True)
    p.add_argument("--seed",type=int,default=2027)
    args=p.parse_args()
    records=[json.loads(s) for s in open(args.input_jsonl,encoding="utf-8") if s.strip()]
    mismatch=build_mismatch(records,seed=args.seed)
    with open(args.output_jsonl,"w",encoding="utf-8") as f:
        for item in mismatch:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
    print(json.dumps({"matched_rows":len(mismatch),"seed":args.seed,"question_only":True}))

if __name__=="__main__":
    main()
