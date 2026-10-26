"""Paired 2x2 component attribution; uses official-scored per-sample prediction exports.
Arms: BB (BELA stage1+BELA stage2); QB (STAR QR + BELA stage2),
      BS (BELA stage1 + STAR stage2), QS (official STAR).
This is NOT a replacement for the official evaluator or a causal guarantee.
Input JSONL: sample_id, image_id, correct(bool), gold_yes_no/pred_yes_no for POPE.
"""
from pathlib import Path
from collections import defaultdict
import json,random

KEYS=("BB","QB","BS","QS")
def read(path):
    out={}
    for lineno,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        key=row.get("sample_id");image=row.get("image_id")
        if not isinstance(key,str) or not isinstance(image,str) or type(row.get("correct")) is not bool:
            raise ValueError(f"invalid sample_id/image_id/correct at line {lineno}")
        if key in out:raise ValueError(f"duplicate ID {key}")
        out[key]=row
    if not out:raise ValueError("empty results")
    return out

def validate(arms,mode):
    if set(arms)!=set(KEYS):raise ValueError("four exact arm keys are required")
    IDs=set(arms["BB"])
    if not IDs or any(set(arms[key])!=IDs for key in KEYS):
        raise ValueError("not a paired four-arm experiment")
    for sid in IDs:
        iid=arms["BB"][sid]["image_id"]
        for key in KEYS:
            row=arms[key][sid]
            if row["image_id"]!=iid:raise ValueError("different image for same question")
            if mode=="pope_f1":
                if row.get("gold_yes_no") not in ("yes","no") or row.get("pred_yes_no") not in ("yes","no"):
                    raise ValueError("invalid official parsed POPE yes/no")
                if row["gold_yes_no"]!=arms["BB"][sid]["gold_yes_no"]:
                    raise ValueError("mismatched gold POPE label")
                if row.get("subset")!=arms["BB"][sid].get("subset"):
                    raise ValueError("mismatched POPE subset")
    return IDs

def measure(arms,mode,ids):
    if mode=="acc":
        return {a:100*sum(arms[a][sid]["correct"] for sid in ids)/len(ids) for a in KEYS}
    if mode!="pope_f1":raise ValueError("only accuracy and POPE F1 supported")
    out={}
    for arm in KEYS:
        tp=fp=fn=0
        for sid in ids:
            row=arms[arm][sid];gold=row["gold_yes_no"];pred=row["pred_yes_no"]
            if pred=="yes" and gold=="yes":tp+=1
            elif pred=="yes":fp+=1
            elif gold=="yes":fn+=1
        out[arm]=100*(2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    return out

def effects(metric):
    BB,QB,BS,QS=[metric[k] for k in KEYS]
    return dict(stage1_given_B_stage2=QB-BB,
                stage1_given_S_stage2=QS-BS,
                stage2_given_B_stage1=BS-BB,
                stage2_given_Q_stage1=QS-QB,
                interaction=QS-QB-BS+BB)

def analyze(arms,mode="acc",boot=500,seed=20261025):
    ids=validate(arms,mode)
    if type(boot)!=int or boot<1:raise ValueError("bootstrap must be >=1 integer")
    byimage=defaultdict(list)
    for sid in ids:byimage[arms["BB"][sid]["image_id"]].append(sid)
    clusters=list(byimage.values())
    point=measure(arms,mode,list(ids))
    observed=effects(point)
    draws={k:[] for k in observed}
    rng=random.Random(seed)
    for _ in range(boot):
        sample=[sid for __ in clusters for sid in clusters[rng.randrange(len(clusters))]]
        part=effects(measure(arms,mode,sample))
        for k,v in part.items():draws[k].append(v)
    intervals={}
    for k,vals in draws.items():
        vals.sort(); intervals[k]=[vals[int(0.025*(boot-1))],vals[int(0.975*(boot-1))]]
    return {"mode":mode,"n_questions":len(ids),"n_image_clusters":len(clusters),
            "arm_scores_percent":point,"point_effects_pp":observed,
            "image_cluster_bootstrap95_effects_pp":intervals,
            "note":"Factorial component swap is diagnostic, not proof of causal modular independence."}

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser()
 for key in KEYS:p.add_argument(f"--{key}",required=True)
 p.add_argument("--mode",choices=("acc","pope_f1"),default="acc")
 p.add_argument("--bootstrap",type=int,default=500)
 a=p.parse_args()
 raw={k:read(getattr(a,k)) for k in KEYS}
 print(json.dumps(analyze(raw,a.mode,a.bootstrap),indent=2,ensure_ascii=False))
