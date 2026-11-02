"""Fail-closed audit of each real V100 run's sample-level budget manifest.

This operates on JSONL from the LIVE integration, not offline estimates.
Each record:
{id, arm_id, model, sample_patch_n, crop_patch_counts,
 crop_original_id_counts, visual_token_counts_per_layer:[L], prune_cut_after:[cut1,cut2],
 prompt_sha256, dense_prompt_sha256, method_stage:...,
 newline_count_before, newline_count_after, newline_count_layer_visibilities:...}
For paper 5-crop reference require N exactly2880.
For native dynamic AnyRes separately report actual per-image ratios, not nominal.
"""
import argparse
import json
from pathlib import Path
from bela_next_budget.budget import source_schedule

def run_check(cfg,records,*,mode="paper"):
    arms={a["arm_id"]:a for a in cfg["arm_matrix"]}
    ids=set()
    out=[]
    for rec in records:
        sampleid=rec["id"]
        arm=arms.get(rec["arm_id"])
        if arm is None:raise ValueError(f"Unknown arm {rec['arm_id']}")
        if (rec["arm_id"],sampleid) in ids:raise ValueError("Duplicate sample in arm")
        ids.add((rec["arm_id"],sampleid))
        schedule=source_schedule(arm)
        n=rec["sample_patch_n"]
        crops=rec["crop_patch_counts"]
        if sum(crops)!=n or any(c<0 for c in crops):
            raise ValueError("Crop token counts do not sum to patch total")
        if mode=="paper" and (n!=2880 or len(crops)!=5 or any(c!=576 for c in crops)):
            raise ValueError("Not a 5×576 paper-reference input, do not compare to STAR")
        if rec.get("prompt_sha256")!=rec.get("dense_prompt_sha256"):
            raise ValueError("Dense vs compressed prompt mismatch")
        if rec.get("zero_mask_newline") is True:
            raise ValueError("A newline was masked by patch-selection path")
        if rec.get("pos_and_cache_verified") is not True:
            raise ValueError("No true position/cache evidence")
        vec=rec["visual_patch_token_counts_per_layer"]
        if len(vec)!=schedule.layers:
            raise ValueError("Invalid decoder layer count")
        if mode=="paper":
            expected=([schedule.k0]*schedule.cuts[0]+
                      [schedule.k1]*(schedule.cuts[1]-schedule.cuts[0])+
                      [schedule.k2]*(schedule.layers-schedule.cuts[1]))
            if vec!=expected:
                raise ValueError(f"Layer patch schedule disagrees with {schedule.name}")
        measured=sum(vec)
        if mode=="paper" and measured!=schedule.w:
            raise ValueError("Measured visual patch W differs from exact lock")
        if measured<0 or measured>n*schedule.layers:
            raise ValueError("Invalid observed patch counts")
        out.append({"id":sampleid,"arm":arm["arm_id"],"actual_patch_n":n,
                    "actual_patch_W":measured,"actual_patch_T":measured/schedule.layers,
                    "actual_compression_pct":100*(1-measured/(n*schedule.layers))})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--jsonl",required=True)
    p.add_argument("--mode",choices=["paper","native"],default="paper")
    p.add_argument("--arms",default=str(Path(__file__).resolve().parents[1]/"configs"/"arms.json"))
    a=p.parse_args()
    cfg=json.loads(Path(a.arms).read_text())
    records=[json.loads(x) for x in Path(a.jsonl).read_text().splitlines() if x.strip()]
    if not records:raise SystemExit("FAIL: no real samples")
    out=run_check(cfg,records,mode=a.mode)
    print(json.dumps(out,ensure_ascii=False,indent=2))
    print(f"PASS {len(out)} records, mode={a.mode}; note GPU KV equivalence MUST be real evidence")
if __name__=="__main__":main()
