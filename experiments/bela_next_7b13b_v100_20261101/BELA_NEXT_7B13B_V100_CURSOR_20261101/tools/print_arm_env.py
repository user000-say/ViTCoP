"""Print the exact BELA-NeXT proposed arm settings for manual integration.

The flags are V2 HANDOFF conventions, NOT supported by the existing repo
until Cursor adds and verifies a gate. Never interpret printing as an actual run.
"""
import argparse
import json
import shlex
from pathlib import Path
from bela_next_budget.budget import source_schedule

def print_arm(cfg,a,*,qa=None,crop=None):
    run={
      "BELA_NEXT_V2_ENABLE":"1",
      "BELA_NEXT_V2_ARM_ID":a["arm_id"],
      "BELA_NEXT_V2_STAGE_K":",".join(map(str,a["stage_k"])),
      "BELA_NEXT_V2_CUTS_AFTER":",".join(map(str,a["cuts_after_layers"])),
      "BELA_NEXT_V2_STAGE1":"DUAL_COVER",
      "BELA_NEXT_V2_QA_SWAPS":str(a["qa_swaps"] if qa is None else qa),
      "BELA_NEXT_V2_STAGE2":"Q_ATTN_ALL",
      "BELA_NEXT_V2_CROP_QUOTA":crop or "uniform",
      "BELA_NEXT_V2_W_LOCK":str(source_schedule(a).w),
      "BELA_NEXT_V2_PATCH_REF_N":str(a["reference_patch_N"]),
      "BELA_NEXT_V2_EXPECTED_LAYERS":str(a["layers"]),
    }
    return run

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",required=True)
    p.add_argument("--qa-swaps",type=int,choices=[0,8,16,24,32])
    p.add_argument("--crop",choices=["uniform","thumbnail_boost"])
    p.add_argument("--matrix",default=str(Path(__file__).resolve().parents[1]/"configs"/"arms.json"))
    a=p.parse_args()
    cfg=json.loads(Path(a.matrix).read_text(encoding="utf-8"))
    arm=next((r for r in cfg["arm_matrix"] if r["arm_id"]==a.arm),None)
    if not arm:
        raise SystemExit("Unrecognized arm; not launching")
    if a.qa_swaps is not None and (a.qa_swaps not in
                                  cfg["secondary_sweep"]["QA_B_T"+str(arm["budget_T"])]):
        raise SystemExit("QA choice is not in locked secondary sweep for this budget")
    print("# These are PROPOSED flags; the live repo MUST implement the feature gate.")
    print(f"# {arm['model']} T={arm['budget_T']} W={source_schedule(arm).w} "
          f"K={arm['stage_k']} cuts={arm['cuts_after_layers']}")
    for k,v in print_arm(cfg,arm,qa=a.qa_swaps,crop=a.crop).items():
        print("export "+k+"="+shlex.quote(v))
    print("# No GPU or model inference executed by this command.")
if __name__=="__main__":
    main()
