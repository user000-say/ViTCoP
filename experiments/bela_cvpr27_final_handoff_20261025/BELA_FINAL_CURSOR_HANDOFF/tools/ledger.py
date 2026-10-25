"""Fail-closed experimental result ledger. Does not calculate benchmark scores."""
import csv, json
from pathlib import Path

FIELDS=("arm","status","source","W_expected","W_actual","gqa","pope_f1",
        "mme_total","mme_perception","mmbench","sciqa","textvqa","ai2d","mmmu",
        "ttft_ms","fixed32_ms","selector_ms","latency_scope",
        "git_sha","checkpoint_sha","evaluator_sha","dataset_manifest_sha")

def _number(s):
    try: return float(s)
    except (TypeError,ValueError):return None

def validate_row(row):
    if any(x not in row for x in FIELDS):
        raise ValueError("missing ledger column(s)")
    status=row["status"]
    if status not in ("NOT_RUN","REPORTED","PARTIAL","VERIFIED_GPU","BLOCKED"):
        raise ValueError(f"bad status: {status}")
    if status=="VERIFIED_GPU":
        # Require actual GPU provenance and six major scores + measured time,
        # without accepting the historical placeholders as verification.
        for field in ("git_sha","checkpoint_sha","evaluator_sha","dataset_manifest_sha",
                      "gqa","pope_f1","mme_total","mmbench","sciqa","textvqa",
                      "ttft_ms","fixed32_ms","W_actual"):
            if row.get(field,"").strip() in ("","NOT_RUN","N/A"):
                raise ValueError(f"VERIFIED_GPU row missing {field}")
        if row["latency_scope"] not in ("FULL_IMAGE_TO_FIRST_TOKEN","DECODER_ONLY"):
            raise ValueError("timing scope must be explicit")
        if int(row["W_expected"])!=int(row["W_actual"]):
            raise ValueError("observed runtime W differs from intended W")
    for score in ("gqa","pope_f1","mmbench","sciqa","textvqa","ai2d","mmmu"):
        t=row.get(score,"").strip()
        if t not in ("","NOT_RUN","N/A"):
            val=_number(t)
            if val is None or not 0<=val<=100:
                raise ValueError(f"{score} must be 0..100")
    return True

def validate_file(path):
    with Path(path).open(newline="",encoding="utf-8") as f:
        rd=csv.DictReader(f)
        if not rd.fieldnames:raise ValueError("missing CSV header")
        rows=list(rd)
    for row in rows:validate_row(row)
    return rows
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("csv_path",nargs="?",default="templates/RESULTS_MASTER.csv")
    args=p.parse_args()
    rows=validate_file(args.csv_path)
    from collections import Counter
    print(json.dumps({"rows":len(rows),"statuses":dict(Counter(r["status"] for r in rows))},
                     ensure_ascii=False,indent=2))
