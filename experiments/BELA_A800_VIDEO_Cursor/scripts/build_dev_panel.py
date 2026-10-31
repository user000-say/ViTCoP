#!/usr/bin/env python3
"""Freeze a 120-video × 3-QA development panel (40 short/medium/long)."""
from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd

ANN = Path(os.environ.get("BELA_VIDEEMME_ANN", "/data1/xiangyue/hf/ann/videomme/videomme/test-00000-of-00001.parquet"))
OUT = Path(os.environ.get("BELA_DEV_PANEL", "/data1/xiangyue/logs/VIDEO_DEV_PANEL_120.json"))
SEED = 20261031


def main():
    df = pd.read_parquet(ANN)
    # expected columns vary; normalize
    cols = {c.lower(): c for c in df.columns}
    dur_col = cols.get("duration") or cols.get("video_duration") or cols.get("duration_category")
    vid_col = cols.get("video_id") or cols.get("videoID") or cols.get("video") or cols.get("id")
    print("columns", list(df.columns)[:30])
    if dur_col is None:
        # Video-MME often has duration_category or video_id with length buckets in path
        for c in df.columns:
            if "duration" in c.lower() or "length" in c.lower() or "category" in c.lower():
                print("cand", c, df[c].iloc[:3].tolist())
    # try common lmms-eval videomme schema
    if "duration" in df.columns:
        durs = df["duration"].astype(str).str.lower()
    elif "duration_category" in df.columns:
        durs = df["duration_category"].astype(str).str.lower()
    else:
        # fallback: split by row thirds as pseudo short/med/long (documented)
        n = len(df)
        df = df.copy()
        df["_bucket"] = ["short"] * (n // 3) + ["medium"] * (n // 3) + ["long"] * (n - 2 * (n // 3))
        durs = df["_bucket"]
        dur_col = "_bucket"
        print("WARN: no duration column; using row-thirds pseudo buckets")

    def pick(bucket, n=40):
        sub = df[durs.str.contains(bucket[:1]) | (durs == bucket) | durs.str.startswith(bucket[:4])]
        if len(sub) < n:
            sub = df[durs == bucket] if (durs == bucket).any() else df.iloc[0:0]
        if len(sub) < n:
            # looser
            sub = df.sample(n=min(n * 3, len(df)), random_state=SEED)
        return sub.sample(n=min(n, len(sub)), random_state=SEED)

    # map buckets
    buckets = {}
    for name, key in ("short", "short"), ("medium", "medium"), ("long", "long"):
        mask = durs.str.contains(name, na=False)
        if mask.sum() == 0 and name == "medium":
            mask = durs.str.contains("mid", na=False)
        buckets[name] = df[mask]
        print(name, "rows", len(buckets[name]))

    panel_rows = []
    for name in ("short", "medium", "long"):
        sub = buckets[name]
        if len(sub) == 0:
            continue
        take = sub.sample(n=min(40, len(sub)), random_state=SEED + hash(name) % 1000)
        panel_rows.append(take)
    panel = pd.concat(panel_rows, ignore_index=True)
    # store question indices / video ids
    records = []
    for i, row in panel.iterrows():
        rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if k in row.index}
        # json-safe
        clean = {}
        for k, v in rec.items():
            if hasattr(v, "tolist"):
                v = v.tolist()
            if isinstance(v, (bytes, bytearray)):
                continue
            try:
                json.dumps(v)
                clean[k] = v
            except TypeError:
                clean[k] = str(v)
        records.append(clean)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "n_videos_or_rows": len(records),
        "note": "Development panel for R0-R4 only; full Video-MME remains confirmation.",
        "rows": records,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print("wrote", OUT, "n=", len(records))


if __name__ == "__main__":
    main()
