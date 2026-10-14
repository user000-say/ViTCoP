#!/usr/bin/env python3
"""Collect exact common-prefix continuation labels from real prepared images."""
import argparse
import hashlib
import json
from pathlib import Path
from e018_tools import load_llava, model_args, prepared, manifest_paths


def main():
    p = argparse.ArgumentParser(__doc__)
    model_args(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--event", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.shards:
        p.error("Invalid shard index/count")
    import torch
    from llava.model.e018_counterfactual import capture_at, continuation_pair
    from llava.model.e018_contract import SCHEMA, FEATURE_NAMES, group_split
    _, model, core, _ = load_llava(args)
    runtime = core._e018_runtime
    if runtime.cfg.mode != "COLLECT":
        p.error("Use E018_MODE=COLLECT; ONLINE collection of labels is forbidden")
    rows = manifest_paths(args.manifest)
    # Exclusive creation avoids quietly appending duplicate or mixed-protocol labels.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("x") as f, torch.no_grad():
        for row in rows:
            shard = int(hashlib.sha256(str(row["image_id"]).encode()).hexdigest()[:8], 16) % args.shards
            if shard != args.shard_index:
                continue
            events = [args.event] if args.event is not None else [x[0] for x in runtime.cfg.events]
            for event in events:
                batch = prepared(row["prepared_path"], core.embed_tokens.weight.device, core.embed_tokens.weight.dtype)
                snap = capture_at(runtime, batch, event)
                pair = continuation_pair(core, model.lm_head, runtime.cfg, snap, row["answer_ids"])
                state_hash = hashlib.sha256(snap["hidden"].detach().cpu().contiguous().numpy().tobytes())
                state_hash.update(json.dumps(dict(layer=event, positions=snap["state"].positions.cpu().tolist(),
                                                 ids=snap["state"].visual_ids.cpu().tolist(),
                                                 actions=[x["action"] for x in snap["state"].events])).encode())
                record = dict(schema=SCHEMA, image_id=row["image_id"], question_id=row["question_id"],
                              split=group_split(row["image_id"]), event=event, endpoint=snap["endpoint"],
                              state_id=state_hash.hexdigest(),
                              rollout_controller_sha256=(hashlib.sha256(Path(runtime.cfg.controller).read_bytes()).hexdigest()
                                                          if runtime.cfg.controller else None),
                              fingerprint=snap["state"].fingerprint, stage1=snap["state"].metadata["stage1"],
                              feature_names=list(FEATURE_NAMES), features=snap["features"].cpu().tolist(),
                              dropped_ids=snap["drop_ids"].cpu().tolist(), answer_length=len(row["answer_ids"]),
                              prepared_sha256=hashlib.sha256(Path(row["prepared_path"]).read_bytes()).hexdigest(),
                              **pair)
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(json.dumps({k: record[k] for k in ("image_id", "question_id", "event", "utility", "delta_token_work")}), flush=True)


if __name__ == "__main__":
    main()
