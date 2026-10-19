"""Apply the global pilot limit BEFORE disjoint sharding."""
import argparse
import hashlib
import json
from pathlib import Path


def partition(rows, *, shards, limit, mode="clip_image"):
    if type(shards) is not int or shards < 1 or type(limit) is not int or limit < 1:
        raise ValueError("positive integer global limit and shard count required")
    if mode not in ("clip_image", "vlm_question"):
        raise ValueError("explicit image-only or question evaluation mode required")
    unique = {}
    for row in rows:
        image = row.get("image_key")
        question = row.get("question_key")
        if not isinstance(image, str) or not image or (mode == "vlm_question" and
                (not isinstance(question, str) or not question)):
            raise ValueError("canonical identities required")
        key = image if mode == "clip_image" else (image, question)
        if key in unique and mode == "vlm_question":
            raise ValueError("duplicate VLM example in the input manifest")
        unique.setdefault(key, row)
    keys = sorted(unique, key=lambda key: hashlib.sha256(repr(key).encode()).hexdigest())[:limit]
    groups = [[unique[key] for i, key in enumerate(keys) if i % shards == shard] for shard in range(shards)]
    assert sum(map(len, groups)) == min(limit, len(unique))
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON array of real image/example records")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True, help="global unique input count, not per-worker count")
    parser.add_argument("--mode", choices=("clip_image", "vlm_question"), default="clip_image")
    args = parser.parse_args()
    groups = partition(json.loads(Path(args.manifest).read_text()), shards=args.shards, limit=args.limit, mode=args.mode)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    for index, rows in enumerate(groups):
        (out/f"shard_{index:02d}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"total":sum(map(len,groups)),"per_shard":list(map(len,groups))}))
