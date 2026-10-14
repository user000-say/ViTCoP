#!/usr/bin/env python3
"""Greedy real-model inference on prepared prefixes. Does NOT read answer_ids.

Outputs answers and per-request structural traces. Feed answers to your frozen
official evaluator; this script does not invent MME/POPE/benchmark scores.
"""
import argparse
import json
from pathlib import Path
from e018_tools import load_llava, model_args, prepared, read_rows


def main():
    p = argparse.ArgumentParser(__doc__)
    model_args(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args()
    import torch
    tok, model, core, _ = load_llava(args)
    r = core._e018_runtime
    if r.cfg.mode not in {"STATIC", "ONLINE", "PROXY"}:
        p.error("Inference uses STATIC/ONLINE/PROXY only")
    root = Path(args.manifest).resolve().parent
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("x") as f, torch.no_grad():
        for row in read_rows(args.manifest):
            prefix = Path(row["prepared_path"])
            prefix = prefix if prefix.is_absolute() else root / prefix
            batch = prepared(prefix, core.embed_tokens.weight.device, core.embed_tokens.weight.dtype)
            result = r.forward(**batch)
            cache = result.past_key_values
            prefill_trace = r.last_trace or r.trace(cache if hasattr(cache, "key_cache") else _dynamic(cache))
            ids = []
            for step in range(args.max_new_tokens):
                token = int(model.lm_head(result.last_hidden_state[:, -1, :]).float().argmax(-1).item())
                ids.append(token)
                if token == tok.eos_token_id or step + 1 == args.max_new_tokens:
                    break
                result = r.forward(input_ids=torch.tensor([[token]], device=core.embed_tokens.weight.device),
                                   past_key_values=cache, use_cache=True, return_dict=True)
                cache = result.past_key_values
            record = dict(image_id=row["image_id"], question_id=row["question_id"],
                          text=tok.decode(ids, skip_special_tokens=True).strip(),
                          generated_ids=ids, answer_length=len(ids), trace=prefill_trace)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(json.dumps({k: record[k] for k in ("image_id", "question_id", "answer_length")}), flush=True)


def _dynamic(cache):
    from transformers.cache_utils import DynamicCache
    return DynamicCache.from_legacy_cache(cache)


if __name__ == "__main__":
    main()
