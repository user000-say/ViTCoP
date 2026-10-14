#!/usr/bin/env python3
"""Measure complete GPU model TTFT and fixed-length generation (single GPU).

Input raw JSONL REQUIRES prompt, question_span_input, image_path, question_id,
image_id. It never reads GT. Text/image preprocessing stays outside the timer;
EVERY timed trial recomputes CLIP -> Stage1 -> projector -> E018 prefill.
CUDA wall time includes Python synchronization/reader/controller overhead.
"""
import argparse
import inspect
import json
from pathlib import Path
import time
import numpy as np
from e018_tools import load_llava, model_args, read_rows


def main():
    p = argparse.ArgumentParser(__doc__)
    model_args(p)
    p.add_argument("--raw-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tokens", type=int, default=32)
    args = p.parse_args()
    import torch
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.model.e018_stage2_support import take_stage1
    tok, model, core, processor = load_llava(args)
    r = core._e018_runtime
    if r.cfg.mode not in {"STATIC", "ONLINE", "PROXY"} or r.cfg.trace:
        p.error("Timing requires STATIC/ONLINE/PROXY and E018_TRACE empty")
    if not torch.cuda.is_available() or core.embed_tokens.weight.device.type != "cuda":
        p.error("Real CUDA timing requires one loaded CUDA model")
    if args.tokens < 1 or args.repeats < 1 or args.warmup < 0:
        p.error("Invalid token/repeat/warmup count")
    device, dtype = core.embed_tokens.weight.device, core.embed_tokens.weight.dtype
    rows = read_rows(args.raw_manifest)
    root = Path(args.raw_manifest).resolve().parent
    items = []
    for row in rows:
        for key in ("prompt", "question_span_input", "image_path", "image_id", "question_id"):
            if key not in row:
                p.error("Timing manifest requires " + key)
        ids = tokenizer_image_token(row["prompt"], tok, IMAGE_TOKEN_INDEX, return_tensors="pt").to(device)
        positions = torch.where(ids == IMAGE_TOKEN_INDEX)[0]
        if positions.numel() != 1:
            p.error("Exactly one image required")
        path = Path(row["image_path"])
        with Image.open(path if path.is_absolute() else root / path) as im:
            sizes = [(im.width, im.height)]
            image = process_images([im.convert("RGB")], processor, model.config)
        if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1:
            p.error("Fixed-resolution single image required")
        a, b = map(int, row["question_span_input"])
        v0 = int(positions[0].item())
        if not v0 < a < b <= len(ids):
            p.error("Question span must be verified against raw input IDs")
        items.append((row, ids[None], image.to(device=device, dtype=dtype), sizes, v0, a, b))
    if not items:
        p.error("No timing samples")
    fn = model.prepare_inputs_labels_for_multimodal
    params = inspect.signature(fn).parameters

    def run(item):
        row, ids, image, sizes, v0, a, b = item
        values = dict(input_ids=ids, position_ids=None, attention_mask=torch.ones_like(ids),
                      past_key_values=None, labels=None, images=image, image_sizes=sizes)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = fn(**{k: v for k, v in values.items() if k in params})
        embeddings = [x for x in result if torch.is_tensor(x) and x.ndim == 3 and x.shape[0] == 1
                      and x.shape[-1] == core.config.hidden_size]
        if len(embeddings) != 1:
            raise RuntimeError("Unsupported installed multimodal expansion; time inside actual worker")
        embeddings = embeddings[0]
        meta = take_stage1()
        # Return consumed provenance to the runtime; no cached vision features reused.
        from llava.model.e018_stage2_support import _STAGE1
        _STAGE1.set(meta)
        n = meta["patch_ids"].numel()
        if embeddings.shape[1] != ids.shape[1] - 1 + n:
            raise RuntimeError("Unexpected MM expansion")
        output = r.forward(inputs_embeds=embeddings, image_shape=n,
                           pre_prompt_length_list=[v0], token_length_list=[embeddings.shape[1]],
                           e006_question_span=(a + n - 1, b + n - 1), return_dict=True, use_cache=True)
        token = model.lm_head(output.last_hidden_state[:, -1, :]).float().argmax(-1)
        torch.cuda.synchronize(device)
        first = time.perf_counter()
        cache = output.past_key_values
        for _ in range(args.tokens - 1):
            # Deliberately do NOT break on EOS; fixed T means exactly T generated tokens.
            output = r.forward(input_ids=token[:, None], past_key_values=cache, return_dict=True, use_cache=True)
            token = model.lm_head(output.last_hidden_state[:, -1, :]).float().argmax(-1)
            cache = output.past_key_values
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        return dict(image_id=row["image_id"], question_id=row["question_id"], tokens=args.tokens,
                    ttft_ms=(first - start)*1000, fixed_total_ms=(end-start)*1000,
                    decode_ms_per_token=(end-first)*1000/max(args.tokens-1, 1),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    visual_work=sum(r.state.per_layer_n), fingerprint=r.state.fingerprint,
                    actual_layers=len(core.layers), physical_events=r.state.physical_events)

    samples = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), Path(args.output).open("x") as f:
        for i in range(args.warmup):
            run(items[i % len(items)])
        for repeat in range(args.repeats):
            for item in items:
                result = run(item)
                result["repeat"] = repeat
                samples.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()
    report = dict(samples=len(samples), warmup=args.warmup, fixed_tokens=args.tokens,
                  preprocess_timed=False, vision_recomputed=True,
                  environment=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(device)))
    for key in ("ttft_ms", "fixed_total_ms", "decode_ms_per_token", "peak_allocated_bytes", "visual_work"):
        values = np.array([x[key] for x in samples])
        report[key] = dict(mean=float(values.mean()), p50=float(np.median(values)), p90=float(np.quantile(values, .9)))
    print(json.dumps(report, indent=2))
    Path(args.output + ".summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
