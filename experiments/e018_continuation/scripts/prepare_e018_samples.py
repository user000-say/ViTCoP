#!/usr/bin/env python3
"""Prepare real multimodal prefixes using the installed LLaVA architecture.

Input JSONL: image_id, question_id, image_path, question, answer_ids.
Optional prompt and question_span_input [start,end) refer to input token IDs.
Auto span is accepted only if prefix tokenization aligns exactly; otherwise fail.
"""
import argparse
import inspect
import json
from pathlib import Path
from e018_tools import export_prepared, load_llava, model_args, read_rows


def main():
    p = argparse.ArgumentParser(__doc__)
    model_args(p)
    p.add_argument("--raw-manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--conversation", default="vicuna_v1")
    args = p.parse_args()
    import torch
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.conversation import conv_templates
    from llava.model.e018_stage2_support import take_stage1
    tok, model, core, processor = load_llava(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(args.raw_manifest).resolve().parent
    with (out / "manifest.jsonl").open("x") as f, torch.no_grad():
        for i, row in enumerate(read_rows(args.raw_manifest)):
            for key in ("image_id", "question_id", "image_path", "question", "answer_ids"):
                if key not in row:
                    raise ValueError("Raw manifest missing " + key)
            question = row["question"]
            image_marker = DEFAULT_IMAGE_TOKEN
            if getattr(model.config, "mm_use_im_start_end", False):
                image_marker = DEFAULT_IM_START_TOKEN + image_marker + DEFAULT_IM_END_TOKEN
            if "prompt" in row:
                prompt = row["prompt"]
            else:
                conv = conv_templates[args.conversation].copy()
                conv.append_message(conv.roles[0], image_marker + "\n" + question)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
            where = torch.where(ids == IMAGE_TOKEN_INDEX)[0]
            if where.numel() != 1:
                raise ValueError("Exactly one image placeholder required")
            image_index = int(where[0].item())
            if "question_span_input" in row:
                a, b = map(int, row["question_span_input"])
            else:
                if prompt.count(question) != 1:
                    raise ValueError("Question is not unique; provide explicit question_span_input")
                start = prompt.index(question)
                prefix_a = tokenizer_image_token(prompt[:start], tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
                prefix_b = tokenizer_image_token(prompt[:start + len(question)], tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
                a, b = len(prefix_a), len(prefix_b)
                if not torch.equal(ids[:a], prefix_a) or not torch.equal(ids[:b], prefix_b):
                    raise ValueError("BPE boundary is not aligned: provide verified question_span_input from worker")
            if not image_index < a < b <= len(ids):
                raise ValueError("User question span must follow image")
            image_path = Path(row["image_path"])
            image_path = image_path if image_path.is_absolute() else root / image_path
            with Image.open(image_path) as im:
                image = process_images([im.convert("RGB")], processor, model.config)
            if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1:
                raise ValueError("Only one fixed-resolution image tensor supported; no anyres/multicrop")
            fn = model.prepare_inputs_labels_for_multimodal
            params = inspect.signature(fn).parameters
            values = dict(input_ids=ids[None].to(core.embed_tokens.weight.device), position_ids=None,
                          attention_mask=torch.ones(1, len(ids), device=core.embed_tokens.weight.device, dtype=torch.long),
                          past_key_values=None, labels=None,
                          images=image.to(device=core.embed_tokens.weight.device, dtype=core.embed_tokens.weight.dtype),
                          image_sizes=[(im.width, im.height)])
            required = [k for k, v in params.items() if v.default is inspect.Parameter.empty and k not in values
                        and v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
            if required:
                raise ValueError("Unknown architecture preparation args: " + str(required) + "; use export_prepared() in your worker")
            result = fn(**{k: v for k, v in values.items() if k in params})
            tensors = [x for x in result if torch.is_tensor(x) and x.ndim == 3
                       and x.shape[0] == 1 and x.shape[-1] == core.config.hidden_size]
            if len(tensors) != 1:
                raise ValueError("Cannot identify a unique expanded inputs_embeds; use existing worker exporter")
            embeds = tensors[0]
            meta = take_stage1()
            n = meta["patch_ids"].numel()
            if embeds.shape[1] != len(ids) - 1 + n:
                raise ValueError("Unexpected expansion/padding; original-worker export required")
            path = out / ("prefix_%06d.pt" % i)
            export_prepared(path, embeds, image_index, (a + n - 1, b + n - 1), meta)
            f.write(json.dumps(dict(image_id=row["image_id"], question_id=row["question_id"],
                                    prepared_path=path.name, answer_ids=row["answer_ids"])) + "\n")
            f.flush()
            print(json.dumps(dict(index=i, visual_count=int(n), question_span=[a+n-1, b+n-1])), flush=True)


if __name__ == "__main__":
    main()
