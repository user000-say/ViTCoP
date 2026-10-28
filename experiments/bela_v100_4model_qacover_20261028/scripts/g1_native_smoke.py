"""G1 start: native FP16 generate on one non-square image. No pruning."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from PIL import Image

OUT = Path("/home/xiangyue/ViTCoP/experiments/bela_v100_4model_qacover_20261028/audit/FP16_BACKEND_SMOKE.json")


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration

    model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
    image = Image.new("RGB", (480, 320), (180, 40, 40))
    processor = AutoProcessor.from_pretrained(model_id)
    t0 = time.time()
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    load_s = time.time() - t0
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is the dominant color? Answer with one word."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=16,
            temperature=None,
            top_p=None,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[-1]:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    report = {
        "model_id": model_id,
        "dtype": "float16",
        "attention": "sdpa",
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "image_size": [480, 320],
        "load_seconds": round(load_s, 2),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
        "prompt": prompt,
        "new_token_ids": [int(x) for x in new_tokens.tolist()],
        "decoded": text,
        "prune": "none",
        "note": "Native path only. Off-equivalence and QA-COVER are not claimed.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
