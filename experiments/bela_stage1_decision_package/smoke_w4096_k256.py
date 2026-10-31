#!/usr/bin/env python3
"""Minimal GPU smoke: Dual COVER K256 + QA B8 + Stage2 events for W4096 schedule A.

Checks: entry keep==256, unique ascending patch IDs, E018 events 11:74,19:36,
one real generate path with question bind. Does not claim ACC.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
DEC = REPO / "experiments/bela_stage1_decision_package"
QA = REPO / "experiments/bela_query_anchor"
R3 = REPO / "experiments/elva_r3"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(QA))
sys.path.insert(0, str(R3))
sys.path.insert(0, str(REPO / "LLaVA"))

from launch_qacover_pilot import PY, env_for  # noqa: E402

CFG = DEC / "configs/S1_QANCHOR_W4096.json"
OUT = DEC / "smoke_w4096"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"


def main() -> None:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    child = os.environ.copy()
    child.update(env_for(CFG, OUT))
    child["BELA_STAGE1_INIT"] = "cover"
    child["CUDA_VISIBLE_DEVICES"] = str(gpu)
    child["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{QA}:{DEC}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
    )
    child["HF_HUB_OFFLINE"] = "1"
    child["HF_DATASETS_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    child["PYTHONUNBUFFERED"] = "1"
    assert child["ELVA_R3_ENTRY_K"] == "256"
    assert child["E018_EVENTS"] == "11:74,19:36"
    assert child["E020_ENABLE"] == "True"
    (OUT / "smoke_env.json").write_text(
        json.dumps(
            {
                k: child[k]
                for k in sorted(child)
                if k.startswith(("ELVA_", "E0", "BELA_", "E014"))
            },
            indent=2,
        )
        + "\n"
    )
    # Inline one-sample generate via e014 path with limit=1 on gqa if available;
    # fall back to direct model generate using fair-256 first panel row.
    code = r"""
import json, os, sys
from pathlib import Path
import torch
from PIL import Image

REPO = Path("/home/xiangyue/ViTCoP")
sys.path.insert(0, str(REPO / "LLaVA"))
sys.path.insert(0, str(REPO / "experiments/elva_r3"))
import apply_llava  # noqa: F401 — overlay already applied via env
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import llava_stage1

panel = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
row = json.loads(panel.read_text().splitlines()[0])
img_path = row.get("image") or row.get("image_path")
question = row.get("question") or row.get("text") or "Describe the image."
model_path = str(REPO / "experiments/e014_latency_fair_256/models_llava")
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, "llava_llama", device_map="cuda:0"
)
image = Image.open(img_path).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)
if isinstance(image_tensor, list):
    image_tensor = image_tensor[0]
image_tensor = image_tensor.to(device="cuda:0", dtype=torch.float16)
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
# Bind question span for QA-COVER (same contract as e014_boot).
os.environ["BELA_QA_QUESTION_TEXT"] = question
with torch.inference_mode():
    out = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor,
        do_sample=False,
        max_new_tokens=16,
        use_cache=True,
    )
keep = llava_stage1.LAST_ENTRY_KEEP
metric = llava_stage1.LAST_ENTRY_METRIC
assert keep is not None, "LAST_ENTRY_KEEP missing"
kids = keep.detach().cpu().tolist()
assert len(kids) == 256, f"entry K={len(kids)} != 256"
assert len(set(kids)) == 256, "duplicate patch IDs"
assert kids == sorted(kids), "patch IDs not ascending"
text = tokenizer.decode(out[0], skip_special_tokens=True)
report = {
    "entry_k": len(kids),
    "entry_ids_head": kids[:8],
    "entry_ids_tail": kids[-8:],
    "events": os.environ.get("E018_EVENTS"),
    "first_selector": os.environ.get("E020_KIND"),
    "e018_selector": os.environ.get("E018_SELECTOR"),
    "qa_mode": os.environ.get("BELA_QA_ENTRY_MODE"),
    "decoded_head": text[:200],
    "budget": 256 * 12 + 74 * 8 + 36 * 12,
}
Path(os.environ["ELVA_R3_AUDIT_DIR"]).joinpath("smoke_report.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print("SMOKE_PASS", json.dumps(report), flush=True)
"""
    import subprocess

    log = OUT / "smoke.log"
    with log.open("w") as handle:
        rc = subprocess.call(
            [PY, "-c", code],
            env=child,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    print(log.read_text()[-2000:], flush=True)
    if rc != 0 or not (OUT / "smoke_report.json").exists():
        raise SystemExit(f"SMOKE_FAIL rc={rc} see {log}")
    print("SMOKE_OK", OUT / "smoke_report.json", flush=True)


if __name__ == "__main__":
    main()
