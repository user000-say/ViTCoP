"""CLI support. CPU tools avoid importing LLaVA/torch."""
import importlib.util
import json
import os
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parents[1]


def contract():
    path = BUNDLE / "llava/model/e018_contract.py"
    spec = importlib.util.spec_from_file_location("e018_contract_standalone", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]


def load_llava(args):
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.model.e018_stage2_support import Runtime
    name = get_model_name_from_path(args.model_path)
    tok, model, processor, context = load_pretrained_model(
        args.model_path, args.model_base, name, device_map=args.device,
        attn_implementation="sdpa")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    core = model.get_model() if hasattr(model, "get_model") else model.model
    if not getattr(core, "e018_enabled", False):
        raise RuntimeError("E018 not installed/enabled in the actual loaded decoder")
    if not hasattr(core, "_e018_runtime"):
        core._e018_runtime = Runtime(core)
    return tok, model, core, processor


def model_args(parser):
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--device", default="cuda:0")


def prepared(path, device, dtype):
    import torch
    # Only load your own tensor-only prepared files, never arbitrary downloaded pickle.
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("inputs_embeds", "visual_start", "visual_count", "question_span", "metadata"):
        if k not in obj:
            raise ValueError("Prepared tensor file missing " + k)
    from llava.model.e018_stage2_support import _STAGE1
    _STAGE1.set(obj["metadata"])
    embeds = obj["inputs_embeds"].to(device=device, dtype=dtype)
    return dict(inputs_embeds=embeds, use_cache=True, return_dict=True,
                image_shape=int(obj["visual_count"]),
                pre_prompt_length_list=[int(obj["visual_start"])],
                token_length_list=[embeds.shape[1]],
                e006_question_span=tuple(obj["question_span"]))


def export_prepared(path, inputs_embeds, visual_start, question_span, metadata):
    """Call from your existing worker AFTER actual MM expansion, BEFORE decoder.

    Never insert answers/labels into inputs_embeds. This exports a reusable prefix
    and original ordered Stage1 patch IDs; does not reconstruct historical COVER.
    """
    import torch
    obj = dict(inputs_embeds=inputs_embeds.detach().cpu(), visual_start=int(visual_start),
               visual_count=int(metadata["patch_ids"].numel()), question_span=list(question_span),
               metadata={k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in metadata.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def manifest_paths(path):
    base = Path(path).resolve().parent
    rows = read_rows(path)
    seen = set()
    for row in rows:
        for key in ("image_id", "question_id", "prepared_path", "answer_ids"):
            if key not in row:
                raise ValueError("Manifest missing " + key)
        key = (str(row["image_id"]), str(row["question_id"]))
        if key in seen:
            raise ValueError("Duplicate image/question in manifest")
        seen.add(key)
        file = Path(row["prepared_path"])
        row["prepared_path"] = str(file if file.is_absolute() else base / file)
        if not row["answer_ids"] or not all(isinstance(x, int) and x >= 0 for x in row["answer_ids"]):
            raise ValueError("answer_ids must be a nonempty tokenized reference answer")
    return rows
