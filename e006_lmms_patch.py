"""Inject the explicit E006 question span into the frozen lmms-eval LLaVA path."""
from __future__ import annotations

import copy
import json
import os
from typing import List

import torch
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.models.llava import Llava
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from lmms_eval.models import llava as llava_mod

from e006_span import explicit_span

_INSTALLED = False
_JOB_AUDIT = {
    "requests_audited": 0,
    "stage1_tokens": None,
    "visual_entering_layer12": None,
    "final_visual_tokens": None,
    "question_span_source": None,
    "selector_prefill_count": None,
    "selector_decode_count": None,
    "num_hidden_layers": None,
    "attention_backend": None,
    "stage3_policy": None,
    "precision": None,
}


def job_audit() -> dict:
    return dict(_JOB_AUDIT)


def _hard(message: str) -> None:
    print("SCIENTIFIC_HARD_FAIL " + message, flush=True)
    raise SystemExit(99)


def _qa_mode() -> str:
    mode = os.environ.get("BELA_QA_ENTRY_MODE", "off").strip().lower()
    return "off" if mode in ("", "none") else mode


_SHUFFLE = {}


def _image_key(doc) -> str:
    for key in ("image_source", "imageId", "image_id"):
        value = doc.get(key) if hasattr(doc, "get") else None
        if value:
            return str(value)
    raise RuntimeError("QA-COVER shuffle requires image_source, imageId, or image_id")


def _sample_id(doc, fallback) -> str:
    for key in ("question_id", "id"):
        value = doc.get(key) if hasattr(doc, "get") else None
        if value is not None and str(value) != "":
            return str(value)
    return str(fallback)


def _freeze_shuffle(task_docs, seed: int):
    from pathlib import Path
    import sys
    root = str(Path("/home/xiangyue/ViTCoP/experiments"))
    if root not in sys.path:
        sys.path.insert(0, root)
    from bela_query_anchor.make_shuffle_map import make_shuffle_map

    names = list(getattr(task_docs, "column_names", []) or [])
    wanted = [c for c in ("question_id", "id", "question", "image_source", "imageId", "image_id") if c in names]
    view = task_docs.select_columns(wanted) if wanted and hasattr(task_docs, "select_columns") else task_docs
    rows = []
    by_sid = {}
    for index in range(len(view)):
        doc = view[index]
        sid = _sample_id(doc, index)
        iid = _image_key(doc)
        question = str(doc.get("question") or "")
        if not question:
            raise RuntimeError(f"QA-COVER shuffle row {sid} has no question text")
        row = {"sample_id": sid, "image_id": iid, "question": question}
        rows.append(row)
        by_sid[sid] = row
    mapping = make_shuffle_map(rows, seed)
    audit = os.environ.get("ELVA_R3_AUDIT_DIR", "")
    if audit:
        path = Path(audit) / "shuffle_map.json"
        if not path.exists():
            path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return mapping, by_sid


def _begin_qa_request(self, task, split, doc_id, contexts, question_input, input_ids_list):
    """Bind the current sample's question ids before vision encode. Batch size is already 1."""
    import sys
    from pathlib import Path
    root = str(Path("/home/xiangyue/ViTCoP/experiments"))
    if root not in sys.path:
        sys.path.insert(0, root)
    from bela_query_anchor.question_protocol import content_mask_from_decodes
    from bela_query_anchor.request_context import begin_question
    from e006_span import _chunk_ids

    if len(doc_id) != 1:
        raise RuntimeError("QA-COVER requires one sample per forward")
    doc = self.task_dict[task][split][doc_id[0]]
    _start, _end, raw_start, raw_end, definition, question = explicit_span(
        self.tokenizer, question_input[0], input_ids_list[0].tolist(),
        doc, contexts[0], IMAGE_TOKEN_INDEX,
    )
    source = os.environ.get("BELA_QA_QUESTION_SOURCE", "actual")
    donor_sid = None
    if source == "frozen_mismatched_query_permutation":
        key = (task, split, int(os.environ.get("BELA_QA_SEED", "17")))
        if key not in _SHUFFLE:
            _SHUFFLE[key] = _freeze_shuffle(
                self.task_dict[task][split], int(os.environ.get("BELA_QA_SEED", "17"))
            )
        mapping, by_sid = _SHUFFLE[key]
        sid = _sample_id(doc, doc_id[0])
        if sid not in mapping:
            raise RuntimeError(f"shuffle map has no sample {sid}")
        donor_sid = mapping[sid]
        if by_sid[donor_sid]["image_id"] == _image_key(doc):
            raise RuntimeError(f"shuffle donor {donor_sid} is the same image as {sid}")
        question = by_sid[donor_sid]["question"]
        definition = "frozen_mismatched_query_permutation"
        ids = torch.tensor(_chunk_ids(self.tokenizer, question), dtype=torch.long)
    else:
        ids = input_ids_list[0][raw_start:raw_end].detach().cpu().long()
    if ids.numel() and torch.any(ids < 0):
        raise RuntimeError("question span still contains an image placeholder")
    mask = None
    if os.environ.get("BELA_QA_CONTENT_MASK", "1") != "0":
        pieces = [self.tokenizer.decode([int(tok)], skip_special_tokens=False) for tok in ids.tolist()]
        mask = torch.tensor(content_mask_from_decodes(pieces), dtype=torch.bool)
    sid = _sample_id(doc, doc_id[0])
    try:
        image_id = _image_key(doc)
    except RuntimeError:
        image_id = ""
    return begin_question(
        ids, mask, question_text=question, definition=definition,
        source=source, sample_id=sid, image_id=image_id, donor_sample_id=donor_sid,
    )


def _audit(raw_model) -> None:
    if os.environ.get("E006_ENABLE", "False") != "True":
        return
    core = raw_model.get_model()
    target_k = int(os.environ["E006_TARGET_K"])
    selector = os.environ["E006_SELECTOR"]
    last = getattr(core, "e006_last", {}) or {}
    selected = last.get("selected_local")
    selected_list = selected.detach().cpu().tolist() if selected is not None else []
    reasons = []
    if int(getattr(core, "e001_last_stage1_visual_tokens", -1)) != 172:
        reasons.append(f"stage1={getattr(core, 'e001_last_stage1_visual_tokens', None)}")
    expected_layers = int(os.environ.get("E006_EXPECT_LAYERS", "40"))
    if int(core.config.num_hidden_layers) != expected_layers:
        reasons.append(f"layers={core.config.num_hidden_layers} expected={expected_layers}")
    if core.config._attn_implementation != "sdpa":
        reasons.append(f"attention={core.config._attn_implementation}")
    if bool(core.config.output_attentions):
        reasons.append("output_attentions")
    if core.stage3_policy != "skip":
        reasons.append(f"stage3={core.stage3_policy}")
    if int(getattr(core, "e001_last_final_visual_tokens", -1)) != target_k:
        reasons.append(f"final={getattr(core, 'e001_last_final_visual_tokens', None)}")
    if last.get("visual_out") != target_k or len(selected_list) != target_k:
        reasons.append(f"visual_out={last.get('visual_out')} n={len(selected_list)}")
    if len(set(selected_list)) != target_k:
        reasons.append("duplicate_or_short_visual_ids")
    if selected_list and (min(selected_list) < 0 or max(selected_list) >= int(last.get("visual_in") or 0)):
        reasons.append("selected_id_outside_visual")
    if int(last.get("prefill_count", core.e006_prefill_count)) != 1:
        reasons.append(f"prefill_count={last.get('prefill_count')}")
    if int(last.get("decode_count", core.e006_decode_count)) != 0:
        reasons.append(f"decode_count={last.get('decode_count')}")
    if selector != "GLOBAL_KNORM" and last.get("span_source") != "explicit":
        reasons.append(f"span_source={last.get('span_source')}")
    if not bool(last.get("sdpa_preserved", getattr(core, "_use_sdpa", False))):
        reasons.append("sdpa_not_preserved")
    visual_in = int(last.get("visual_in") or -1)
    visual_out = int(last.get("visual_out") or -1)
    # G0 prunes before layer 12, so the tokens that enter the layer are visual_out.
    # G1-G4 keep 172 tokens through layer 12 and prune after it.
    if selector == "GLOBAL_KNORM":
        entering = visual_out
        if visual_in != 172:
            reasons.append(f"g0_pre_prune_visual={visual_in}")
        if entering != target_k:
            reasons.append(f"visual_entering_layer12={entering}")
    else:
        entering = visual_in
        if entering != 172:
            reasons.append(f"visual_entering_layer12={entering}")
    if reasons:
        _hard("; ".join(reasons))
    weight = getattr(getattr(core, "embed_tokens", None), "weight", None)
    dtype = "" if weight is None else str(weight.dtype)
    _JOB_AUDIT["requests_audited"] += 1
    _JOB_AUDIT["stage1_tokens"] = 172
    _JOB_AUDIT["visual_entering_layer12"] = entering
    _JOB_AUDIT["final_visual_tokens"] = target_k
    _JOB_AUDIT["question_span_source"] = None if selector == "GLOBAL_KNORM" else last.get("span_source")
    _JOB_AUDIT["selector_prefill_count"] = int(last.get("prefill_count"))
    _JOB_AUDIT["selector_decode_count"] = int(last.get("decode_count"))
    _JOB_AUDIT["num_hidden_layers"] = int(core.config.num_hidden_layers)
    _JOB_AUDIT["attention_backend"] = str(core.config._attn_implementation)
    _JOB_AUDIT["stage3_policy"] = str(core.stage3_policy)
    _JOB_AUDIT["precision"] = "fp16" if "float16" in dtype or "fp16" in dtype else dtype


def _generate_until(self, requests: List[Instance]) -> List[str]:
    res = []

    def _collate(x):
        toks = self.tok_encode(x[0])
        return -len(toks), x[0]

    re_ords = llava_mod.utils.Collator([reg.args for reg in requests], _collate, grouping=True)
    chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
    num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
    pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
    for chunk in chunks:
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        if len(contexts) != 1:
            _hard(f"batch_size must stay 1, got {len(contexts)}")
        task = task[0]
        split = split[0]
        batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
        flattened_visuals = self.flatten(batched_visuals)
        gen_kwargs = all_gen_kwargs[0]
        until = [self.tok_decode(self.eot_token_id)]
        if "until" in gen_kwargs:
            until = gen_kwargs.pop("until")
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected gen_kwargs['until'] to be str or list, got {type(until)}")
        if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
            self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
            llava_mod.eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
        if flattened_visuals:
            image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
            if type(image_tensor) is list:
                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
        else:
            image_tensor = None

        question_input = []
        for visual, context in zip(batched_visuals, contexts):
            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                question = " ".join(image_tokens) + "\n" + context
            else:
                question = context
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            question_input.append(conv.get_prompt())

        gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)

        span_kwargs = {}
        if os.environ.get("E006_ENABLE", "False") == "True":
            doc = self.task_dict[task][split][doc_id[0]]
            try:
                start, end, _rs, _re, definition, _question = explicit_span(
                    self.tokenizer,
                    question_input[0],
                    input_ids_list[0].tolist(),
                    doc,
                    contexts[0],
                    IMAGE_TOKEN_INDEX,
                )
            except RuntimeError as exc:
                _hard(str(exc))
            span_kwargs["e006_question_span"] = (start, end)
            self._e006_span_definition = definition

        if os.environ.get("E007_TOKEN_AUDIT") == "True":
            print(
                "[E007_REQUEST_DOC] "
                + json.dumps({"task": task, "doc_id": int(doc_id[0]), "split": split}, separators=(",", ":")),
                flush=True,
            )
        qa_token = None
        if _qa_mode() != "off":
            qa_token = _begin_qa_request(
                self, task, split, doc_id, contexts, question_input, input_ids_list
            )
        try:
            try:
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    image_sizes=gen_kwargs["image_sizes"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    **span_kwargs,
                )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except SystemExit:
                raise
            except Exception as exc:
                print(f"ENGINEERING_FAIL generate {type(exc).__name__}: {exc}", flush=True)
                raise
        finally:
            if qa_token is not None:
                from bela_query_anchor.request_context import end_question
                end_question(qa_token)
        _audit(self.model)
        res.extend(text_outputs)
        self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), text_outputs)
        pbar.update(1)
        del until
    res = re_ords.get_original(res)
    pbar.close()
    return res


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    Llava.generate_until = _generate_until
    _INSTALLED = True
