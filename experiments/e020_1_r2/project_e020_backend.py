"""Real LLaVA teacher-forced backend for E020.1 R2 replay pairdata (v2 features)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path("/home/xiangyue/ViTCoP")
PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "LLaVA"))

from e006_span import explicit_span  # noqa: E402
from e014_arms import apply_arm_env, _E020_EVENTS  # noqa: E402
from features import token_features  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402


def _arm_name(stage1: str, schedule_name: str) -> str:
    return f"E020_{stage1}_{schedule_name}_REF32"


class ProjectE020Backend:
    kind = "real"
    supports_identity_kv_purge = False

    def __init__(self, config: dict):
        self.config = config
        self.qa_path = Path(config["qa_jsonl"])
        self.model_id = config.get("model", "liuhaotian/llava-v1.5-7b")
        self.device = config.get("device", "cuda:0")
        self.conv_template = config.get("conv_template", "llava_v1")
        self._examples = None
        print(f"[e0201r2_backend] loading {self.model_id} on {self.device}", flush=True)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_id, None, "llava", device_map=self.device
        )
        self.model.eval()
        os.environ.setdefault("E006_ENABLE", "True")

    def examples(self):
        if self._examples is None:
            rows = []
            with self.qa_path.open() as f:
                for line in f:
                    row = json.loads(line)
                    if Path(row["path"]).is_file():
                        rows.append(row)
            self._examples = rows
            print(f"[e0201r2_backend] {len(rows)} calibration examples", flush=True)
        yield from self._examples

    def _prompt_and_ids(self, example, answer_text):
        context = example["question"]
        question = (
            context
            if DEFAULT_IMAGE_TOKEN in context
            else DEFAULT_IMAGE_TOKEN + "\n" + context
        )
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer_text)
        prompt = conv.get_prompt()
        if answer_text is None:
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        return prompt, question, context, input_ids

    @torch.no_grad()
    def run_teacher_forced(
        self,
        example,
        *,
        stage1,
        schedule_name,
        first_keep_patch_ids,
        forced_final_ids,
        capture_prompt_state,
    ):
        if schedule_name not in _E020_EVENTS:
            raise ValueError(f"unknown schedule {schedule_name}")
        apply_arm_env(_arm_name(stage1, schedule_name))
        os.environ["E020_ENABLE"] = "False"
        os.environ["E020_COLLECT"] = "True"
        os.environ["E019_ENABLE"] = "False"
        if first_keep_patch_ids is not None:
            os.environ["E018_FORCE_FIRST_IDS"] = ",".join(
                str(int(x)) for x in first_keep_patch_ids
            )
        else:
            os.environ.pop("E018_FORCE_FIRST_IDS", None)

        image = Image.open(example["path"]).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        answer = str(example["answer"])
        prompt_q, question, context, ids_q = self._prompt_and_ids(example, None)
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        if not answer_ids:
            raise ValueError(f"empty answer tokens for {example.get('image_id')}")

        doc = {"question": example["question"]}
        start, end, *_rest = explicit_span(
            self.tokenizer, prompt_q, ids_q.tolist(), doc, context, IMAGE_TOKEN_INDEX
        )

        captured = {}

        def observer(ctx):
            if "first" in captured:
                return
            captured["first"] = {
                "layer": int(ctx["layer"]),
                "keep": ctx["keep"].detach().cpu(),
                "q_attn": ctx["q_attn"].detach().float().cpu().numpy(),
                "visual": ctx["visual"].detach().float().cpu().numpy()
                if "visual" in ctx
                else None,
                "patch_ids": ctx["patch_ids"].detach().cpu().numpy().astype(np.int64),
            }

        from llava.model import e018_stage2_support as e018mod

        core = self.model.get_model() if hasattr(self.model, "get_model") else self.model.model
        old_init = e018mod.Runtime.__init__

        def wrapped_init(self_rt, model, cfg=None):
            old_init(self_rt, model, cfg)
            self_rt.observer = observer

        e018mod.Runtime.__init__ = wrapped_init
        try:
            if hasattr(core, "_e018_runtime"):
                delattr(core, "_e018_runtime")
            input_ids = ids_q.unsqueeze(0).to(self.device)
            ans = torch.tensor([answer_ids], device=self.device, dtype=input_ids.dtype)
            out = self.model(
                input_ids=input_ids,
                images=image_tensor.unsqueeze(0)
                if image_tensor.ndim == 3
                else image_tensor,
                image_sizes=[image.size],
                e006_question_span=(start, end),
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            losses = [
                torch.nn.functional.cross_entropy(out.logits[:, -1, :].float(), ans[:, 0])
            ]
            for i in range(1, ans.shape[1]):
                step = self.model(
                    input_ids=ans[:, i - 1 : i],
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                    e006_question_span=(start, end),
                )
                past = step.past_key_values
                losses.append(
                    torch.nn.functional.cross_entropy(
                        step.logits[:, -1, :].float(), ans[:, i]
                    )
                )
            answer_nll = float(torch.stack(losses).mean().item())
            ids_np = np.asarray(answer_ids, dtype="<i8")
            tokens_hash = hashlib.sha256(ids_np.tobytes()).hexdigest()
        finally:
            e018mod.Runtime.__init__ = old_init
            os.environ["E020_COLLECT"] = "False"
            os.environ.pop("E018_FORCE_FIRST_IDS", None)

        if "first" not in captured:
            raise RuntimeError("first prune observer did not fire")
        snap = captured["first"]
        selected = snap["keep"].numpy().astype(np.int64)
        patch_ids = snap["patch_ids"]
        scores = snap["q_attn"]
        if snap["visual"] is not None:
            vis = snap["visual"]
            g = min(4, vis.shape[0])
            idx = np.linspace(0, vis.shape[0] - 1, g).astype(int)
            response = vis[idx]
        else:
            response = scores[None, :]

        previous = None
        history_ok = False
        rt = getattr(core, "_e018_runtime", None)
        if rt is not None and getattr(rt, "_e020_collect_prev", None) is not None:
            prev_g, prev_ids = rt._e020_collect_prev
            prev_g = prev_g.numpy()
            prev_ids = prev_ids.numpy()
            order = {int(p): i for i, p in enumerate(prev_ids)}
            cols = [order[int(p)] for p in patch_ids]
            previous = prev_g[:, cols]
            history_ok = True

        # R2 current line: do not invent history — pass previous only if real.
        feats = token_features(
            scores, response, previous=previous if history_ok else None
        )
        events = _E020_EVENTS[schedule_name]
        last_layer = int(events.split(",")[-1].split(":")[0])
        first_layer = int(events.split(",")[0].split(":")[0])
        final_k = int(events.split(",")[-1].split(":")[1])
        snapshot = {
            "image_id": str(example["image_id"]),
            "question_id": str(example.get("question_id", example["image_id"])),
            "features": feats.astype(np.float64),
            "scores": scores.astype(np.float64),
            "selected": selected,
            "patch_ids": patch_ids,
            "after": first_layer + 1,
            "depth": int(getattr(core.config, "num_hidden_layers", 32)),
            "next_after": last_layer + 1,
            "final_k": final_k,
            "uses_gold_features": False,
            "feature_origin": "prompt_only_pre_event",
            "feature_availability": {"history": bool(history_ok)},
        }
        prefix_payload = {
            "image_id": snapshot["image_id"],
            "schedule": schedule_name,
            "stage1": stage1,
            "patch_ids": patch_ids.tolist(),
            "selected": selected.tolist(),
            "force": None
            if first_keep_patch_ids is None
            else list(map(int, first_keep_patch_ids)),
        }
        prefix_hash = hashlib.sha256(
            json.dumps(prefix_payload, sort_keys=True).encode()
        ).hexdigest()
        return {
            "snapshot": snapshot,
            "answer_nll": answer_nll,
            "prefix_hash": prefix_hash,
            "cache_policy": "layer_local",
            "answer_tokens_hash": tokens_hash,
            "answer_token_count": len(answer_ids),
        }


def make(config):
    return ProjectE020Backend(config)
