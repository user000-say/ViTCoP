"""lmms-eval adapter for official CDPruner LLaVA tree (NeurIPS 2025).

Must run with PYTHONPATH = third_party/CDPruner BEFORE any other llava.
Uses isolated env matching CDPruner deps (e.g. /home/xiangyue/envs/starpro).

Keeps a fixed visual_token_num K for the whole decoder (official API),
NOT BELA layer-budget W=sum K_l.
"""
from __future__ import annotations

from datetime import timedelta
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


@register_model("llava_cdpruner")
class LlavaCDPruner(lmms):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache: bool = True,
        visual_token_num: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        assert int(batch_size) == 1
        self.visual_token_num = int(visual_token_num)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        self.device_map = f"cuda:{accelerator.local_process_index}"

        model_name = model_name or get_model_name_from_path(pretrained)
        if "llava" not in model_name.lower():
            model_name = "llava-v1.5-7b"
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained,
            None,
            model_name,
            device_map=self.device_map,
            visual_token_num=self.visual_token_num,
        )
        self._config = self._model.config
        self.model.eval()
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.batch_size_per_gpu = int(batch_size)
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            self.model = accelerator.prepare(self.model)
        AcceleratorState().deepspeed_plugin is None

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self.accelerator.local_process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task0 = task[0]
            split0 = split[0]
            batched_visuals = [
                doc_to_visual[0](self.task_dict[task0][split0][ids]) for ids in doc_id
            ]
            flat_visuals = self.flatten(batched_visuals)
            gen_kwargs = all_gen_kwargs[0]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            image_tensor = process_images(flat_visuals, self._image_processor, self._config)
            if isinstance(image_tensor, list):
                image_tensor = [
                    _image.to(dtype=torch.float16, device=self.device) for _image in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

            question_input = []
            for visual, context in zip(batched_visuals, contexts):
                if DEFAULT_IMAGE_TOKEN not in context:
                    image_tokens = (
                        [DEFAULT_IMAGE_TOKEN] * len(visual)
                        if isinstance(visual, list)
                        else [DEFAULT_IMAGE_TOKEN]
                    )
                    question = " ".join(image_tokens) + "\n" + context
                else:
                    question = context
                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                question_input.append(conv.get_prompt())

            input_ids_list = [
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                for prompt in question_input
            ]
            pad_token_ids = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=pad_token_ids
            ).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            # Official CDPruner passes cleaned question text into CLIP text tower / DPP.
            texts = []
            for context in contexts:
                q = context.replace("\nAnswer the question using a single word or phrase.", "")
                texts.append(q)
            text_arg = texts[0] if len(texts) == 1 else texts

            cont = self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                image_sizes=[v.size for v in flat_visuals],
                texts=text_arg,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            if isinstance(cont, tuple):
                cont = cont[0]
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), text_outputs)
            pbar.update(1)
        pbar.close()
        return re_ords.get_original(res)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("CDPruner adapter: loglikelihood not used for ACC")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("CDPruner adapter: multi-round not used for ACC")
