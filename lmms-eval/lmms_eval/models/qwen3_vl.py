"""Dense Qwen3-VL / Qwen3.5 eval on V100. FP16 + SDPA. No extra package install."""

import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


def process_vision_info(conversations):
    images = []
    for conversation in conversations:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") != "image":
                    continue
                image = item["image"]
                if isinstance(image, Image.Image):
                    images.append(image.convert("RGB"))
                elif isinstance(image, str) and image.startswith("data:image"):
                    raw = base64.b64decode(image.split(",", 1)[1])
                    images.append(Image.open(BytesIO(raw)).convert("RGB"))
                else:
                    raise ValueError(f"unsupported image input: {type(image)}")
    return (images or None), None


class Qwen3Family(lmms):
    model_cls = None

    def __init__(
        self,
        pretrained: str,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        assert self.model_cls is not None
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device
        self._model = self.model_cls.from_pretrained(
            pretrained,
            torch_dtype=torch.float16,
            device_map=self.device_map,
            attn_implementation="sdpa",
        ).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU]
            self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

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
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            until = [item for item in until if item != "\n\n"]
            contexts = list(contexts)
            batched_messages = []
            for i, context in enumerate(contexts):
                context = context.replace("<image>", "")
                contexts[i] = context
                processed_visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, Image.Image):
                        buffer = BytesIO()
                        visual.convert("RGB").save(buffer, format="JPEG")
                        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        processed_visuals.append(
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{encoded}",
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )
                batched_messages.append(
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": processed_visuals + [{"type": "text", "text": context}]},
                    ]
                )
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            max_new_tokens = int(gen_kwargs.get("max_new_tokens", 128))
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=self.use_cache,
            )
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                answers[i] = ans
            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError


@register_model("qwen3_vl")
class Qwen3VL(Qwen3Family):
    from transformers import Qwen3VLForConditionalGeneration

    model_cls = Qwen3VLForConditionalGeneration
