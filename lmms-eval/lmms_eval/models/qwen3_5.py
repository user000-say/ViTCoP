from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen3_vl import Qwen3Family
from transformers import Qwen3_5ForConditionalGeneration


@register_model("qwen3_5")
class Qwen3_5(Qwen3Family):
    """Unpruned Qwen3.5-9B. Stage1-only pruning is not this path."""

    model_cls = Qwen3_5ForConditionalGeneration
