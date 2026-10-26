
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import warnings
import json
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig
import os
import gc

import copy
import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput, \
GenerateEncoderDecoderOutput,GenerateBeamDecoderOnlyOutput,GenerateBeamEncoderDecoderOutput
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

import math
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM,LlamaPreTrainedModel,Cache,DynamicCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaSdpaAttention,\
                        LlamaAttention,LlamaFlashAttention2,apply_rotary_pos_emb,repeat_kv,\
                        _prepare_4d_causal_attention_mask,_prepare_4d_causal_attention_mask_for_sdpa,LlamaRMSNorm,\
                        LlamaMLP
import warnings
from transformers.modeling_outputs import BaseModelOutputWithPast,CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]

class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    # Beam methods
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        ## TODO
        ## FASTV
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        _rope_len = getattr(self, "_e018_rope_seq_len", None)
        if _rope_len is None:
            _rope_len = position_ids.max().item() + 1
        cos, sin = self.rotary_emb(value_states, seq_len=int(_rope_len))
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        ## TODO
        ## FASTV
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        _rope_len = getattr(self, "_e018_rope_seq_len", None)
        if _rope_len is None:
            _rope_len = position_ids.max().item() + 1
        cos, sin = self.rotary_emb(value_states, seq_len=int(_rope_len))
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        ## TODO
        ## FASTV
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # Prefer Python rope length from E018 decode to avoid .item() CUDA sync each layer.
        _rope_len = getattr(self, "_e018_rope_seq_len", None)
        if _rope_len is None:
            _rope_len = position_ids.max().item() + 1
        cos, sin = self.rotary_emb(value_states, seq_len=int(_rope_len))
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # E018-BEGIN: independent SDPA reader; never enabled on the legacy path.
        e018_callback = getattr(self, "_e018_callback", None)
        if e018_callback is not None and q_len > 1:
            e018_callback(query_states, key_states, value_states, attention_mask)
        # E018-END

        # E010 analysis-only local fusion influence. The SDPA call below is unchanged.
        if (
            getattr(self, "_e010_collect_local_influence", False)
            and q_len > 1
            and not output_attentions
        ):
            from llava.model.e010_runtime_support import local_zero_contribution_influence
            span = getattr(self, "_e010_question_span", None)
            v0 = int(getattr(self, "_e010_visual_start", -1))
            vn = int(getattr(self, "_e010_visual_count", 0))
            if span is not None and vn > 0 and v0 >= 0 and v0 + vn <= int(key_states.shape[-2]):
                q_idx = torch.arange(int(span[0]), int(span[1]), device=query_states.device)
                q_idx = q_idx[(q_idx >= 0) & (q_idx < q_len)]
                if int(q_idx.numel()) > 0:
                    q_side = query_states.index_select(2, q_idx).float()
                    k_vis = key_states[:, :, v0:v0 + vn, :].float()
                    logits = torch.matmul(q_side, k_vis.transpose(2, 3)) / math.sqrt(self.head_dim)
                    probs = torch.softmax(logits, dim=-1)
                    positions = torch.arange(vn, device=probs.device)
                    self._e010_last_influence = local_zero_contribution_influence(
                        probs, value_states[:, :, v0:v0 + vn, :], positions
                    ).detach()

        # ==================================================================
        # E006-QED-R1 2026-10-07: side question-response statistics.
        #
        # This does NOT request/return the model attention matrix and does NOT
        # change the SDPA main path.  It materializes only
        #   [batch, heads, N_question, prefix_length]
        # for layer 12 during prefill, using the post-RoPE Q/K tensors that the
        # layer already computed.  The normal SDPA call below remains intact.
        # ==================================================================
        if getattr(self, "_e006_enabled", False) and q_len > 1:
            selector = getattr(self, "_e006_selector", "").upper()
            if selector not in {"Q_ATTN", "Q_RESP", "QED", "QED_VISUAL"}:
                raise RuntimeError(f"[E006] invalid SDPA-side selector: {selector}")
            if output_attentions:
                raise RuntimeError("[E006] output_attentions=True is forbidden")

            visual_start = int(getattr(self, "_e006_visual_start"))
            visual_count = int(getattr(self, "_e006_visual_count"))
            visual_end = visual_start + visual_count
            question_indices = getattr(self, "_e006_question_indices").to(query_states.device)
            target_k = int(getattr(self, "_e006_target_k"))
            eps = float(getattr(self, "_e006_eps", 1e-8))
            e008_candidate_mask = getattr(self, "_e008_candidate_mask", None)
            e008_mode = getattr(self, "_e008_mode", None)

            if e008_candidate_mask is not None:
                if e008_candidate_mask.numel() != visual_count:
                    raise RuntimeError(
                        f"[E008] candidate mask length={e008_candidate_mask.numel()} "
                        f"!= visual_count={visual_count}"
                    )
                if getattr(self, "_e008_strict_check", False):
                    candidate_count = int(e008_candidate_mask.sum().item())
                    if candidate_count != 172:
                        raise RuntimeError(
                            f"[E008] candidate count={candidate_count}, expected 172"
                        )
                if target_k > 172:
                    raise RuntimeError(
                        f"[E008] target_k={target_k} > Core172 candidate_count=172"
                    )

            if visual_count <= 0 or visual_end > kv_seq_len:
                raise RuntimeError(
                    f"[E006] bad visual span [{visual_start},{visual_end}) for kv_len={kv_seq_len}"
                )
            if question_indices.numel() == 0:
                raise RuntimeError("[E006] empty question span")
            if int(question_indices.max().item()) >= q_len:
                raise RuntimeError(
                    f"[E006] question index {int(question_indices.max().item())} >= q_len={q_len}"
                )
            if target_k > visual_count:
                raise RuntimeError(f"[E006] target_k={target_k} > visual_count={visual_count}")

            events = None
            if getattr(self, "_e006_profile", False) and query_states.is_cuda:
                events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
                events[0].record()

            # [B,H,Qq,D] x [B,H,D,L] -> [B,H,Qq,L].  Float32 only for
            # this small selector side-path to make entropy/cosine numerically
            # stable; main SDPA dtype/semantics are untouched.
            q_side = query_states.index_select(2, question_indices).float()
            k_side = key_states.float()
            side_logits = torch.matmul(q_side, k_side.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                side_mask = attention_mask.index_select(2, question_indices)[..., :kv_seq_len].float()
                side_logits = side_logits + side_mask
            else:
                # Main SDPA uses is_causal=True when mask is None.  Reproduce
                # that causal legality only for the selected question rows.
                key_pos = torch.arange(kv_seq_len, device=side_logits.device).view(1, 1, 1, -1)
                q_pos = question_indices.view(1, 1, -1, 1)
                side_logits = side_logits.masked_fill(key_pos > q_pos, torch.finfo(side_logits.dtype).min)

            selector_logits = side_logits
            if e008_candidate_mask is not None and e008_mode == "CONTEXT_ONLY":
                # This modifies ONLY the small E006/QED side copy. The real
                # scaled_dot_product_attention below still sees all 192 visual
                # tokens. Therefore transient tokens can still alter the actual
                # transformer representations while being absent from QED's
                # selection statistic.
                selector_logits = side_logits.clone()
                transient_local = torch.where(~e008_candidate_mask)[0]
                transient_abs = transient_local + int(visual_start)
                selector_logits.index_fill_(
                    -1,
                    transient_abs,
                    torch.finfo(selector_logits.dtype).min,
                )

            side_prob = torch.softmax(selector_logits, dim=-1)
            # Average heads first, then isolate the current visual tokens.
            visual_prob = side_prob[..., visual_start:visual_end].mean(dim=1)[0]  # [Qq,Nv]
            g = visual_prob.sum(dim=-1)
            P = visual_prob / (g.unsqueeze(-1) + eps)
            w = g / (g.sum() + eps)
            r = (w.unsqueeze(-1) * P).sum(dim=0)
            r = r / (r.sum() + eps)

            if events is not None:
                events[1].record()

            D_Q = None
            rho = None
            score = r
            if selector == "Q_RESP":
                D_Q = torch.ones((), device=r.device, dtype=r.dtype)
            elif selector in {"QED", "QED_VISUAL"}:
                entropy_count = (
                    172
                    if (
                        e008_candidate_mask is not None
                        and e008_mode == "CONTEXT_ONLY"
                    )
                    else visual_count
                )
                D_Q = -(r * torch.log(r + eps)).sum() / math.log(
                    float(entropy_count)
                )
                D_Q = D_Q.clamp(min=0.0, max=1.0)

            if selector in {"Q_RESP", "QED"}:
                # z_i[t] = sqrt(w_t) P_{t,i}, shape [Nv,Qq].
                z = P.transpose(0, 1) * torch.sqrt(w + eps).unsqueeze(0)
                z_norm = F.normalize(z, p=2, dim=-1, eps=eps)
                sim = torch.matmul(z_norm, z_norm.transpose(0, 1)).clamp(min=0.0, max=1.0)
                higher = r.view(1, -1) > r.view(-1, 1)
                masked = sim.masked_fill(~higher, -1.0)
                rho = masked.max(dim=1).values
                rho = torch.where(rho < 0, torch.zeros_like(rho), rho)
                if selector == "Q_RESP":
                    score = r * (1.0 - rho)
                else:
                    score = r * (1.0 - D_Q * rho)

            if events is not None:
                events[2].record()

            # E008 restricts final-evidence candidacy to Core172 while leaving
            # the REAL layer-12 SDPA untouched. In SURVIVAL_ABLATION, all 192
            # still shape QED response geometry; in CONTEXT_ONLY, transient
            # keys are additionally absent from the side-copy statistic above.
            selection_score = score
            if e008_candidate_mask is not None:
                selection_score = score.masked_fill(
                    ~e008_candidate_mask,
                    torch.finfo(score.dtype).min,
                )
            if getattr(self, "_e010_analysis_full576", False):
                from llava.model.e010_runtime_support import analysis_entropy_from_distribution
                full_h, full_d = analysis_entropy_from_distribution(r.detach())
                self._e010_full_context = {
                    "H_raw": full_h.detach(),
                    "D_Q_analysis": full_d.detach(),
                    "FullContextResponseScore": score.detach(),
                    "candidate_count": int(r.numel()),
                }

            # QED_VISUAL postpones rho/score until the full layer-12 decoder
            # output exists. Q_ATTN/Q_RESP/QED already have final scores.
            if selector != "QED_VISUAL":
                selected_local = torch.topk(
                    selection_score,
                    target_k,
                    largest=True,
                    sorted=False,
                ).indices.sort().values
            else:
                selected_local = None

            if events is not None:
                events[3].record()

            self._e006_last = {
                "selector": selector,
                "visual_start": visual_start,
                "visual_count": visual_count,
                "question_indices": question_indices.detach(),
                "span_source": getattr(self, "_e006_span_source", "unknown"),
                "r": r.detach(),
                "D_Q": None if D_Q is None else D_Q.detach(),
                "rho": None if rho is None else rho.detach(),
                "score": score.detach(),
                "selection_score": selection_score.detach(),
                "e008_candidate_mask": (
                    None
                    if e008_candidate_mask is None
                    else e008_candidate_mask.detach()
                ),
                "e008_mode": e008_mode,
                "selected_local": None if selected_local is None else selected_local.detach(),
                "events": events,
            }
            # E012-SD-P1 2026-10-09: on-device sufficient statistics for a
            # diagnostic-only OFFLINE selector audit. Not used for inference.
            if getattr(self, "_e012_diag", False):
                self._e006_last["P"] = P.detach()
                self._e006_last["w"] = w.detach()

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        ## TODO
        ####################################################################################
        ## VITCOP
        self.use_vitcop = os.environ.get("USE_VITCOP", "False") == "True" # Whether to use multi-level pruning
        self.shallow_pruned_layer = int(os.environ.get("SHALLOW_PRUNED_LAYER", "2"))   # Shallow layer pruning
        self.deep_pruned_layer = int(os.environ.get("DEEP_PRUNED_LAYER", "22")) # Deep layer pruning
        self.vitcop_pruned_ratio = float(os.environ.get("VITCOP_PRUNED_RARIO", "0.25"))   # Overall pruning ratio
        self.vision_prune_ratio = float(os.environ.get('VISION_PRUNE_RARIO', "0.5"))   # Retention ratio for ViT layers

        self.use_t3guide = os.environ.get("USE_T3GUIDE", "True") == "True"   # Enable text-token aware stage-III guidance
        self.t3guide_lambda = float(os.environ.get("T3GUIDE_LAMBDA", "0.8"))   # Weight for text-conditioned score in Stage III
        self.t3guide_temperature = float(os.environ.get("T3GUIDE_TEMPERATURE", "1.0"))   # Temperature for text-token weighting
        self.t3guide_cluster_protect = os.environ.get("T3GUIDE_CLUSTER_PROTECT", "True") == "True"   # Keep stage-II diversity in Stage III

        # ------------------------------------------------------------------
        # Stage-II Summary Reliability Enhancement (SRE)
        # ------------------------------------------------------------------
        # Motivation:
        #   Your Stage-1 already performs global importance/diversity selection.
        #   Stage-II should not repeat global diversity. Instead, it should ask a
        #   cluster-level question: "Is this merged evidence group reliable enough?"
        #   If a visual group is compact, one merged summary is enough. If a group
        #   has high residual evidence that the summary cannot represent, SRE keeps
        #   a few complementary elite tokens from that group.
        #
        # Default behavior is conservative: SRE is a small residual correction on
        # top of the original ViTCoP Stage-II quota, not a full replacement.
        self.use_stage2_sre = os.environ.get("USE_STAGE2_SRE", "False") == "True"
        self.sre_quota_blend = float(os.environ.get("SRE_QUOTA_BLEND", "0.30"))
        self.sre_residual_lambda = float(os.environ.get("SRE_RESIDUAL_LAMBDA", "0.25"))
        self.sre_redundancy_lambda = float(os.environ.get("SRE_REDUNDANCY_LAMBDA", "0.08"))
        self.sre_group_reliability_power = float(os.environ.get("SRE_GROUP_RELIABILITY_POWER", "1.0"))
        self.sre_keep_original_anchor = os.environ.get("SRE_KEEP_ORIGINAL_ANCHOR", "True") == "True"
        self.debug_stage2_sre = os.environ.get("DEBUG_STAGE2_SRE", "False") == "True"

        # ------------------------------------------------------------------
        # Stage-II Lite Router (SRE + coverage + conservative query prior)
        # ------------------------------------------------------------------
        # This is designed for the paper story you want:
        #   1) do not add a heavy deep semantic pruning module;
        #   2) move reliable selection to Stage-II;
        #   3) keep the original ViTCoP cluster structure so latency stays low;
        #   4) add only training-free, vectorized signals: K-norm, residual,
        #      spatial coverage/rarity, and a small query-hidden similarity prior.
        self.use_stage2_lite_router = os.environ.get("USE_STAGE2_LITE_ROUTER", "False") == "True"
        self.stage2_query_lambda = float(os.environ.get("STAGE2_QUERY_LAMBDA", "0.12"))
        self.stage2_coverage_lambda = float(os.environ.get("STAGE2_COVERAGE_LAMBDA", "0.18"))
        self.stage2_rarity_lambda = float(os.environ.get("STAGE2_RARITY_LAMBDA", "0.10"))
        self.stage2_mmr_query_lambda = float(os.environ.get("STAGE2_MMR_QUERY_LAMBDA", "0.10"))
        self.stage2_mmr_coverage_lambda = float(os.environ.get("STAGE2_MMR_COVERAGE_LAMBDA", "0.18"))
        self.use_stage2_weighted_merge = os.environ.get("USE_STAGE2_WEIGHTED_MERGE", "False") == "True"
        self.stage2_merge_temperature = float(os.environ.get("STAGE2_MERGE_TEMPERATURE", "1.0"))
        self.stage2_merge_query_lambda = float(os.environ.get("STAGE2_MERGE_QUERY_LAMBDA", "0.10"))
        self.stage2_merge_residual_lambda = float(os.environ.get("STAGE2_MERGE_RESIDUAL_LAMBDA", "0.15"))

        # Stage-III policy.  For latency-first experiments, use:
        #   STAGE3_POLICY=skip       : delete Stage-III entirely.
        #   STAGE3_POLICY=uniform    : deterministic low-cost tail pruning.
        #   STAGE3_POLICY=original   : original ViTCoP / T3Guide branch.
        self.stage3_policy = os.environ.get("STAGE3_POLICY", "original").lower()
        self.stage3_uniform_protect_clusters = os.environ.get("STAGE3_UNIFORM_PROTECT_CLUSTERS", "True") == "True"

        # E001-S1-A 2026-09-18: strict experiment gate.  Stage1 is the only
        # experimental factor in E001-S1, therefore Stage2/Stage3 custom paths
        # must fail fast instead of silently contaminating S0-S3.
        self.e001_method_id = os.environ.get("E001_METHOD_ID", "UNSET").upper()
        self.e001_stage1_strict = os.environ.get("E001_STAGE1_STRICT", "False") == "True"
        self.e001_config_audit = os.environ.get("E001_CONFIG_AUDIT", "False") == "True"
        self.e001_token_audit = os.environ.get("E001_TOKEN_AUDIT", "False") == "True"

        if self.e001_stage1_strict:
            violations = []
            if self.use_stage2_sre:
                violations.append("USE_STAGE2_SRE=True")
            if self.use_stage2_lite_router:
                violations.append("USE_STAGE2_LITE_ROUTER=True")
            if self.use_stage2_weighted_merge:
                violations.append("USE_STAGE2_WEIGHTED_MERGE=True")
            if self.use_t3guide:
                violations.append("USE_T3GUIDE=True")
            if self.stage3_policy != "original":
                violations.append(f"STAGE3_POLICY={self.stage3_policy}")
            if violations:
                raise RuntimeError(
                    "[E001-S1-A] Strict Stage2/Stage3 isolation failed: "
                    + ", ".join(violations)
                )

        if self.e001_config_audit:
            print(
                f"[E001_CONFIG_LLM] method_id={self.e001_method_id} "
                f"stage2_policy={'sre' if self.use_stage2_sre else 'original_vitcop'} "
                f"stage3_policy={self.stage3_policy} sre_enabled={self.use_stage2_sre} "
                f"t3guide_enabled={self.use_t3guide} lite_router={self.use_stage2_lite_router} "
                f"weighted_merge={self.use_stage2_weighted_merge}"
            )

        if self.use_stage2_sre or self.use_stage2_lite_router:
            print(
                f"[ViTCoP Stage2-SRE/Lite Config] "
                f"sre={self.use_stage2_sre}, lite={self.use_stage2_lite_router}, "
                f"quota_blend={self.sre_quota_blend}, "
                f"residual_lambda={self.sre_residual_lambda}, "
                f"redundancy_lambda={self.sre_redundancy_lambda}, "
                f"query_lambda={self.stage2_query_lambda}, "
                f"coverage_lambda={self.stage2_coverage_lambda}, "
                f"weighted_merge={self.use_stage2_weighted_merge}, "
                f"stage3_policy={self.stage3_policy}"
            )

        # ==================================================================
        # E006-QED-R1 2026-10-07
        # Query-conditioned evidence-role selector at layer 12.
        # IMPORTANT: this branch is calibration/training free.  It preserves
        # the normal SDPA attention path and only performs a small side QK
        # computation for user-question query rows during PREFILL.
        #
        # Supported arms:
        #   GLOBAL_KNORM : existing cheap layer-11 K-norm anchor (G0)
        #   Q_ATTN       : query relevance r_i (G1)
        #   Q_RESP       : response-space redundancy with fixed weight 1 (G2)
        #   QED          : response-space redundancy weighted by D_Q (G3)
        #   QED_VISUAL   : visual-hidden cosine redundancy weighted by D_Q (G4)
        # ==================================================================
        # E014-AEE-BEGIN: read isolated experiment settings before K guards
        # E014: read early because E006/E007/E012 budget gates follow.
        # OFF-by-default, frozen historical runs preserve their own restrictions.
        self.e014_enable = os.environ.get("E014_ENABLE", "False") == "True"
        self.e014_variant = os.environ.get("E014_VARIANT", "AEE").upper()
        self.e014_b = int(os.environ.get("E014_B", "16"))
        # E014-R2-BEGIN: small, explicit experiment-only diagnostic switches
        # Original QED math, SDPA and E012 deferral code are unchanged.
        self.e014_allow_deferred = os.environ.get("E014_ALLOW_DEFERRED", "False") == "True"
        self.e014_allow_qattn = os.environ.get("E014_ALLOW_QATTN", "False") == "True"
        # E014-R2-END
        # E014-AEE-END: independent experiment settings
        self.e006_enable = os.environ.get("E006_ENABLE", "False") == "True"
        self.e006_selector = os.environ.get("E006_SELECTOR", "GLOBAL_KNORM").upper()
        self.e006_layer = int(os.environ.get("E006_LAYER", "12"))
        self.e006_target_k = int(os.environ.get("E006_TARGET_K", "96"))
        self.e006_allow_fallback_64 = os.environ.get("E006_ALLOW_FALLBACK_64", "False") == "True"
        # E016 cascade: "8:96,10:64,16:32,20:16" = after those layers keep K tokens.
        # Ranking is frozen at the first layer via QED scores; later steps take prefixes.
        self.e016_cascade = self._parse_e016_cascade(os.environ.get("E016_CASCADE", ""))
        self.e006_profile = os.environ.get("E006_PROFILE", "False") == "True"
        self.e006_debug = os.environ.get("E006_DEBUG", "False") == "True"
        self.e006_require_explicit_question_span = os.environ.get(
            "E006_REQUIRE_EXPLICIT_QUESTION_SPAN", "False"
        ) == "True"
        # Fallback span trims are only for smoke/debug when the caller cannot
        # provide e006_question_span=(start,end).  Publication-facing runs
        # should use the explicit span path and set REQUIRE_EXPLICIT=True.
        self.e006_question_prefix_tokens = int(os.environ.get("E006_QUESTION_PREFIX_TOKENS", "0"))
        self.e006_question_suffix_tokens = int(os.environ.get("E006_QUESTION_SUFFIX_TOKENS", "0"))
        self.e006_eps = float(os.environ.get("E006_EPS", "1e-8"))

        # ================================================================
        # E007-ELP 2026-10-07: cross-file guards + budget/provenance audit.
        #
        # E007 does NOT change the frozen QED formula. It only:
        #   1) formalizes QED64 as a matched-budget point,
        #   2) asserts the Stage1 candidate-pool size from clip_encoder,
        #   3) optionally emits selected-local indices for provenance.
        # ================================================================
        self.e007_enable = os.environ.get("E007_ENABLE", "False") == "True"
        self.e007_stage1_mode = os.environ.get(
            "E007_STAGE1_MODE", "ORIGINAL_172"
        ).upper()
        self.e007_token_audit = os.environ.get(
            "E007_TOKEN_AUDIT", "False"
        ) == "True"

        # E008-Causal: full-accuracy causal intervention on HR_TOPK_192.
        # The successful QED formula and real SDPA path remain unchanged.
        self.e008_enable = os.environ.get("E008_ENABLE", "False") == "True"
        self.e008_mode = os.environ.get(
            "E008_MODE", "SURVIVAL_ABLATION"
        ).upper()
        self.e008_audit = os.environ.get("E008_AUDIT", "False") == "True"
        self.e008_strict_check = os.environ.get(
            "E008_STRICT_CHECK", "True"
        ) == "True"

        # E009 Context Conservation. Both E009 arms expose exactly 188
        # visual tokens to real LLM layers: 172 QED-eligible Core tokens
        # plus 16 context-only tokens (raw or HREC).
        self.e009_enable = os.environ.get("E009_ENABLE", "False") == "True"
        self.e009_mode = os.environ.get(
            "E009_MODE", "HREC16_CONTEXT"
        ).upper()
        self.e009_audit = os.environ.get(
            "E009_AUDIT", "False"
        ) == "True"
        self.e009_strict_check = os.environ.get(
            "E009_STRICT_CHECK", "True"
        ) == "True"

        self._e007_stage1_expected = {
            "ORIGINAL_172": 172,
            "HR_TOPK_192": 192,
            "HR_COVER_192": 192,
        }

        self.e006_prefill_count = 0
        self.e006_decode_count = 0
        self.e006_last = {}

        if self.e006_enable:
            valid = {"GLOBAL_KNORM", "Q_ATTN", "Q_RESP", "QED", "QED_VISUAL"}
            # E012-SD-P1 2026-10-09: accuracy-only, reproducible final-K random control.
            if os.environ.get("E012_ENABLE", "False") == "True":
                valid.add("RANDOM_FINAL")
            if self.e006_selector not in valid:
                raise ValueError(f"[E006] Unsupported selector: {self.e006_selector}")
            if self.e006_layer != 12:
                raise ValueError(f"[E006] E006_LAYER must remain frozen at 12, got {self.e006_layer}")
            if self.e006_target_k != 96:
                e006_registered_fallback = (
                    self.e006_target_k == 64 and self.e006_allow_fallback_64
                )
                e007_registered_budget_point = (
                    self.e007_enable and self.e006_target_k == 64
                )
                # E014-AEE-BEGIN: E014-only final-K32 registration
                # Only E014 may extend the registered budget to K32.
                e014_registered_budget_point = (
                    self.e014_enable and self.e006_target_k in {64, 32}
                )
                # ELVA exact-W schedule: final K must match ELVA_R3_FINAL_K.
                # Allow 8..64 so W1024/W2048 schedule sweeps (handoff 136-arm) can run;
                # still requires ELVA_R3_ENABLE and exact FINAL_K match (not free-form).
                elva_registered_budget_point = (
                    os.environ.get("ELVA_R3_ENABLE") == "True"
                    and self.e006_target_k == int(os.environ.get("ELVA_R3_FINAL_K", "-1"))
                    and 8 <= self.e006_target_k <= 64
                )
                # E016: final K may be 16 when an explicit decreasing cascade is set.
                e016_registered_budget_point = (
                    bool(self.e016_cascade)
                    and self.e006_target_k == int(self.e016_cascade[-1][1])
                    and self.e006_target_k in {96, 64, 32, 16}
                )
                # E014-AEE-END: E014-only final-K32 registration
                if not (
                    e006_registered_fallback or e007_registered_budget_point
                    or e014_registered_budget_point
                    or e016_registered_budget_point
                    or elva_registered_budget_point
                ):
                    raise ValueError(
                        f"[E006/E007] E006_TARGET_K must be 96, or 64 as either "
                        f"the registered E006 fallback or the E007 matched-budget point. "
                        f"Got {self.e006_target_k}"
                    )
            if self.stage3_policy != "skip":
                raise ValueError(f"[E006] STAGE3_POLICY must be skip, got {self.stage3_policy}")
            if self.use_stage2_sre or self.use_stage2_lite_router or self.use_stage2_weighted_merge:
                raise ValueError("[E006] SRE/Lite/weighted-merge must all be disabled")
            if self.use_t3guide:
                raise ValueError("[E006] USE_T3GUIDE must be False")

        if self.e007_enable:
            if not self.e006_enable:
                raise ValueError("[E007] E007_ENABLE=True requires E006_ENABLE=True")
            # E012-SD-P1 2026-10-09: permit matched Core172 baselines without
            # changing the frozen E007 selector contract when E012 is OFF.
            e012_control = (
                os.environ.get("E012_ENABLE", "False") == "True"
                and self.e006_selector in {"Q_ATTN", "RANDOM_FINAL"}
            )
            if self.e006_selector != "QED" and not e012_control:
                raise ValueError(
                    f"[E007] E007 is frozen to QED except E012 controls, "
                    f"got {self.e006_selector}"
                )
            if self.e007_stage1_mode not in self._e007_stage1_expected:
                raise ValueError(
                    f"[E007] unsupported E007_STAGE1_MODE={self.e007_stage1_mode}"
                )
            # E014-AEE-BEGIN: K32 allowed only with E014_ENABLE; E016 adds final-16 cascade
            allowed_k = {96, 64, 32} if self.e014_enable else {96, 64}
            if self.e016_cascade:
                allowed_k = allowed_k | {16, 32, 64, 96}
            if os.environ.get("ELVA_R3_ENABLE") == "True":
                fk = int(os.environ.get("ELVA_R3_FINAL_K", "-1"))
                if 8 <= fk <= 64:
                    allowed_k = allowed_k | {fk}
            if self.e006_target_k not in allowed_k:
                raise ValueError(
                    f"[E007] target K invalid for this registered experiment: "
                    f"{self.e006_target_k}"
                )
            # E014-AEE-END: preserve legacy E007 K96/K64 gate

        if self.e008_enable:
            if not self.e007_enable:
                raise ValueError("[E008] E008_ENABLE=True requires E007_ENABLE=True")
            if self.e007_stage1_mode != "HR_TOPK_192":
                raise ValueError(
                    "[E008] E007_STAGE1_MODE must be HR_TOPK_192"
                )
            if self.e006_selector != "QED":
                raise ValueError("[E008] E006_SELECTOR must remain QED")
            if os.environ.get("E007_SKIP_UNUSED_VIC", "False") != "True":
                raise ValueError(
                    "[E008] E007_SKIP_UNUSED_VIC=True is required for mask transport"
                )
            if self.e008_mode not in {"SURVIVAL_ABLATION", "CONTEXT_ONLY"}:
                raise ValueError(f"[E008] unsupported E008_MODE={self.e008_mode}")
            if self.e006_target_k not in {96, 64}:
                raise ValueError("[E008] target K must be 96 or 64")

        if self.e009_enable:
            if self.e008_enable:
                raise ValueError("[E009] E008_ENABLE must be False")
            if not self.e007_enable:
                raise ValueError("[E009] E007_ENABLE=True is required")
            if self.e007_stage1_mode != "ORIGINAL_172":
                raise ValueError(
                    "[E009] E007_STAGE1_MODE must be ORIGINAL_172"
                )
            if os.environ.get("E007_SKIP_UNUSED_VIC", "False") != "True":
                raise ValueError("[E009] E007_SKIP_UNUSED_VIC=True is required")
            if self.e009_mode not in {
                "RAW188_CONTEXT",
                "HREC16_CONTEXT",
                "RANDOM188_CONTEXT",
                "NULL188_CONTEXT",
            }:
                raise ValueError(
                    f"[E009] unsupported E009_MODE={self.e009_mode}"
                )
            if self.e006_selector != "QED":
                raise ValueError("[E009] selector must remain QED")
            if self.e006_target_k != 96:
                raise ValueError(
                    "[E009] first prototype is frozen to QED96; "
                    "do not run QED64 before the full-accuracy gate"
                )

        # E010 measures the deployment layer and context lifetime. With
        # E010_ENABLE unset, e010_qed_layer stays on the frozen E006/E009 layer.
        from llava.model.e010_runtime_support import E010Config
        # Literal names are part of the E010 source contract.
        os.environ.get("E010_ENABLE")
        os.environ.get("E010_QED_LAYER")
        os.environ.get("E010_CONTEXT_EXIT_LAYER")
        os.environ.get("E010_KEEP_CONTEXT_AFTER_QED")
        os.environ.get("E010_QED_SIDE_ONLY")
        self.e010 = E010Config.from_env()
        self.e010_qed_layer = int(self.e006_layer)
        if self.e010.enabled:
            if not 0 <= int(self.e010.qed_layer) < int(self.config.num_hidden_layers):
                raise ValueError(
                    f"[E010] E010_QED_LAYER={self.e010.qed_layer} is outside the model"
                )
            self.e010_qed_layer = int(self.e010.qed_layer)
        self._e010_visual_removed = 0

        # E012-SD-P1 2026-10-09: optional selection/deletion separation.
        # S and D are zero-based decoder layer_idx values; physical removal is
        # POST-layer D. E012 disabled => exact historical E010 execution path.
        self.e012_enable = os.environ.get("E012_ENABLE", "False") == "True"
        self.e012_delete_layer = int(os.environ.get(
            "E012_DELETE_LAYER", str(self.e010_qed_layer)
        ))
        self.e012_diag = os.environ.get("E012_DIAG", "False") == "True"
        # E012-SD-P1 2026-10-09: optional decode1/decode2 and prefill
        # layer-KV topology audit. DISABLE during every wall-clock run.
        self.e012_kv_audit = os.environ.get("E012_KV_AUDIT", "False") == "True"
        self.e012_decode_visit = 0
        self.e012_prefill_len = None
        self.e012_random_seed = int(os.environ.get("E012_RANDOM_SEED", "20261009"))
        self.e012_last = {}
        # E013-P1 2026-10-09: only extend Stage1 metadata acceptance.
        self.e013_enable = os.environ.get('E013_ENABLE', 'False') == 'True'
        self.e013_variant = os.environ.get('E013_VARIANT', 'EXTRA').upper()
        self.e013_b = int(os.environ.get('E013_B', '16'))
        self.e013_strict_check = os.environ.get('E013_STRICT', 'False') == 'True'
        self.e013_has_context = (
            self.e013_enable and self.e013_variant == 'EXTRA' and self.e013_b > 0
        )
        if self.e013_enable:
            if not self.e012_enable:
                raise ValueError('[E013] requires E012_ENABLE=True')
            if self.e008_enable or self.e009_enable:
                raise ValueError('[E013] E008/E009 must be off')
            if self.e013_variant not in {'EXTRA', 'REPLACE'} or not 0 <= self.e013_b <= 32:
                raise ValueError('[E013] invalid variant/budget')
            if self.e006_selector != 'QED':
                raise ValueError('[E013] freeze E006_SELECTOR=QED for causal Stage1 study')
            if self.e012_delete_layer != self.e010_qed_layer:
                raise ValueError('[E013] freeze S=D; do not confound Stage1 and S/D')
            if self.e010_qed_layer not in {8, 12}:
                raise ValueError('[E013] S=D must be layer8 (primary) or 12 (E008 bridge)')
        # E014-AEE-BEGIN: fixed Stage1/Stage2 invariants
        # E014 E013-independent contract: 172 *eligible* visual positions,
        # same QED and S=D as the E013 primary/bridge experiments.
        if self.e014_enable:
            if self.e013_enable or self.e008_enable or self.e009_enable:
                raise ValueError('[E014] cannot combine with E013/E008/E009')
            if not self.e012_enable or not self.e010.enabled:
                raise ValueError('[E014] requires E010 and E012')
            if not self.e007_enable or self.e007_stage1_mode != 'ORIGINAL_172':
                raise ValueError('[E014] E007 ORIGINAL_172 required')
            if os.environ.get('E007_SKIP_UNUSED_VIC', 'False') != 'True':
                raise ValueError('[E014] E007_SKIP_UNUSED_VIC=True required')
            # E014-R2: Q_ATTN is a bounded diagnostic, never the default.
            if self.e006_selector != 'QED' and not (
                self.e014_allow_qattn and self.e006_selector == 'Q_ATTN'
            ):
                raise ValueError('[E014] E006_SELECTOR must be QED; Q_ATTN requires E014_ALLOW_QATTN=True')
            elva_final = (
                os.environ.get("ELVA_R3_ENABLE") == "True"
                and self.e006_target_k == int(os.environ.get("ELVA_R3_FINAL_K", "-1"))
                and 8 <= self.e006_target_k <= 64
            )
            if self.e006_target_k not in {96, 64, 32} and not elva_final and not (
                self.e016_cascade and self.e006_target_k == int(self.e016_cascade[-1][1])
                and self.e006_target_k in {16, 32, 64, 96}
            ):
                raise ValueError('[E014] final K must be 96,64,32 (or E016 cascade final)')
            # Keep live K32 layer sweep {8,10,12,14,16}; do NOT shrink to R2's {8,10,12}.
            # S=D remains frozen for ordinary accuracy; explicit K32 S8D10 diagnostic only.
            S, D = self.e010_qed_layer, self.e012_delete_layer
            if S not in {8, 10, 12, 14, 16} or D not in {8, 10, 12, 14, 16}:
                raise ValueError('[E014] S/D layer must be one of 8,10,12,14,16')
            if D != S and not (
                self.e014_allow_deferred and S == 8 and D == 10
                and self.e006_target_k == 32
                and self.e014_variant in {'BASE', 'FIXED_COVER'}
                and self.e006_selector == 'QED'
            ):
                raise ValueError('[E014] S!=D requires explicit K32 S8D10 diagnostic with BASE/FIXED_COVER')
            if self.e014_variant not in {
                'BASE', 'FIXED_COVER', 'FIXED_COVER_PROXY',
                'FIXED_RISK', 'ADAPTIVE_B_ONLY', 'AEE',
                # E015 regional Stage1 exchange variants
                'GLOBAL_RETAINED_MEAN', 'REGION_MEAN',
                'REGION_DUAL_SALIENT', 'REGION_DUAL_NEAREST',
                'REGION_DUAL_CENTER', 'REGION_JOINT',
                # E015 dynamic COVER / regional AEE
                'E015_PROBE', 'E015_COVER_FIXED', 'E015_COVER_DYNAMIC',
                'E015_AEE_FIXED', 'E015_AEE_DYNAMIC',
                'E015_GLOBAL_COVER_FIXED', 'E015_GLOBAL_COVER_DYNAMIC',
                'E015_GLOBAL_AEE_FIXED', 'E015_GLOBAL_AEE_DYNAMIC',
                'E015_SCORE_RECON',
                'E016_MARGINAL', 'E016_PROTECTED', 'E016_SPATIAL',
                'E016_HIER', 'E016_CENTERED',
            } or self.e014_b not in {0, 8, 16, 24, 32}:
                raise ValueError('[E014] invalid variant or fixed exchange budget')
            if os.environ.get('E014_AUDIT', 'False') == 'True' and (
                os.environ.get('COMPUTE_EFFICIENCY', 'False') == 'True'
            ):
                raise ValueError('[E014] audit must be OFF in timing')
        # E014-AEE-END: fixed Stage1/Stage2 invariants
        if self.e012_enable:
            errors = []
            if not self.use_vitcop or not self.e006_enable or not self.e010.enabled:
                errors.append("USE_VITCOP, E006_ENABLE, E010_ENABLE must be True")
            if not self.e007_enable or self.e007_stage1_mode != "ORIGINAL_172":
                errors.append("requires E007_ENABLE=True and ORIGINAL_172")
            if os.environ.get("E007_SKIP_UNUSED_VIC", "False") != "True":
                errors.append("requires E007_SKIP_UNUSED_VIC=True")
            if self.e008_enable or self.e009_enable:
                errors.append("E008/E009 context variants are unsupported")
            # E014-AEE-BEGIN: scoped E012 K64/K32 extension; E016 adds cascade finals
            elva_final = (
                os.environ.get("ELVA_R3_ENABLE") == "True"
                and self.e006_target_k == int(os.environ.get("ELVA_R3_FINAL_K", "-1"))
                and 8 <= self.e006_target_k <= 64
            )
            if self.e006_target_k != 96 and not (
                self.e014_enable and self.e006_target_k in {64, 32}
            ) and not elva_final and not (
                self.e016_cascade and self.e006_target_k == int(self.e016_cascade[-1][1])
            ):
                errors.append("E012 frozen K96; only E014 additionally permits K64/K32")
            # E014-AEE-END: scoped E012 K64/K32 extension
            if self.e006_selector not in {"QED", "Q_ATTN", "RANDOM_FINAL"}:
                errors.append("requires QED/Q_ATTN/RANDOM_FINAL selector")
            if bool(self.e010.qed_side_only):
                errors.append("E010_QED_SIDE_ONLY must be False")
            if bool(self.e010.keep_context_after_qed):
                errors.append("E010_KEEP_CONTEXT_AFTER_QED must be False")
            if int(self.e010.context_exit_layer) >= 0:
                errors.append("E010_CONTEXT_EXIT_LAYER must be -1")
            if not self.e010_qed_layer <= self.e012_delete_layer < self.config.num_hidden_layers:
                errors.append("require 0 <= selection layer <= deletion layer < num_layers")
            if self.e016_cascade:
                first_l, first_k = self.e016_cascade[0]
                last_l, last_k = self.e016_cascade[-1]
                if int(self.e010_qed_layer) != int(first_l):
                    errors.append("E016 first cascade layer must equal E010_QED_LAYER")
                if int(self.e012_delete_layer) != int(first_l):
                    errors.append("E016 requires S=D at first cascade layer (no E012 defer)")
                if int(self.e006_target_k) != int(last_k):
                    errors.append("E016 final cascade K must equal E006_TARGET_K")
                if int(last_l) >= int(self.config.num_hidden_layers):
                    errors.append("E016 cascade layer exceeds num_hidden_layers")
                if self.e012_kv_audit or self.e012_diag:
                    errors.append("E016 cascade forbids E012_KV_AUDIT/DIAG")
            if (self.e012_diag or self.e012_kv_audit) and os.environ.get("COMPUTE_EFFICIENCY", "False") == "True":
                errors.append("E012_DIAG/E012_KV_AUDIT may not run with COMPUTE_EFFICIENCY")
            if self.e010.collect_local_influence or self.e010.analysis_full576:
                errors.append("E010 analysis collector flags must be False")
            if errors:
                raise ValueError("[E012-SD-P1] " + "; ".join(errors))

        # E018-BEGIN: isolated decoder execution; existing E014 guards remain intact.
        self.e018_enabled = os.environ.get("E018_ENABLE", "False") == "True"
        if self.e018_enabled:
            if not self.e014_enable:
                raise ValueError("E018 requires the audited E014 Stage1 path")
            if self.e012_diag or self.e012_kv_audit or self.e006_profile:
                raise ValueError("Disable legacy diagnostics/profile; use E018_TRACE for new topology")
            if os.environ.get("COMPUTE_EFFICIENCY", "False") == "True":
                raise ValueError("Legacy FLOPs timer omits E018. Use measure_e018.py.")
        # E018-END

        self.compute_efficiency = os.environ.get("COMPUTE_EFFICIENCY", "False") == "True"   # Whether to compute FLOPs, Prefill Latency, Inference Latency (per token)
        self.prefill_time = 0
        self.all_FLOPs = 0
        self.num_forward = 0
        self.inference_time_per_token = 0
        self.max_gpu_mem = 0
        self.num_token = 0

        # E001-S1-B 2026-09-18: timing-control metadata for reproducible
        # screening/canonical-internal runs. Warmup requests execute normally but
        # are excluded from the existing CUDA-event timing accumulators.
        self.e001_warmup_requests = int(os.environ.get("E001_WARMUP_REQUESTS", "0"))
        self.e001_timing_mode = os.environ.get("E001_TIMING_MODE", "UNSPECIFIED")
        self.e001_request_index = -1
        self.e001_current_request_measured = False
        self.e001_decode_token_index = 0
        self.e001_last_stage1_visual_tokens = None
        self.e001_last_stage2_visual_tokens = None
        self.e001_last_final_visual_tokens = None

        self.stage2_token_sources = None   # Provenance of stage-II visual tokens for cluster protection in Stage III
        self.stage2_visual_token_num = None
        # E006/ViTCoP cache guards.  Original code populates these immediately
        # before the corresponding prune layer; initialize explicitly so a bad
        # execution order fails cleanly instead of raising AttributeError.
        self.k_shallow = None
        self.k_deep = None
        ####################################################################################

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # ======================================================================
    # E006-QED-R1 helpers.  These helpers deliberately live in LlamaModel so
    # the selector can reuse the already-computed layer-12 SDPA Q/K tensors
    # without enabling output_attentions or changing the main attention path.
    # ======================================================================
    def _e006_resolve_question_indices(
        self,
        current_seq_len,
        visual_end,
        token_length_list,
        e006_question_span,
        device,
    ):
        if e006_question_span is not None:
            if isinstance(e006_question_span, torch.Tensor):
                span = e006_question_span.detach().view(-1).tolist()
            else:
                span = list(e006_question_span)
            if len(span) != 2:
                raise ValueError(f"[E006] e006_question_span must contain [start,end), got {span}")
            q_start, q_end = int(span[0]), int(span[1])
            span_source = "explicit"
        else:
            if self.e006_require_explicit_question_span:
                raise RuntimeError(
                    "[E006] Explicit user-question span required. Pass "
                    "e006_question_span=(start,end) from the worker."
                )
            valid_len = int(token_length_list[0]) if len(token_length_list) else int(current_seq_len)
            q_start = int(visual_end) + int(self.e006_question_prefix_tokens)
            q_end = valid_len - int(self.e006_question_suffix_tokens)
            span_source = "post_visual_fallback"

        q_start = max(int(q_start), int(visual_end))
        q_end = min(int(q_end), int(current_seq_len))
        if q_end <= q_start:
            raise RuntimeError(
                f"[E006] Invalid question span [{q_start},{q_end}) for "
                f"visual_end={visual_end}, seq_len={current_seq_len}"
            )
        return torch.arange(q_start, q_end, device=device, dtype=torch.long), span_source

    @staticmethod
    def _e006_positive_redundancy(score_ref, signatures, eps=1e-8):
        # rho_i = max_{j:r_j>r_i} [cos(z_i,z_j)]_+ ; the most relevant
        # token (and any tie without a strictly-higher-r token) gets rho=0.
        z = F.normalize(signatures.float(), p=2, dim=-1, eps=eps)
        sim = torch.matmul(z, z.transpose(0, 1)).clamp(min=0.0, max=1.0)
        higher = score_ref.view(1, -1) > score_ref.view(-1, 1)
        masked = sim.masked_fill(~higher, -1.0)
        rho = masked.max(dim=1).values
        rho = torch.where(rho < 0, torch.zeros_like(rho), rho)
        return rho

    def _e006_configure_layer12_side(
        self, decoder_layer, hidden_states, attention_mask, position_ids,
        visual_start, visual_count, token_length_list, e006_question_span,
        e008_candidate_mask=None, e008_mode=None
    ):
        attn = decoder_layer.self_attn
        if not isinstance(attn, LlamaSdpaAttention):
            raise RuntimeError(
                f"[E006] SDPA must be preserved; got attention class {attn.__class__.__name__}"
            )
        if self.config.output_attentions:
            raise RuntimeError("[E006] output_attentions=True would break the SDPA fast path")

        visual_end = int(visual_start) + int(visual_count)
        q_idx, span_source = self._e006_resolve_question_indices(
            hidden_states.shape[1], visual_end, token_length_list,
            e006_question_span, hidden_states.device
        )
        attn._e006_enabled = True
        # E012-SD-P1 2026-10-09: random final-K control keeps the same side
        # Q/K operator workload as Q_ATTN; ONLY the final survivor IDs change.
        attn._e006_selector = (
            "Q_ATTN" if self.e012_enable and self.e006_selector == "RANDOM_FINAL"
            else self.e006_selector
        )
        attn._e012_diag = bool(self.e012_diag)
        attn._e006_target_k = self.e006_target_k
        attn._e006_visual_start = int(visual_start)
        attn._e006_visual_count = int(visual_count)
        attn._e006_question_indices = q_idx
        attn._e006_profile = bool(self.e006_profile)
        attn._e006_eps = float(self.e006_eps)
        attn._e006_last = None
        attn._e006_span_source = span_source
        attn._e008_candidate_mask = (
            None if e008_candidate_mask is None
            else e008_candidate_mask.to(
                device=hidden_states.device, dtype=torch.bool
            )
        )
        attn._e008_mode = None if e008_mode is None else str(e008_mode)
        # E013-P1: expensive GPU syncs in strict audits must be OFF in timing.
        attn._e008_strict_check = bool(
            self.e013_strict_check if self.e013_has_context else
            (self.e009_strict_check if self.e009_enable else self.e008_strict_check)
        )

    @staticmethod
    def _parse_e016_cascade(raw: str):
        """Parse '8:96,10:64,16:32,20:16' → ((8,96),(10,64),...). Empty → ()."""
        text = (raw or "").strip()
        if not text:
            return ()
        steps = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"[E016] bad cascade entry {part!r}; want LAYER:K")
            layer_s, k_s = part.split(":", 1)
            layer, k = int(layer_s), int(k_s)
            if layer < 0 or k < 1:
                raise ValueError(f"[E016] invalid layer/k in {part!r}")
            steps.append((layer, k))
        if not steps:
            return ()
        layers = [s[0] for s in steps]
        ks = [s[1] for s in steps]
        if layers != sorted(layers) or len(set(layers)) != len(layers):
            raise ValueError("[E016] cascade layers must be strictly increasing")
        if ks != sorted(ks, reverse=True) or len(set(ks)) != len(ks):
            raise ValueError("[E016] cascade Ks must be strictly decreasing")
        return tuple(steps)

    @staticmethod
    def _e006_prune_sequence(hidden_states, position_ids, attention_mask, visual_start, visual_count, selected_local):
        visual_end = int(visual_start) + int(visual_count)
        selected_local = selected_local.to(device=hidden_states.device, dtype=torch.long).unique(sorted=True)
        selected_abs = selected_local + int(visual_start)
        before = torch.arange(0, int(visual_start), device=hidden_states.device, dtype=torch.long)
        after = torch.arange(visual_end, hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
        all_keep = torch.cat([before, selected_abs, after], dim=0)

        new_hidden = hidden_states.index_select(1, all_keep)
        new_position_ids = position_ids.index_select(1, all_keep) if position_ids is not None else None
        new_attention_mask = attention_mask
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                new_attention_mask = attention_mask.index_select(2, all_keep).index_select(3, all_keep)
            elif attention_mask.dim() == 2:
                new_attention_mask = attention_mask.index_select(1, all_keep)
            else:
                raise RuntimeError(f"[E006] Unsupported attention mask rank: {attention_mask.dim()}")
        return new_hidden, new_position_ids, new_attention_mask, all_keep

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cluster_labels = None,
        image_shape = 576,
        token_length_list = [],
        pre_prompt_length_list = [],
        e006_question_span = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # E018-BEGIN: skip all legacy selection, deletion and timing branches.
        if getattr(self, "e018_enabled", False):
            from llava.model.e018_stage2_support import e018_forward
            return e018_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict, cluster_labels=cluster_labels, image_shape=image_shape,
                token_length_list=token_length_list, pre_prompt_length_list=pre_prompt_length_list,
                e006_question_span=e006_question_span)
        # E018-END
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # E012-SD-P1 2026-10-09: record request's *unpruned* prefill
        # length so the true per-layer cache shape can be asserted. We
        # do not manipulate existing caches.
        if self.e012_enable:
            if not use_cache:
                raise RuntimeError("[E012] cache must be enabled in all runs")
            if seq_length > 1:
                self.e012_prefill_len = int(seq_length)
                self.e012_decode_visit = 0
            else:
                self.e012_decode_visit += 1

        # E012-SD-P1 2026-10-09: disallow legacy fallback visual_start=35.
        # The causal intervention must use the exact multimodal prefix span.
        if self.e012_enable and seq_length > 1:
            if batch_size != 1 or len(pre_prompt_length_list) != 1 or len(token_length_list) != 1:
                raise RuntimeError(
                    "[E012] Requires batch=1 and explicit multimodal "
                    "pre_prompt_length_list/token_length_list metadata"
                )
            if e006_question_span is None and self.e006_require_explicit_question_span:
                raise RuntimeError("[E012] Explicit question span missing on prefill")

        if seq_length > 1:
            self.stage2_token_sources = None
            self.stage2_visual_token_num = None
            if self.e006_enable:
                self.e006_prefill_count = 0
                self.e006_decode_count = 0
                self.e006_last = {}
            if self.e016_cascade:
                self._e016_rank_orig = None
                self._e016_alive_orig = None
                self.e016_last = {}

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # E012-SD-P1 2026-10-09: strictly per-forward storage, never carried
        # to decode requests or the next benchmark sample.
        e012_pending_selected = None
        e012_pending_visual_start = None
        e012_pending_visual_count = None
        if self.e012_enable and seq_length > 1:
            self.e012_last = {}

        # E001-S1-B 2026-09-18: separate untimed warmup requests from measured
        # requests without changing model loading or inference semantics.
        e001_is_multimodal = len(pre_prompt_length_list) != 0
        e001_is_prefill = e001_is_multimodal and hidden_states.shape[1] != 1
        e001_is_decode = e001_is_multimodal and hidden_states.shape[1] == 1
        if e001_is_prefill:
            self.e001_request_index += 1
            self.e001_current_request_measured = (
                self.e001_request_index >= self.e001_warmup_requests
            )
            self.e001_decode_token_index = 0
            if self.compute_efficiency and self.e001_request_index == self.e001_warmup_requests:
                torch.cuda.reset_peak_memory_stats()

        e001_efficiency_active = (
            self.compute_efficiency
            and self.e001_current_request_measured
            and (e001_is_prefill or e001_is_decode)
        )

        if e001_efficiency_active:
            total_start_event = torch.cuda.Event(enable_timing=True)
            total_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            total_start_event.record()

        for decoder_layer in self.layers:
            if decoder_layer.self_attn.layer_idx == 0 and seq_length > 1:
                self._e010_visual_removed = 0
            ## Compute FLOPs, Prefill Latency, Inference Latency (per token)
            if e001_efficiency_active and e001_is_prefill:
                n = hidden_states.shape[1]                                      # number of tokens
                d = hidden_states.shape[2]                                      # hidden state size 
                m = self.layers[decoder_layer.self_attn.layer_idx].mlp.up_proj.out_features       # intermediate size of the FFN
                self.all_FLOPs += 4 * n * (d**2) + 2 *(n**2) * d + 3*n*d*m 

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            
            ## TODO
            ########################################################################################################################
            ## VITCOP
            elif self.use_vitcop:
                ## Prune based on spatial similarity clustering from visual priors and attention scores from text semantics

                def token_filter_l2_norm_cluster(cluster_labels, l2_norm, R, visual_start, token_features=None, text_features=None):
                    """
                    ViTCoP Stage-II with optional Summary Reliability Enhancement (SRE).

                    Original ViTCoP Stage-II is kept as the fallback. When SRE is enabled,
                    the method no longer repeats Stage-1 global importance/diversity. It keeps
                    the original cluster-and-merge structure, but changes two local decisions:

                    1) cluster residual routing:
                       baseline quota is still size-based, but a small fraction is reallocated
                       toward clusters whose merged summary is unreliable, measured by residual
                       distance to the cluster centroid and K-norm importance.

                    2) intra-cluster elite choice:
                       elites are not selected by pure K-norm only. They are selected by
                       K-norm importance plus residual complementarity to the group summary,
                       with a mild penalty for choosing duplicate elites.

                    This creates a Stage-II objective different from Stage-1:
                    Stage-1 = global token selection before LLM.
                    Stage-2 = reliability-aware evidence group compression inside LLM.
                    """
                    device = l2_norm.device
                    cluster_mask = cluster_labels >= -1
                    labels = torch.unique(cluster_labels[cluster_mask])
                    K = labels.size(0)

                    selected_abs, merged_groups, merged_positions = [], [], []
                    selected_source_groups, merged_source_groups = [], []

                    if K == 0 or R <= 0:
                        return selected_abs, merged_groups, merged_positions, selected_source_groups, merged_source_groups

                    l2_norm = l2_norm.float()

                    # E001-S1-A 2026-09-18: exact upstream ViTCoP Stage-II
                    # fallback.  Do not route S0-S3 through the modified SRE quota
                    # implementation when SRE/Lite Router are OFF.  The quota,
                    # K-norm elite selection, anchor position and merge groups below
                    # mirror chaser682/ViTCoP main exactly for valid experiment inputs.
                    if (not self.use_stage2_sre) and (not self.use_stage2_lite_router):
                        cluster_l2_sum = torch.stack([
                            l2_norm[cluster_labels == cid].sum() for cid in labels
                        ])
                        sizes = torch.tensor([
                            (cluster_labels == cid).sum() for cid in labels
                        ], device=device)

                        if K > R:
                            topk_indices = torch.topk(
                                cluster_l2_sum, k=R - 1, largest=False
                            ).indices
                            retained_cids = labels[topk_indices]
                            merged_cids = [
                                cid for cid in labels.tolist()
                                if cid not in retained_cids.tolist()
                            ]

                            for cid in retained_cids:
                                abs_indices = (cluster_labels == cid).nonzero(
                                    as_tuple=False).squeeze(1)
                                merged_groups.append(
                                    (abs_indices + visual_start).tolist())
                                sorted_idx = torch.sort(abs_indices).values
                                mid_idx = len(sorted_idx) // 2
                                merged_positions.append(
                                    sorted_idx[mid_idx].item() + visual_start)
                                merged_source_groups.append((int(cid.item()),))

                            merged_indices = torch.cat([
                                (cluster_labels == cid).nonzero(
                                    as_tuple=False).squeeze(1)
                                for cid in merged_cids
                            ])
                            if merged_indices.numel() > 0:
                                merged_groups.append(
                                    (merged_indices + visual_start).tolist())
                                sorted_idx = torch.sort(merged_indices).values
                                mid_idx = len(sorted_idx) // 2
                                merged_positions.append(
                                    sorted_idx[mid_idx].item() + visual_start)
                                merged_source_groups.append(tuple(sorted(
                                    int(cid) for cid in merged_cids)))
                        else:
                            cluster_sizes = torch.stack([
                                (cluster_labels == label).sum() for label in labels
                            ])
                            total_cluster_tokens = cluster_sizes.sum()
                            weights = (cluster_sizes.float() - 1) / (
                                total_cluster_tokens - K)
                            initial_quota = (weights * (R - K)).round().to(torch.long)
                            max_quota = torch.clamp(sizes - 1, min=0)
                            quotas = torch.minimum(initial_quota, max_quota)

                            diff = (R - K) - quotas.sum().item()
                            if diff != 0:
                                sort_idx = torch.argsort(
                                    cluster_l2_sum, descending=(diff < 0))
                                for i in sort_idx:
                                    if diff > 0 and quotas[i] < max_quota[i]:
                                        quotas[i] += 1
                                        diff -= 1
                                    elif diff < 0 and quotas[i] > 0:
                                        quotas[i] -= 1
                                        diff += 1
                                    if diff == 0:
                                        break

                            for i, cid in enumerate(labels):
                                abs_indices = (cluster_labels == cid).nonzero(
                                    as_tuple=False).squeeze(1)
                                size = abs_indices.size(0)
                                keep_count = min(quotas[i].item(), size - 1)
                                if keep_count > 0:
                                    _, topk_idx = torch.topk(
                                        l2_norm[abs_indices],
                                        k=keep_count,
                                        largest=False,
                                    )
                                    top_indices = abs_indices[topk_idx]
                                    selected_abs.extend(
                                        (top_indices + visual_start).tolist())
                                    selected_source_groups.extend(
                                        [(int(cid.item()),)] * int(top_indices.numel()))
                                    keep_mask = torch.ones(
                                        size, dtype=torch.bool, device=device)
                                    keep_mask[topk_idx] = False
                                    merge_indices = abs_indices[keep_mask]
                                else:
                                    merge_indices = abs_indices

                                if merge_indices.numel() > 0:
                                    merged_groups.append(
                                        (merge_indices + visual_start).tolist())
                                    sorted_idx = torch.sort(merge_indices).values
                                    mid_idx = len(sorted_idx) // 2
                                    merged_positions.append(
                                        sorted_idx[mid_idx].item() + visual_start)
                                    merged_source_groups.append((int(cid.item()),))

                        return (
                            selected_abs,
                            merged_groups,
                            merged_positions,
                            selected_source_groups,
                            merged_source_groups,
                        )

                    def _norm01(x, eps=1e-6):
                        x = x.float()
                        if x.numel() == 0:
                            return x
                        x_min = x.min()
                        x_max = x.max()
                        if (x_max - x_min).abs() < eps:
                            return torch.zeros_like(x)
                        return (x - x_min) / (x_max - x_min + eps)

                    def _get_feat_norm():
                        if (not self.use_stage2_sre) or token_features is None:
                            return None
                        if token_features.numel() == 0:
                            return None
                        return F.normalize(token_features.float(), p=2, dim=-1, eps=1e-6)

                    feat_norm = _get_feat_norm()

                    def _token_query_score():
                        if (not self.use_stage2_lite_router) or feat_norm is None or text_features is None:
                            return None
                        if text_features.numel() == 0 or feat_norm.numel() == 0:
                            return None
                        txt = text_features.float()
                        if txt.dim() == 3:
                            txt = txt[0]
                        if txt.numel() == 0:
                            return None
                        txt = F.normalize(txt, p=2, dim=-1, eps=1e-6)
                        # Lightweight query prior: one norm-weighted text summary, not attention maps.
                        tw = torch.softmax(text_features.float().norm(p=2, dim=-1).view(-1), dim=0)
                        text_summary = F.normalize((tw.unsqueeze(-1) * txt).sum(dim=0), p=2, dim=0, eps=1e-6)
                        return torch.matmul(feat_norm, text_summary).float()

                    query_score_all = _token_query_score()

                    def _coverage_score(abs_indices):
                        # Fast spatial coverage proxy: clusters with larger 2-D spread deserve safer routing.
                        if (not self.use_stage2_lite_router) or abs_indices.numel() <= 1:
                            return torch.tensor(0.0, device=device)
                        side = int(round(math.sqrt(float(l2_norm.numel()))))
                        if side <= 1:
                            return torch.tensor(0.0, device=device)
                        y = (abs_indices.float() // side) / max(float(side - 1), 1.0)
                        x = (abs_indices.float() % side) / max(float(side - 1), 1.0)
                        xy = torch.stack([x, y], dim=-1)
                        center = xy.mean(dim=0, keepdim=True)
                        return torch.norm(xy - center, dim=-1).mean()

                    def _rarity_score(abs_indices):
                        # Small clusters are often OCR/small-object/spatial evidence.
                        if (not self.use_stage2_lite_router) or abs_indices.numel() == 0:
                            return torch.tensor(0.0, device=device)
                        return torch.tensor(1.0 / math.sqrt(float(abs_indices.numel()) + 1.0), device=device, dtype=torch.float32)

                    def _cluster_query(abs_indices):
                        if query_score_all is None or abs_indices.numel() == 0:
                            return torch.tensor(0.0, device=device)
                        return query_score_all.index_select(0, abs_indices).mean()

                    def _cluster_indices(cid):
                        return (cluster_labels == cid).nonzero(as_tuple=False).squeeze(1)

                    def _original_mid_position(abs_indices):
                        # Keep the engineered ViTCoP anchor behavior unchanged by default.
                        sorted_idx = torch.sort(abs_indices).values
                        mid_idx = len(sorted_idx) // 2
                        return sorted_idx[mid_idx].item() + visual_start

                    def _importance_rep_position(abs_indices):
                        rep = abs_indices[torch.argmin(l2_norm.index_select(0, abs_indices))]
                        return int(rep.item()) + visual_start

                    def _merged_position(abs_indices):
                        if self.use_stage2_sre and (not self.sre_keep_original_anchor):
                            return _importance_rep_position(abs_indices)
                        return _original_mid_position(abs_indices)

                    def _cluster_residual(abs_indices):
                        """How much information would be lost if this cluster becomes one summary."""
                        if feat_norm is None or abs_indices.numel() <= 1:
                            return torch.tensor(0.0, device=device)
                        z = feat_norm.index_select(0, abs_indices)  # [n, d]
                        center = F.normalize(z.mean(dim=0), p=2, dim=0, eps=1e-6)
                        residual = (1.0 - torch.matmul(z, center)).clamp(min=0.0, max=2.0)
                        return residual.mean()

                    def _cluster_importance(abs_indices):
                        # Smaller K-norm means more important.
                        vals = l2_norm.index_select(0, abs_indices)
                        return -vals.mean()

                    # Original ViTCoP statistics.
                    cluster_l2_sum = torch.stack([l2_norm[cluster_labels == cid].sum() for cid in labels])
                    sizes = torch.tensor([(cluster_labels == cid).sum() for cid in labels], device=device)

                    # SRE statistics. They are used only when SRE is enabled and token features exist.
                    if self.use_stage2_sre and feat_norm is not None:
                        group_indices = [_cluster_indices(cid) for cid in labels]
                        group_residual = torch.stack([_cluster_residual(idx) for idx in group_indices])
                        group_importance = torch.stack([_cluster_importance(idx) for idx in group_indices])
                        group_residual_n = _norm01(group_residual).pow(max(float(self.sre_group_reliability_power), 0.1))
                        group_importance_n = _norm01(group_importance)
                        # A cluster deserves extra elites when it is both important and hard to summarize.
                        sre_group_score = 0.55 * group_importance_n + 0.45 * group_residual_n

                        if self.use_stage2_lite_router:
                            group_query = torch.stack([_cluster_query(idx) for idx in group_indices])
                            group_coverage = torch.stack([_coverage_score(idx) for idx in group_indices])
                            group_rarity = torch.stack([_rarity_score(idx) for idx in group_indices])
                            sre_group_score = (
                                sre_group_score
                                + max(float(self.stage2_query_lambda), 0.0) * _norm01(group_query)
                                + max(float(self.stage2_coverage_lambda), 0.0) * _norm01(group_coverage)
                                + max(float(self.stage2_rarity_lambda), 0.0) * _norm01(group_rarity)
                            )
                    else:
                        group_indices = None
                        group_residual = None
                        sre_group_score = None

                    if K > R:
                        # Original case: more clusters than budget. Conservative modification:
                        # mostly follow K-norm cluster cost, with a small SRE bias toward unreliable clusters.
                        if self.use_stage2_sre and sre_group_score is not None:
                            original_keep_score = _norm01(-cluster_l2_sum)
                            blend = min(max(float(self.sre_quota_blend), 0.0), 1.0)
                            keep_score = (1.0 - blend) * original_keep_score + blend * sre_group_score
                            topk_indices = torch.topk(keep_score, k=R - 1, largest=True).indices
                            retained_cids = labels[topk_indices]
                        else:
                            topk_indices = torch.topk(cluster_l2_sum, k=R - 1, largest=False).indices
                            retained_cids = labels[topk_indices]

                        merged_cids = [cid for cid in labels.tolist() if cid not in retained_cids.tolist()]

                        for cid in retained_cids:
                            abs_indices = _cluster_indices(cid)
                            merged_groups.append((abs_indices + visual_start).tolist())
                            merged_positions.append(_merged_position(abs_indices))
                            merged_source_groups.append((int(cid.item()),))

                        if len(merged_cids) > 0:
                            merged_indices = torch.cat([_cluster_indices(cid) for cid in merged_cids])
                            if merged_indices.numel() > 0:
                                merged_groups.append((merged_indices + visual_start).tolist())
                                merged_positions.append(_merged_position(merged_indices))
                                merged_source_groups.append(tuple(sorted(int(cid) for cid in merged_cids)))

                        if self.debug_stage2_sre:
                            print(f"[Stage2-SRE] K>R: K={int(K)}, R={int(R)}, kept={int(R-1)}")

                        return selected_abs, merged_groups, merged_positions, selected_source_groups, merged_source_groups

                    # Original case: one summary per cluster, then keep R-K elite tokens.
                    cluster_sizes = torch.stack([(cluster_labels == label).sum() for label in labels])
                    total_cluster_tokens = cluster_sizes.sum()
                    denom = torch.clamp(total_cluster_tokens - K, min=1)
                    original_weights = (cluster_sizes.float() - 1) / denom
                    extra_budget = max(0, int(R - K))
                    original_raw_quota = original_weights * extra_budget

                    max_quota = torch.clamp(sizes - 1, min=0)

                    if self.use_stage2_sre and sre_group_score is not None and extra_budget > 0:
                        capacity_mask = (max_quota.float() > 0).float()
                        sre_alloc = sre_group_score * capacity_mask
                        if sre_alloc.sum() <= 1e-6:
                            sre_raw_quota = original_raw_quota
                        else:
                            sre_raw_quota = sre_alloc / torch.clamp(sre_alloc.sum(), min=1e-6) * extra_budget
                        blend = min(max(float(self.sre_quota_blend), 0.0), 1.0)
                        raw_quota = (1.0 - blend) * original_raw_quota + blend * sre_raw_quota
                    else:
                        raw_quota = original_raw_quota

                    quotas = torch.floor(raw_quota).long()
                    quotas = torch.minimum(quotas, max_quota)

                    # Adjust quotas so that their sum is R-K.
                    diff = extra_budget - quotas.sum().item()
                    if diff > 0:
                        frac = raw_quota - torch.floor(raw_quota)
                        order_score = frac
                        if self.use_stage2_sre and sre_group_score is not None:
                            order_score = order_score + 1e-4 * sre_group_score
                        order = torch.argsort(order_score, descending=True)
                        for i in order.tolist():
                            if diff <= 0:
                                break
                            if quotas[i] < max_quota[i]:
                                quotas[i] += 1
                                diff -= 1
                    elif diff < 0:
                        order = torch.argsort(raw_quota, descending=False)
                        for i in order.tolist():
                            if diff >= 0:
                                break
                            if quotas[i] > 0:
                                quotas[i] -= 1
                                diff += 1

                    def _select_elites(abs_indices, keep_count):
                        keep_count = min(int(keep_count), max(0, abs_indices.numel() - 1))
                        if keep_count <= 0:
                            return abs_indices[:0], None
                        vals = l2_norm.index_select(0, abs_indices)

                        if (not self.use_stage2_sre) or feat_norm is None or abs_indices.numel() <= 1:
                            _, topk_idx = torch.topk(vals, k=keep_count, largest=False)
                            return abs_indices[topk_idx], topk_idx

                        z = feat_norm.index_select(0, abs_indices)
                        center = F.normalize(z.mean(dim=0), p=2, dim=0, eps=1e-6)
                        sim_to_center = torch.matmul(z, center)
                        residual = _norm01((1.0 - sim_to_center).clamp(min=0.0, max=2.0))
                        importance = _norm01(-vals)
                        sim = torch.mm(z, z.t()).clamp(min=-1.0, max=1.0)

                        residual_lambda = max(float(self.sre_residual_lambda), 0.0)
                        redundancy_lambda = max(float(self.sre_redundancy_lambda), 0.0)

                        selected_local = []
                        available = torch.ones(abs_indices.numel(), dtype=torch.bool, device=device)

                        # First elite: still strongest importance for safety.
                        seed = int(torch.argmax(importance).item())
                        selected_local.append(seed)
                        available[seed] = False

                        while len(selected_local) < keep_count and available.any():
                            sel_t = torch.tensor(selected_local, device=device, dtype=torch.long)
                            max_sim = sim[:, sel_t].max(dim=1).values if len(selected_local) > 0 else torch.zeros_like(importance)
                            score = importance + residual_lambda * residual - redundancy_lambda * torch.clamp(max_sim, min=0.0)
                            if self.use_stage2_lite_router:
                                if query_score_all is not None:
                                    q_local = _norm01(query_score_all.index_select(0, abs_indices))
                                    score = score + max(float(self.stage2_mmr_query_lambda), 0.0) * q_local
                                # Local spatial coverage: prefer candidates far from already selected local positions.
                                side = int(round(math.sqrt(float(l2_norm.numel()))))
                                if side > 1 and len(selected_local) > 0:
                                    yy = (abs_indices.float() // side) / max(float(side - 1), 1.0)
                                    xx = (abs_indices.float() % side) / max(float(side - 1), 1.0)
                                    xy = torch.stack([xx, yy], dim=-1)
                                    sel_xy = xy.index_select(0, sel_t)
                                    min_dist = torch.cdist(xy, sel_xy).min(dim=1).values
                                    score = score + max(float(self.stage2_mmr_coverage_lambda), 0.0) * _norm01(min_dist)
                            score[~available] = torch.finfo(score.dtype).min
                            best = int(torch.argmax(score).item())
                            if not available[best]:
                                break
                            selected_local.append(best)
                            available[best] = False

                        selected_local_t = torch.tensor(selected_local[:keep_count], device=device, dtype=torch.long)
                        return abs_indices[selected_local_t], selected_local_t

                    # Execute merge and keep logic.
                    for i, cid in enumerate(labels):
                        abs_indices = _cluster_indices(cid)
                        size = abs_indices.size(0)
                        keep_count = min(quotas[i].item(), size - 1)
                        if keep_count > 0:
                            top_indices, topk_idx = _select_elites(abs_indices, keep_count)
                            selected_abs.extend((top_indices + visual_start).tolist())
                            selected_source_groups.extend([(int(cid.item()),)] * int(top_indices.numel()))

                            keep_mask = torch.ones(size, dtype=torch.bool, device=device)
                            if topk_idx is None:
                                # Fallback: map selected ids to local positions.
                                selected_set = set(int(x) for x in top_indices.tolist())
                                for local_pos, global_local_idx in enumerate(abs_indices.tolist()):
                                    if int(global_local_idx) in selected_set:
                                        keep_mask[local_pos] = False
                            else:
                                keep_mask[topk_idx] = False
                            merge_indices = abs_indices[keep_mask]
                        else:
                            merge_indices = abs_indices

                        if merge_indices.numel() > 0:
                            merged_groups.append((merge_indices + visual_start).tolist())
                            merged_positions.append(_merged_position(merge_indices))
                            merged_source_groups.append((int(cid.item()),))

                    if self.debug_stage2_sre:
                        qsum = int(quotas.sum().item()) if 'quotas' in locals() else -1
                        avg_res = float(group_residual.mean().item()) if group_residual is not None else -1.0
                        print(
                            f"[Stage2-SRE] K<=R: K={int(K)}, R={int(R)}, extra={extra_budget}, "
                            f"quota_sum={qsum}, selected={len(selected_abs)}, merged={len(merged_groups)}, "
                            f"avg_residual={avg_res:.4f}"
                        )

                    return selected_abs, merged_groups, merged_positions, selected_source_groups, merged_source_groups

                def normalize_score(score, eps=1e-6):
                    if score.numel() == 0:
                        return score
                    score = score.float()
                    score_min = score.min()
                    score_max = score.max()
                    denom = torch.clamp(score_max - score_min, min=eps)
                    return (score - score_min) / denom

                def compute_text_conditioned_score(hidden_states, visual_start, visual_end, text_start, temperature=1.0, eps=1e-6):
                    num_visual = max(visual_end - visual_start, 0)
                    if num_visual <= 0:
                        return hidden_states.new_zeros((0,), dtype=torch.float32)
                    if text_start >= hidden_states.shape[1]:
                        return hidden_states.new_zeros((num_visual,), dtype=torch.float32)

                    text_hidden = hidden_states[:, text_start:, :].float()
                    visual_hidden = hidden_states[:, visual_start:visual_end, :].float()

                    if text_hidden.shape[1] == 0 or visual_hidden.shape[1] == 0:
                        return hidden_states.new_zeros((num_visual,), dtype=torch.float32)

                    temperature = max(float(temperature), eps)

                    # Dynamic text-token importance prior
                    text_importance = text_hidden.norm(p=2, dim=-1)
                    text_weights = torch.softmax(text_importance / temperature, dim=-1)

                    # Text-guided visual evidence score via text-visual similarity
                    text_hidden = F.normalize(text_hidden, p=2, dim=-1, eps=eps)
                    visual_hidden = F.normalize(visual_hidden, p=2, dim=-1, eps=eps)
                    text_to_visual = torch.matmul(text_hidden, visual_hidden.transpose(1, 2))
                    text_score = (text_weights.unsqueeze(-1) * text_to_visual).sum(dim=1).squeeze(0)

                    return text_score.float()

                def fuse_visual_text_scores(text_score, visual_l2_norm, lambda_text=0.8):
                    lambda_text = min(max(float(lambda_text), 0.0), 1.0)
                    text_score = normalize_score(text_score)
                    visual_score = normalize_score(-visual_l2_norm.float())
                    return lambda_text * text_score + (1.0 - lambda_text) * visual_score

                def select_topk_with_source_protection(score, source_groups, k):
                    device = score.device
                    num_tokens = score.numel()
                    if num_tokens == 0:
                        return torch.empty(0, device=device, dtype=torch.long)

                    k = max(1, min(int(k), num_tokens))
                    if source_groups is None or len(source_groups) != num_tokens:
                        return torch.topk(score, k=k, largest=True).indices.sort().values

                    cluster_best = {}
                    for token_idx, source_group in enumerate(source_groups):
                        if source_group is None:
                            continue
                        for cid in source_group:
                            cid = int(cid)
                            token_score = float(score[token_idx].item())
                            best_entry = cluster_best.get(cid)
                            if best_entry is None or token_score > best_entry[1]:
                                cluster_best[cid] = (token_idx, token_score)

                    if len(cluster_best) == 0:
                        return torch.topk(score, k=k, largest=True).indices.sort().values

                    protected = sorted({entry[0] for entry in cluster_best.values()})
                    protected_tensor = torch.tensor(protected, device=device, dtype=torch.long)

                    if protected_tensor.numel() > k:
                        protected_scores = score.index_select(0, protected_tensor)
                        protected_tensor = protected_tensor[torch.topk(protected_scores, k=k, largest=True).indices]
                        return protected_tensor.sort().values

                    if protected_tensor.numel() == k:
                        return protected_tensor.sort().values

                    all_indices = torch.arange(num_tokens, device=device)
                    keep_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
                    keep_mask[protected_tensor] = False
                    remaining = all_indices[keep_mask]

                    extra_k = min(k - protected_tensor.numel(), remaining.numel())
                    if extra_k > 0:
                        remaining_scores = score.index_select(0, remaining)
                        extra = remaining[torch.topk(remaining_scores, k=extra_k, largest=True).indices]
                    else:
                        extra = remaining[:0]

                    keep = torch.cat([protected_tensor, extra]).unique(sorted=True)
                    if keep.numel() > k:
                        keep_scores = score.index_select(0, keep)
                        keep = keep[torch.topk(keep_scores, k=k, largest=True).indices].sort().values
                    return keep.sort().values

                v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 35 # 35
                text_token_start = v_token_start + image_shape # 611
                text_token_num = max(seq_length - text_token_start, 0)
                v_token_num = image_shape

                e008_candidate_mask = None
                if (self.e008_enable or self.e009_enable or self.e013_has_context) and seq_length > 1:
                    tag = "E013" if self.e013_has_context else ("E009" if self.e009_enable else "E008")
                    if cluster_labels is None:
                        raise RuntimeError(f"[{tag}] cluster_labels metadata is missing")
                    if cluster_labels.dim() != 2 or cluster_labels.shape[0] != 1:
                        raise RuntimeError(
                            f"[{tag}] expected cluster_labels [1,N], got "
                            f"{tuple(cluster_labels.shape)}"
                        )
                    if int(cluster_labels.shape[1]) != int(v_token_num):
                        raise RuntimeError(
                            f"[{tag}] label length={cluster_labels.shape[1]} "
                            f"!= visual_count={v_token_num}"
                        )
                    e008_candidate_mask = (
                        cluster_labels[0].to(hidden_states.device) >= 0
                    )

                    strict = (
                        self.e013_strict_check if self.e013_has_context else
                        (self.e009_strict_check if self.e009_enable else self.e008_strict_check)
                    )
                    if strict and decoder_layer.self_attn.layer_idx == 0:
                        core_count = int(e008_candidate_mask.sum().item())
                        context_count = int((~e008_candidate_mask).sum().item())
                        expected_context = (
                            self.e013_b if self.e013_has_context else
                            (16 if self.e009_enable else 20)
                        )
                        region_mode = self.e009_enable and self.e009_mode == "REGION_TRANSIENT"
                        if core_count != 172:
                            raise RuntimeError(
                                f"[{tag}] Core candidate count={core_count}, expected 172"
                            )
                        if not region_mode and context_count != expected_context:
                            raise RuntimeError(
                                f"[{tag}] context count={context_count}, "
                                f"expected {expected_context}"
                            )
                        if region_mode and context_count < 1:
                            raise RuntimeError("[E010] region transient context is empty")
                        if self.e009_enable and not region_mode and int(v_token_num) != 188:
                            raise RuntimeError(
                                f"[E009] visual_count={v_token_num}, expected 188"
                            )

                # E007 cross-file invariant: clip_encoder and Llama must agree
                # on the candidate-pool size. Check once at layer 0 during prefill.
                if (
                    self.e007_enable
                    and not self.e009_enable
                    and not self.e013_has_context
                    and decoder_layer.self_attn.layer_idx == 0
                    and seq_length > 1
                ):
                    expected_stage1 = self._e007_stage1_expected[self.e007_stage1_mode]
                    if int(v_token_num) != int(expected_stage1):
                        raise RuntimeError(
                            f"[E007] Stage1/Llama mismatch: mode={self.e007_stage1_mode} "
                            f"expected={expected_stage1} actual={v_token_num}"
                        )

                # E001-S1-B 2026-09-18: record actual post-Stage1 visual count
                # without adding tensor work to the pruning algorithm.
                if decoder_layer.self_attn.layer_idx == 0 or not self.e010.enabled:
                    self.e001_last_stage1_visual_tokens = int(v_token_num)
                ## Calculate the number of tokens for shallow and deep pruning based on the overall pruning ratio
                shallow_token_num = round((32 * self.vitcop_pruned_ratio / self.vision_prune_ratio - self.shallow_pruned_layer) / (self.deep_pruned_layer - self.shallow_pruned_layer + (32 - self.deep_pruned_layer) / 5) * v_token_num)
                deep_token_num = round(shallow_token_num / 5)

                # ===============================================================
                # E006-QED-R1 G0: existing cheap Global K-norm anchor.
                # This intentionally keeps the historical semantics: k_shallow is
                # captured from the previous layer (layer 11 when prune layer=12).
                # G1-G4 are NOT pruned here; they run layer 12 with all 172 visual
                # tokens, reuse its post-RoPE Q/K side statistics, then prune after
                # the full decoder block.
                # ===============================================================
                if (
                    self.e006_enable
                    and self.e006_selector == "GLOBAL_KNORM"
                    and decoder_layer.self_attn.layer_idx == self.e006_layer
                    and seq_length > 1
                ):
                    if self.shallow_pruned_layer != self.e006_layer:
                        raise RuntimeError(
                            f"[E006] SHALLOW_PRUNED_LAYER must equal E006_LAYER={self.e006_layer} "
                            f"for GLOBAL_KNORM; got {self.shallow_pruned_layer}"
                        )
                    if getattr(self, "k_shallow", None) is None:
                        raise RuntimeError("[E006] GLOBAL_KNORM requires layer-11 k_shallow cache")
                    visual_start = v_token_start
                    visual_end = visual_start + v_token_num
                    if self.e006_target_k > v_token_num:
                        raise RuntimeError(f"[E006] target_k={self.e006_target_k} > visual_count={v_token_num}")
                    image_k_l2_norm = self.k_shallow.mean(dim=1).norm(p=2, dim=-1)[0][visual_start:visual_end]
                    selected_local = torch.topk(
                        image_k_l2_norm, self.e006_target_k, largest=False
                    ).indices.sort().values
                    hidden_states, position_ids, attention_mask, _ = self._e006_prune_sequence(
                        hidden_states, position_ids, attention_mask, visual_start, v_token_num, selected_local
                    )
                    self.stage2_visual_token_num = int(self.e006_target_k)
                    # Stage3 is forced to TRUE-SKIP in E006, so cluster/source
                    # provenance is intentionally not materialized (avoids an
                    # unnecessary GPU->CPU sync in canonical TTFT runs).
                    self.stage2_token_sources = None
                    self.e001_last_stage2_visual_tokens = int(self.e006_target_k)
                    if self.stage3_policy == "skip":
                        self.e001_last_final_visual_tokens = int(self.e006_target_k)
                    self.e006_prefill_count += 1
                    self.e006_last = {
                        "selector": "GLOBAL_KNORM",
                        "visual_in": int(v_token_num),
                        "visual_out": int(self.e006_target_k),
                        "selected_local": selected_local.detach(),
                        "prefill_count": int(self.e006_prefill_count),
                        "decode_count": int(self.e006_decode_count),
                        "sdpa_preserved": bool(self._use_sdpa),
                    }
                    if self.e006_debug:
                        print(
                            "[E006_G0] "
                            f"visual_in={v_token_num} visual_out={self.e006_target_k} "
                            f"selected={selected_local.detach().cpu().tolist()}"
                        )

                # Original ViTCoP Stage2 remains byte-for-byte active when E006
                # is disabled.  E006 G1-G4 deliberately bypass this pre-layer
                # pruning because their evidence is measured at layer 12 itself.
                # Shallow pruning logic (pruning based on spatial similarity clusters)
                if decoder_layer.self_attn.layer_idx == self.shallow_pruned_layer and seq_length > 1 and not self.e006_enable:
                    device = hidden_states.device
                    visual_start = v_token_start
                    visual_end = v_token_start + v_token_num

                    # Get the L2 norm of k_shallow, and prune based on visual token cluster information and the L2 norm of k_shallow
                    image_k_l2_norm = self.k_shallow.mean(dim=1).norm(p=2, dim=-1)[0][visual_start:visual_end]

                    selected_abs, merged_groups, merged_positions, selected_source_groups, merged_source_groups = token_filter_l2_norm_cluster(
                        cluster_labels[0],
                        image_k_l2_norm,
                        shallow_token_num,
                        visual_start,
                        token_features=hidden_states[0, visual_start:visual_end, :],
                        text_features=hidden_states[:, visual_end:, :]
                    )

                    # Build all position indices
                    non_visual_before = torch.arange(0, visual_start, device=device)
                    non_visual_after = torch.arange(visual_end, hidden_states.shape[1], device=device)

                    selected_abs_tensor = torch.tensor(selected_abs, device=device, dtype=torch.long)
                    merged_positions_tensor = torch.tensor(merged_positions, device=device, dtype=torch.long)

                    # Create merged_tokens.  Default keeps ViTCoP's anchor + mean.
                    # Stage2 weighted merge is a lightweight replacement for plain mean:
                    # it emphasizes K-norm important tokens, residual evidence, and a small
                    # query prior, without computing attention maps or adding parameters.
                    merged_tokens = []
                    for i, group in enumerate(merged_groups):
                        group_tensor = torch.tensor(group, device=device, dtype=torch.long)
                        anchor = hidden_states[:, merged_positions_tensor[i], :]
                        if self.use_stage2_weighted_merge and group_tensor.numel() > 1:
                            local_group = group_tensor - visual_start
                            group_hidden = hidden_states[:, group_tensor, :]
                            imp = normalize_score(-image_k_l2_norm.index_select(0, local_group))

                            z = F.normalize(hidden_states[0, group_tensor, :].float(), p=2, dim=-1, eps=1e-6)
                            center = F.normalize(z.mean(dim=0), p=2, dim=0, eps=1e-6)
                            residual = normalize_score((1.0 - torch.matmul(z, center)).clamp(min=0.0, max=2.0))

                            merge_score = imp + max(float(self.stage2_merge_residual_lambda), 0.0) * residual
                            if hidden_states.shape[1] > visual_end and self.stage2_merge_query_lambda > 0:
                                txt = hidden_states[:, visual_end:, :].float()
                                if txt.shape[1] > 0:
                                    tw = torch.softmax(txt.norm(p=2, dim=-1).view(-1), dim=0)
                                    txtn = F.normalize(txt[0], p=2, dim=-1, eps=1e-6)
                                    text_summary = F.normalize((tw.unsqueeze(-1) * txtn).sum(dim=0), p=2, dim=0, eps=1e-6)
                                    q = normalize_score(torch.matmul(z, text_summary))
                                    merge_score = merge_score + max(float(self.stage2_merge_query_lambda), 0.0) * q

                            temp = max(float(self.stage2_merge_temperature), 1e-6)
                            weight = torch.softmax(merge_score / temp, dim=0).view(1, -1, 1).to(group_hidden.dtype)
                            summary = (weight * group_hidden).sum(dim=1, keepdim=True)
                            merged_token = anchor + summary
                        else:
                            merged_token = anchor + hidden_states[:, group_tensor, :].mean(dim=1, keepdim=True)
                        merged_tokens.append(merged_token)

                    merged_tokens_tensor = torch.cat(merged_tokens, dim=1) if merged_tokens else hidden_states[:, :0, :]
                    selected_tokens_tensor = hidden_states[:, selected_abs_tensor, :] if selected_abs else hidden_states[:, :0, :]

                    # Extract non-visual tokens
                    non_visual_before_tensor = hidden_states[:, non_visual_before, :]
                    non_visual_after_tensor = hidden_states[:, non_visual_after, :]

                    # Concatenate tokens in order: non-visual-before + kept tokens + merged tokens + non-visual-after
                    mid_positions = torch.cat([selected_abs_tensor, merged_positions_tensor])
                    mid_tokens = torch.cat([selected_tokens_tensor, merged_tokens_tensor], dim=1)

                    # Keep provenance of stage-II tokens for cluster-protected Stage III
                    mid_source_groups = selected_source_groups + merged_source_groups

                    # Get sorting indices
                    sorted_mid, indices = torch.sort(mid_positions)
                    mid_tokens_sorted = mid_tokens[:, indices, :]
                    self.stage2_token_sources = [mid_source_groups[i] for i in indices.tolist()]
                    self.stage2_visual_token_num = len(self.stage2_token_sources)
                    # E001-S1-B 2026-09-18: diagnostic Stage2 token count.
                    self.e001_last_stage2_visual_tokens = int(self.stage2_visual_token_num)
                    if self.e001_token_audit:
                        print(
                            f"[E001_STAGE2_TOKENS] method_id={self.e001_method_id} "
                            f"stage2_policy={'sre' if self.use_stage2_sre else 'original_vitcop'} "
                            f"stage2_visual_tokens={self.e001_last_stage2_visual_tokens}"
                        )
                    # Final concatenation
                    hidden_states = torch.cat([non_visual_before_tensor, mid_tokens_sorted, non_visual_after_tensor], dim=1)
                    # Update position_ids
                    position_ids = torch.cat([non_visual_before, sorted_mid, non_visual_after], dim=0).unsqueeze(0)

                    if attention_mask is not None:
                        attention_mask = attention_mask[:, :, :hidden_states.shape[1], :hidden_states.shape[1]]

                # Deep pruning policy.  For the latency-first story, Stage-III can be
                # skipped or replaced by deterministic uniform tail pruning.  This avoids
                # expensive/unreliable deep semantic scoring while preserving a fair ablation
                # path back to original ViTCoP/T3Guide.
                if decoder_layer.self_attn.layer_idx == self.deep_pruned_layer and seq_length > 1:
                    if self.stage3_policy != "skip":
                        device = hidden_states.device
                        visual_start = v_token_start
                        current_visual_token_num = self.stage2_visual_token_num if self.stage2_visual_token_num is not None else shallow_token_num
                        visual_end = v_token_start + current_visual_token_num
                        deep_token_num = max(1, min(deep_token_num, current_visual_token_num))

                        if self.stage3_policy in ["uniform", "random", "lite"]:
                            # Deterministic uniform selection: random-like cost/assumption, stable results.
                            if self.stage3_uniform_protect_clusters and self.stage2_token_sources is not None:
                                # First keep one representative per source group when possible.
                                protected = []
                                seen = set()
                                for idx, src in enumerate(self.stage2_token_sources):
                                    key = tuple(src) if src is not None else (idx,)
                                    if key not in seen:
                                        protected.append(idx)
                                        seen.add(key)
                                protected = protected[:deep_token_num]
                                if len(protected) < deep_token_num:
                                    all_local = torch.linspace(0, current_visual_token_num - 1, steps=deep_token_num, device=device).round().long().unique(sorted=True)
                                    extra = [int(x) for x in all_local.tolist() if int(x) not in protected]
                                    protected.extend(extra[:deep_token_num - len(protected)])
                                topk_local = torch.tensor(protected[:deep_token_num], device=device, dtype=torch.long).sort().values
                            else:
                                topk_local = torch.linspace(0, current_visual_token_num - 1, steps=deep_token_num, device=device).round().long().unique(sorted=True)
                                if topk_local.numel() < deep_token_num:
                                    fill = torch.arange(current_visual_token_num, device=device, dtype=torch.long)
                                    mask = torch.ones(current_visual_token_num, device=device, dtype=torch.bool)
                                    mask[topk_local] = False
                                    topk_local = torch.cat([topk_local, fill[mask][:deep_token_num - topk_local.numel()]]).sort().values
                        else:
                            # Original ViTCoP/T3Guide branch.
                            image_k_l2_norm = self.k_deep.mean(dim=1).norm(p=2, dim=-1)[0][visual_start:visual_end]

                            if self.use_t3guide:
                                text_token_start = visual_end
                                text_conditioned_score = compute_text_conditioned_score(
                                    hidden_states=hidden_states,
                                    visual_start=visual_start,
                                    visual_end=visual_end,
                                    text_start=text_token_start,
                                    temperature=self.t3guide_temperature,
                                )
                                fused_score = fuse_visual_text_scores(
                                    text_score=text_conditioned_score,
                                    visual_l2_norm=image_k_l2_norm,
                                    lambda_text=self.t3guide_lambda,
                                )

                                if self.t3guide_cluster_protect:
                                    topk_local = select_topk_with_source_protection(
                                        fused_score,
                                        self.stage2_token_sources,
                                        deep_token_num,
                                    )
                                else:
                                    topk_local = torch.topk(fused_score, deep_token_num, largest=True).indices.sort().values
                            else:
                                topk_local = torch.topk(image_k_l2_norm, deep_token_num, largest=False).indices.sort().values

                        selected_indices = topk_local + visual_start
                        non_visual_before = torch.arange(0, visual_start, device=device)
                        non_visual_after = torch.arange(visual_end, hidden_states.shape[1], device=device)
                        all_keep = torch.cat([non_visual_before, selected_indices, non_visual_after]).unique(sorted=True)

                        hidden_states = hidden_states.index_select(1, all_keep)
                        position_ids = all_keep.unsqueeze(0)
                        # E001-S1-A/B 2026-09-18: Stage3 stays Original in strict
                        # Stage1 experiments; record the resulting visual count only.
                        self.e001_last_final_visual_tokens = int(deep_token_num)
                        if self.e001_token_audit:
                            print(
                                f"[E001_STAGE3_TOKENS] method_id={self.e001_method_id} "
                                f"stage3_policy={self.stage3_policy} "
                                f"t3guide_enabled={self.use_t3guide} "
                                f"final_visual_tokens={self.e001_last_final_visual_tokens}"
                            )

                        if attention_mask is not None:
                            attention_mask = attention_mask[:, :, :hidden_states.shape[1], :hidden_states.shape[1]]

                # E006-QED-R1 decode audit: selectors must never execute during
                # decode.  We count the layer-12 decode visit only; the side selector
                # itself remains disabled.
                if (
                    self.e006_enable
                    and decoder_layer.self_attn.layer_idx == self.e010_qed_layer
                    and seq_length == 1
                ):
                    # This counter is a visit counter, not a selector-execution counter.
                    # e006_last['decode_count'] remains selector executions (=0).
                    pass

                # Attention mask adjustment for the generation phase (seq_length=1)
                if decoder_layer.self_attn.layer_idx == self.shallow_pruned_layer and seq_length == 1:
                    if attention_mask is not None:
                        past_key_values_length = past_key_values.get_usable_length(seq_length, self.shallow_pruned_layer)
                        attention_mask = attention_mask[:, :, :seq_length, :past_key_values_length + seq_length]

                if decoder_layer.self_attn.layer_idx == self.deep_pruned_layer and seq_length == 1:
                    if attention_mask is not None:
                        past_key_values_length = past_key_values.get_usable_length(seq_length, self.deep_pruned_layer)
                        attention_mask = attention_mask[:, :, :seq_length, :past_key_values_length + seq_length]

                # E006-QED-R1: configure the layer-12 SDPA module to collect
                # only the side question-row x prefix-key scores.  The normal
                # scaled_dot_product_attention call still executes unchanged.
                if (
                    self.e006_enable
                    and self.e006_selector in {"Q_ATTN", "Q_RESP", "QED", "QED_VISUAL", "RANDOM_FINAL"}
                    and decoder_layer.self_attn.layer_idx == self.e010_qed_layer
                    and seq_length > 1
                ):
                    if output_attentions:
                        raise RuntimeError("[E006] output_attentions must remain False")
                    removed = int(getattr(self, "_e010_visual_removed", 0))
                    qed_span = e006_question_span
                    if removed and e006_question_span is not None:
                        qed_span = (
                            int(e006_question_span[0]) - removed,
                            int(e006_question_span[1]) - removed,
                        )
                    self._e006_configure_layer12_side(
                        decoder_layer=decoder_layer,
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        visual_start=v_token_start,
                        visual_count=v_token_num,
                        token_length_list=token_length_list,
                        e006_question_span=qed_span,
                        e008_candidate_mask=e008_candidate_mask,
                        e008_mode=(
                            "CONTEXT_ONLY"
                            if (self.e009_enable or self.e013_has_context)
                            else (self.e008_mode if self.e008_enable else None)
                        ),
                    )
                    self.e006_prefill_count += 1

                # Post-layer pruning changes cache geometry: layer 12 keeps the
                # full Stage1 candidate-pool cache, while layer 13+ cache only the selected
                # E006_TARGET_K visual tokens. On decode, shrink the causal mask exactly
                # when entering layer 13 so it matches that layer's cache length.
                if (
                    self.e006_enable
                    and self.e006_selector in {"Q_ATTN", "Q_RESP", "QED", "QED_VISUAL", "RANDOM_FINAL"}
                    and decoder_layer.self_attn.layer_idx == self.e010_qed_layer + 1
                    and seq_length == 1
                    and attention_mask is not None
                ):
                    layer_past = past_key_values.get_usable_length(seq_length, decoder_layer.self_attn.layer_idx)
                    attention_mask = attention_mask[:, :, :seq_length, :layer_past + seq_length]

                if self.e010.enabled and seq_length == 1 and attention_mask is not None and past_key_values is not None:
                    layer_past = past_key_values.get_usable_length(
                        seq_length, decoder_layer.self_attn.layer_idx
                    )
                    attention_mask = attention_mask[:, :, :seq_length, :layer_past + seq_length]
                if self.e010.enabled and seq_length > 1:
                    decoder_layer.self_attn._e010_collect_local_influence = bool(
                        self.e010.collect_local_influence
                        and int(decoder_layer.self_attn.layer_idx) in set(int(x) for x in self.e010.analysis_layers)
                    )
                    decoder_layer.self_attn._e010_analysis_full576 = bool(self.e010.analysis_full576)
                    span_now = e006_question_span
                    removed_now = int(getattr(self, "_e010_visual_removed", 0))
                    if removed_now and span_now is not None:
                        span_now = (
                            int(span_now[0]) - removed_now,
                            int(span_now[1]) - removed_now,
                        )
                    decoder_layer.self_attn._e010_question_span = span_now
                    decoder_layer.self_attn._e010_visual_start = int(v_token_start)
                    decoder_layer.self_attn._e010_visual_count = int(v_token_num)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                # E006-QED-R1 G1-G4 prune AFTER the full layer-12 decoder block.
                # This is the key topology change relative to Original ViTCoP:
                # layer 12 supplies mature cross-modal response evidence, then
                # layers 13+ process only the selected E006_TARGET_K visual tokens.
                if (
                    self.e006_enable
                    and self.e006_selector in {"Q_ATTN", "Q_RESP", "QED", "QED_VISUAL", "RANDOM_FINAL"}
                    and decoder_layer.self_attn.layer_idx == self.e010_qed_layer
                    and seq_length > 1
                    and not (self.e010.enabled and self.e010.qed_side_only)
                ):
                    attn = decoder_layer.self_attn
                    side = getattr(attn, "_e006_last", None)
                    if side is None:
                        raise RuntimeError("[E006] layer-12 side selector did not produce statistics")

                    raw_score = side["score"]
                    score = side.get("selection_score", raw_score)
                    rho = side.get("rho")
                    if self.e006_selector == "QED_VISUAL":
                        visual_start = int(side["visual_start"])
                        visual_count = int(side["visual_count"])
                        post_layer_hidden = layer_outputs[0]
                        visual_hidden = post_layer_hidden[0, visual_start:visual_start + visual_count, :]
                        rho = self._e006_positive_redundancy(
                            side["r"], visual_hidden, eps=float(self.e006_eps)
                        )
                        score = side["r"] * (1.0 - side["D_Q"] * rho)

                    # E012-SD-P1 2026-10-09: final RANDOM K96 control is
                    # reproducible by request order and seed. It is an
                    # ACCURACY control, not a latency-equal selector.
                    if self.e012_enable and self.e006_selector == "RANDOM_FINAL":
                        rng = torch.Generator(device="cpu")
                        rng.manual_seed(
                            self.e012_random_seed + 100003 * int(self.e001_request_index)
                        )
                        selected_local = torch.randperm(
                            int(side["visual_count"]), generator=rng
                        )[:self.e006_target_k].to(
                            device=score.device, dtype=torch.long
                        ).sort().values
                    elif self.e016_cascade:
                        # Freeze a single best-first ranking; first cut uses schedule K0.
                        first_k = int(self.e016_cascade[0][1])
                        if first_k > int(side["visual_count"]):
                            raise RuntimeError(
                                f"[E016] first K={first_k} > visual_count={side['visual_count']}"
                            )
                        rank = torch.topk(
                            score, k=int(side["visual_count"]), largest=True, sorted=True
                        ).indices
                        self._e016_rank_orig = rank.detach().to(dtype=torch.long)
                        selected_local = rank[:first_k].sort().values
                        self._e016_alive_orig = selected_local.detach().clone()
                    else:
                        selected_local = torch.topk(
                            score, k=self.e006_target_k, largest=True, sorted=False
                        ).indices.sort().values
                    want_k = (
                        int(self.e016_cascade[0][1])
                        if self.e016_cascade
                        else int(self.e006_target_k)
                    )
                    if selected_local.unique().numel() != want_k:
                        raise RuntimeError("[E006] selected visual indices are not unique")

                    e008_would_select_transient = None
                    if self.e008_enable or self.e009_enable or self.e013_has_context:
                        candidate_mask = side.get("e008_candidate_mask")
                        if candidate_mask is None:
                            raise RuntimeError("[E008] layer-12 candidate mask missing")

                        if (self.e013_strict_check if self.e013_has_context else self.e008_strict_check):
                            if bool(
                                (
                                    ~candidate_mask.index_select(
                                        0, selected_local
                                    )
                                ).any().item()
                            ):
                                raise RuntimeError(
                                    "[E008] transient token survived final QED selection"
                                )

                        if self.e008_enable and self.e008_audit:
                            unmasked_topk = torch.topk(
                                raw_score,
                                k=self.e006_target_k,
                                largest=True,
                                sorted=False,
                            ).indices
                            e008_would_select_transient = int(
                                (
                                    ~candidate_mask.index_select(
                                        0, unmasked_topk
                                    )
                                ).sum().item()
                            )

                    keep_local = selected_local
                    if (
                        self.e010.enabled
                        and self.e010.keep_context_after_qed
                        and int(self.e010.context_exit_layer) > int(self.e010_qed_layer)
                        and (self.e008_enable or self.e009_enable)
                    ):
                        candidate_mask = side.get("e008_candidate_mask")
                        if candidate_mask is None:
                            raise RuntimeError("[E010] candidate mask missing for context lifetime")
                        context_local = torch.arange(
                            candidate_mask.numel(), device=selected_local.device
                        )[~candidate_mask.to(selected_local.device)]
                        if int(context_local.numel()) > 0:
                            keep_local = torch.cat(
                                [selected_local, context_local]
                            ).unique(sorted=True)

                    # E012-SD-P1 2026-10-09: the same S-layer selected IDs
                    # are either deleted NOW (S==D; legacy path) or stored
                    # for post-layer D. No QED re-scoring happens at D.
                    e012_defer = (
                        self.e012_enable
                        and self.e012_delete_layer > self.e010_qed_layer
                        and not self.e016_cascade
                    )
                    if e012_defer:
                        if e012_pending_selected is not None:
                            raise RuntimeError("[E012] selector fired twice in one prefill")
                        e012_pending_selected = selected_local.detach()
                        e012_pending_visual_start = int(side["visual_start"])
                        e012_pending_visual_count = int(side["visual_count"])
                        # Keep entire Core172 in hidden_states/position_ids/mask
                        # through the end of decoder layer D.
                    else:
                        pruned_hidden, position_ids, attention_mask, all_keep = self._e006_prune_sequence(
                            layer_outputs[0],
                            position_ids,
                            attention_mask,
                            int(side["visual_start"]),
                            int(side["visual_count"]),
                            keep_local,
                        )
                        if self.e010.enabled:
                            if cluster_labels is not None and int(cluster_labels.shape[-1]) == int(side["visual_count"]):
                                cluster_labels = cluster_labels.index_select(
                                    1, keep_local.to(device=cluster_labels.device)
                                )
                            image_shape = int(keep_local.numel())
                            self._e010_visual_removed = int(getattr(self, "_e010_visual_removed", 0)) + (
                                int(side["visual_count"]) - image_shape
                            )
                        layer_outputs = (pruned_hidden,) + tuple(layer_outputs[1:])

                    # Resolve diagnostic CUDA events only in profile mode.
                    timing = {}
                    if self.e006_profile and side.get("events") is not None:
                        torch.cuda.synchronize()
                        ev = side["events"]
                        timing = {
                            "side_qk_ms": float(ev[0].elapsed_time(ev[1])),
                            "response_similarity_ms": float(ev[1].elapsed_time(ev[2])),
                            "score_ms": float(ev[2].elapsed_time(ev[3])),
                        }

                    if not e012_defer:
                        out_k = (
                            int(self.e016_cascade[0][1])
                            if self.e016_cascade
                            else int(self.e006_target_k)
                        )
                        self.stage2_visual_token_num = out_k
                        self.stage2_token_sources = None
                        self.e001_last_stage2_visual_tokens = out_k
                        if self.stage3_policy == "skip" and not self.e016_cascade:
                            self.e001_last_final_visual_tokens = out_k
                        if self.e016_cascade:
                            image_shape = out_k
                            v_token_num = out_k

                    # Keep diagnostics on-device in normal runs so canonical
                    # outer wall-clock TTFT is not polluted by .item()/.cpu()
                    # synchronization.  Debug/profile mode may materialize them.
                    self.e006_last = {
                        "selector": self.e006_selector,
                        # E014-AEE-BEGIN: static identity only, no fabricated dynamic budget
                        # Dynamic exchange budget/IDs live at the CLIP
                        # vision tower, not inside LlamaModel.  Do not
                        # fabricate them from E014_B in adaptive runs.
                        "e014_enable": bool(self.e014_enable),
                        "e014_variant": (
                            self.e014_variant if self.e014_enable else None
                        ),
                        "e014_stage1_visual_out": (
                            172 if self.e014_enable else None
                        ),
                        # E014-AEE-END: static provenance
                        "e010_qed_layer": int(self.e010_qed_layer),
                        "e010_enabled": bool(self.e010.enabled),
                        "span_source": side["span_source"],
                        "question_indices": side["question_indices"].detach(),
                        "visual_in": int(side["visual_count"]),
                        "visual_out": int(self.e006_target_k),
                        # E012-SD-P1 2026-10-09: planned final K vs physical K.
                        "e012_deferred": bool(e012_defer),
                        "e012_selection_layer": int(self.e010_qed_layer),
                        "e012_deletion_layer": int(self.e012_delete_layer),
                        "visual_physical_after_selection": (
                            int(side["visual_count"]) if e012_defer else int(self.e006_target_k)
                        ),
                        "D_Q": None if side["D_Q"] is None else side["D_Q"].detach(),
                        "r": side["r"].detach(),
                        "rho": None if rho is None else rho.detach(),
                        "selected_local": selected_local.detach(),
                        "prefill_count": int(self.e006_prefill_count),
                        "decode_count": int(self.e006_decode_count),
                        "sdpa_preserved": True,
                        "output_attentions": False,
                        "e007_enable": bool(self.e007_enable),
                        "e007_stage1_mode": (
                            self.e007_stage1_mode if self.e007_enable else None
                        ),
                        "e007_request_id": (
                            int(self.e001_request_index) if self.e007_enable else None
                        ),
                        "e008_enable": bool(self.e008_enable),
                        "e008_mode": (
                            self.e008_mode if self.e008_enable else None
                        ),
                        "e008_would_select_transient": e008_would_select_transient,
                        "e009_enable": bool(self.e009_enable),
                        "e009_mode": (
                            self.e009_mode if self.e009_enable else None
                        ),
                        **timing,
                    }

                    # E012-SD-P1 2026-10-09: OFFLINE-ONLY diagnostic.
                    # Avoid E012_DIAG=True for publication-facing latency: JSON
                    # conversion synchronizes CUDA and adds massive CPU I/O.
                    if self.e012_enable and self.e012_diag:
                        r_cpu = side["r"].detach().float().cpu()
                        relevance_ids = torch.topk(
                            r_cpu, self.e006_target_k, sorted=False
                        ).indices.sort().values.tolist()
                        audit_record = {
                            "request_index": int(self.e001_request_index),
                            "selector": self.e006_selector,
                            "selection_layer": int(self.e010_qed_layer),
                            "deletion_layer": int(self.e012_delete_layer),
                            "visual_in": int(side["visual_count"]),
                            "final_k": int(self.e006_target_k),
                            "selected_local": selected_local.detach().cpu().tolist(),
                            "qattn_selected_local": relevance_ids,
                            "r": r_cpu.tolist(),
                            "D_Q": (
                                None if side["D_Q"] is None
                                else float(side["D_Q"].detach().float().cpu().item())
                            ),
                            "rho": (
                                None if rho is None
                                else rho.detach().float().cpu().tolist()
                            ),
                            "w": (
                                None if side.get("w") is None
                                else side["w"].detach().float().cpu().tolist()
                            ),
                            "P": (
                                None if side.get("P") is None
                                else side["P"].detach().float().cpu().tolist()
                            ),
                            "random_seed": (
                                self.e012_random_seed
                                if self.e006_selector == "RANDOM_FINAL" else None
                            ),
                        }
                        print("[E012_DIAG_JSON] " + json.dumps(audit_record, separators=(",", ":")))

                    if self.e009_enable and self.e009_audit:
                        candidate_mask = side.get("e008_candidate_mask")
                        final_context = int(
                            (
                                ~candidate_mask.index_select(0, selected_local)
                            ).sum().item()
                        )
                        payload = {
                            "request_id": int(self.e001_request_index),
                            "mode": self.e009_mode,
                            "visual_in": int(side["visual_count"]),
                            "candidate_count": 172,
                            "context_count": 16,
                            "target_k": int(self.e006_target_k),
                            "final_context_survivors": final_context,
                            "side_mode": "CONTEXT_ONLY",
                            "sdpa_preserved": True,
                        }
                        print(
                            "[E009_QED_JSON] "
                            + json.dumps(payload, separators=(",", ":"))
                        )

                    if self.e008_enable and self.e008_audit:
                        e008_payload = {
                            "request_id": int(self.e001_request_index),
                            "mode": self.e008_mode,
                            "target_k": int(self.e006_target_k),
                            "visual_in": int(side["visual_count"]),
                            "candidate_count": 172,
                            "transient_count": 20,
                            "would_select_transient_unmasked": (
                                int(e008_would_select_transient)
                                if e008_would_select_transient is not None
                                else None
                            ),
                            "final_selected_transient": 0,
                        }
                        print(
                            "[E008_QED_JSON] "
                            + json.dumps(e008_payload, separators=(",", ":"))
                        )

                    # Diagnostic-only provenance line. selected_local indexes
                    # directly into the Stage1 output order emitted by
                    # clip_encoder's [E007_STAGE1_JSON].
                    if self.e007_enable and self.e007_token_audit:
                        selected_local_cpu = selected_local.detach().cpu().tolist()
                        audit_payload = {
                            "request_id": int(self.e001_request_index),
                            "stage1_mode": self.e007_stage1_mode,
                            "visual_in": int(side["visual_count"]),
                            "visual_out": int(self.e006_target_k),
                            "selected_local": selected_local_cpu,
                            "selector": self.e006_selector,
                            "layer": int(
                                self.e010_qed_layer if self.e012_enable else self.e006_layer
                            ),
                            "stage3_policy": self.stage3_policy,
                            "sdpa_preserved": True,
                        }
                        print(
                            "[E007_QED_JSON] "
                            + json.dumps(audit_payload, separators=(",", ":"))
                        )

                    if self.e006_debug:
                        q_cpu = side["question_indices"].detach().cpu()
                        r_cpu = side["r"].detach().float().cpu()
                        rho_cpu = None if rho is None else rho.detach().float().cpu()
                        dq = None if side["D_Q"] is None else float(side["D_Q"].item())
                        debug_log = {
                            "selector": self.e006_selector,
                            "span_source": side["span_source"],
                            "question_start": int(q_cpu[0].item()),
                            "question_end": int(q_cpu[-1].item()) + 1,
                            "question_token_count": int(q_cpu.numel()),
                            "visual_in": int(side["visual_count"]),
                            "visual_out": int(self.e006_target_k),
                            "D_Q": dq,
                            "r_min": float(r_cpu.min().item()),
                            "r_mean": float(r_cpu.mean().item()),
                            "r_max": float(r_cpu.max().item()),
                            "rho_min": None if rho_cpu is None else float(rho_cpu.min().item()),
                            "rho_mean": None if rho_cpu is None else float(rho_cpu.mean().item()),
                            "rho_max": None if rho_cpu is None else float(rho_cpu.max().item()),
                            "selected_local": selected_local.detach().cpu().tolist(),
                            "prefill_count": int(self.e006_prefill_count),
                            "decode_count": int(self.e006_decode_count),
                            "sdpa_preserved": True,
                            "output_attentions": False,
                            **timing,
                        }
                        print(f"[E006_QED] {debug_log}")

                # Clear one-shot side context after the prefill layer executes.
                if (
                    self.e006_enable
                    and decoder_layer.self_attn.layer_idx == self.e010_qed_layer
                    and hasattr(decoder_layer.self_attn, "_e006_enabled")
                ):
                    decoder_layer.self_attn._e006_enabled = False
                if decoder_layer.self_attn.layer_idx == self.shallow_pruned_layer - 1 and seq_length > 1:
                    self.k_shallow = layer_outputs[2 if output_attentions else 1][decoder_layer.self_attn.layer_idx][0]

                if decoder_layer.self_attn.layer_idx == self.deep_pruned_layer - 1 and seq_length > 1:
                    self.k_deep = layer_outputs[2 if output_attentions else 1][decoder_layer.self_attn.layer_idx][0]
            ## VITCOP
            ########################################################################################################################

            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            if (
                self.e010.enabled
                and int(self.e010.context_exit_layer) >= 0
                and decoder_layer.self_attn.layer_idx == int(self.e010.context_exit_layer)
                and seq_length > 1
                and cluster_labels is not None
                and cluster_labels.dim() == 2
            ):
                labels_now = cluster_labels[0]
                keep_context_free = torch.arange(
                    labels_now.numel(), device=labels_now.device
                )[labels_now >= 0]
                removed_now = int(labels_now.numel() - keep_context_free.numel())
                if removed_now > 0:
                    v_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 35
                    pruned_hidden, position_ids, attention_mask, _all_keep = self._e006_prune_sequence(
                        layer_outputs[0],
                        position_ids,
                        attention_mask,
                        int(v_start),
                        int(labels_now.numel()),
                        keep_context_free,
                    )
                    layer_outputs = (pruned_hidden,) + tuple(layer_outputs[1:])
                    cluster_labels = cluster_labels.index_select(1, keep_context_free)
                    image_shape = int(keep_context_free.numel())
                    self._e010_visual_removed = int(getattr(self, "_e010_visual_removed", 0)) + removed_now
                    self._e010_last_context_exit = {
                        "layer": int(decoder_layer.self_attn.layer_idx),
                        "removed": removed_now,
                        "visual_after": image_shape,
                    }

            # E012-SD-P1 2026-10-09: physically delete the exact S-layer
            # survivor complement AFTER decoder layer D, without re-selection.
            # Layers <=D keep full Core172 KV; layers >D receive K96 tokens.
            if (
                self.e012_enable
                and seq_length > 1
                and self.e012_delete_layer > self.e010_qed_layer
                and decoder_layer.self_attn.layer_idx == self.e012_delete_layer
            ):
                if e012_pending_selected is None:
                    raise RuntimeError("[E012] D reached without frozen S-layer selection")
                if int(v_token_start) != e012_pending_visual_start or int(image_shape) != e012_pending_visual_count:
                    raise RuntimeError(
                        "[E012] visual span changed between S and D; "
                        "the fixed-survivor intervention is invalid"
                    )
                pruned_hidden, position_ids, attention_mask, _keep = self._e006_prune_sequence(
                    layer_outputs[0],
                    position_ids,
                    attention_mask,
                    int(e012_pending_visual_start),
                    int(e012_pending_visual_count),
                    e012_pending_selected,
                )
                if cluster_labels is not None:
                    if cluster_labels.dim() != 2 or int(cluster_labels.shape[-1]) != e012_pending_visual_count:
                        raise RuntimeError("[E012] cluster metadata length changed before D")
                    cluster_labels = cluster_labels.index_select(
                        1, e012_pending_selected.to(device=cluster_labels.device)
                    )
                removed = e012_pending_visual_count - int(e012_pending_selected.numel())
                image_shape = int(e012_pending_selected.numel())
                self._e010_visual_removed += removed
                self.stage2_visual_token_num = image_shape
                self.stage2_token_sources = None
                self.e001_last_stage2_visual_tokens = image_shape
                self.e001_last_final_visual_tokens = image_shape
                layer_outputs = (pruned_hidden,) + tuple(layer_outputs[1:])
                self.e012_last = {
                    "selection_layer": int(self.e010_qed_layer),
                    "deletion_layer": int(self.e012_delete_layer),
                    "selected_local": e012_pending_selected.detach(),
                    "visual_before": int(e012_pending_visual_count),
                    "visual_after": image_shape,
                    "selector_calls": int(self.e006_prefill_count),
                }
                e012_pending_selected = None

            # E016 cascade follow-up cuts (after layers 10/16/20/...): prefix of
            # the frozen QED ranking among currently alive original locals.
            if (
                self.e016_cascade
                and self.e006_enable
                and seq_length > 1
                and decoder_layer.self_attn.layer_idx != self.e010_qed_layer
            ):
                layer_idx_now = int(decoder_layer.self_attn.layer_idx)
                for sched_layer, sched_k in self.e016_cascade[1:]:
                    if layer_idx_now != int(sched_layer):
                        continue
                    rank = getattr(self, "_e016_rank_orig", None)
                    alive = getattr(self, "_e016_alive_orig", None)
                    if rank is None or alive is None:
                        raise RuntimeError("[E016] cascade ranking missing before follow-up cut")
                    alive_list = alive.detach().tolist()
                    alive_set = set(alive_list)
                    keep_orig = []
                    for idx in rank.detach().tolist():
                        if idx in alive_set:
                            keep_orig.append(int(idx))
                        if len(keep_orig) >= int(sched_k):
                            break
                    if len(keep_orig) != int(sched_k):
                        raise RuntimeError(
                            f"[E016] could not gather K={sched_k} from alive={len(alive_list)}"
                        )
                    # Current visual positions are spatial-sorted alive originals.
                    alive_sorted = sorted(alive_list)
                    pos = {orig: i for i, orig in enumerate(alive_sorted)}
                    selected_curr = torch.tensor(
                        [pos[o] for o in keep_orig],
                        device=layer_outputs[0].device,
                        dtype=torch.long,
                    )
                    visual_count_now = int(image_shape)
                    if visual_count_now != len(alive_sorted):
                        raise RuntimeError(
                            f"[E016] image_shape={visual_count_now} != alive={len(alive_sorted)}"
                        )
                    pruned_hidden, position_ids, attention_mask, _ = self._e006_prune_sequence(
                        layer_outputs[0],
                        position_ids,
                        attention_mask,
                        int(v_token_start),
                        visual_count_now,
                        selected_curr,
                    )
                    if cluster_labels is not None and cluster_labels.dim() == 2:
                        cluster_labels = cluster_labels.index_select(
                            1, selected_curr.to(device=cluster_labels.device)
                        )
                    image_shape = int(sched_k)
                    v_token_num = int(sched_k)
                    self._e010_visual_removed = int(getattr(self, "_e010_visual_removed", 0)) + (
                        visual_count_now - image_shape
                    )
                    self._e016_alive_orig = torch.tensor(
                        keep_orig, device=rank.device, dtype=torch.long
                    ).sort().values
                    self.stage2_visual_token_num = image_shape
                    self.e001_last_stage2_visual_tokens = image_shape
                    if int(sched_k) == int(self.e006_target_k) and self.stage3_policy == "skip":
                        self.e001_last_final_visual_tokens = image_shape
                    layer_outputs = (pruned_hidden,) + tuple(layer_outputs[1:])
                    self.e016_last = {
                        "layer": layer_idx_now,
                        "keep": int(sched_k),
                        "alive_before": visual_count_now,
                    }
                    break

            hidden_states = layer_outputs[0]

            # E012-SD-P1 2026-10-09: verify prefill KV topology and the
            # first two decode visits.  This checks the REAL per-layer
            # DynamicCache lengths; no fake all-layer KV compaction claims.
            if self.e012_enable and self.e012_kv_audit and (
                seq_length > 1 or self.e012_decode_visit <= 2
            ):
                layer_idx = int(decoder_layer.self_attn.layer_idx)
                kv_len = int(past_key_values.get_seq_length(layer_idx))
                prefix_len = self.e012_prefill_len
                if prefix_len is None:
                    raise RuntimeError("[E012] missing matching multimodal prefill")
                # E013-P1 2026-10-09: EXTRA uses 172+b prefill visual positions;
                # compaction removes (172+b-K), not the Core172-only (172-K).
                visual_before = 172 + (int(self.e013_b) if self.e013_has_context else 0)
                removed_at_layer = (
                    0 if layer_idx <= self.e012_delete_layer
                    else int(self.e006_target_k) - visual_before
                )
                expected = prefix_len + (0 if seq_length > 1 else self.e012_decode_visit) + removed_at_layer
                if kv_len != expected:
                    raise RuntimeError(
                        f"[E012_KV] layer={layer_idx} prefill={seq_length > 1} "
                        f"observed={kv_len} expected={expected}"
                    )
                print("[E012_KV_JSON] " + json.dumps({
                    "request_index": int(self.e001_request_index),
                    "layer": layer_idx,
                    "phase": "prefill" if seq_length > 1 else f"decode{self.e012_decode_visit}",
                    "cache_tokens": kv_len,
                    "expected_tokens": expected,
                    "S": int(self.e010_qed_layer),
                    "D": int(self.e012_delete_layer),
                    "K": int(self.e006_target_k),
                    "visual_before": int(visual_before),
                }, separators=(",", ":")))

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # E012-SD-P1 2026-10-09: never accept a truncated prefill where
        # S ran but D did not execute.
        if self.e012_enable and seq_length > 1 and e012_pending_selected is not None:
            raise RuntimeError("[E012] prefill ended with pending, undeleted Core tokens")

        if e001_efficiency_active and e001_is_prefill:
            total_end_event.record()
            torch.cuda.synchronize()
            prefill_time_ms = total_start_event.elapsed_time(total_end_event)
            self.prefill_time += prefill_time_ms
            self.num_forward += 1
            FLOPs_avg_sample = (self.all_FLOPs / self.num_forward) * 1e-12
            self.max_gpu_mem = max(
                self.max_gpu_mem,
                torch.cuda.max_memory_allocated() / 1024 / 1024,
            )
            # E001-S1-B 2026-09-18: structured raw timing line. CUDA timing
            # covers the Llama decoder prefill only; it is NOT full VLM TTFT.
            print(
                f"[E001_EFF] timing_mode={self.e001_timing_mode} "
                f"method_id={self.e001_method_id} request_id={self.e001_request_index} "
                f"metric=prefill_ms value={prefill_time_ms:.6f} "
                f"stage1_visual_tokens={self.e001_last_stage1_visual_tokens} "
                f"stage2_visual_tokens={self.e001_last_stage2_visual_tokens} "
                f"final_visual_tokens={self.e001_last_final_visual_tokens} "
                f"peak_gpu_memory_mb={self.max_gpu_mem:.3f} "
                f"tflops_avg_sample={FLOPs_avg_sample:.6f}"
            )

        if e001_efficiency_active and e001_is_decode:
            total_end_event.record()
            torch.cuda.synchronize()
            inference_time_ms = total_start_event.elapsed_time(total_end_event)
            self.inference_time_per_token += inference_time_ms
            self.num_token += 1
            self.e001_decode_token_index += 1
            self.max_gpu_mem = max(
                self.max_gpu_mem,
                torch.cuda.max_memory_allocated() / 1024 / 1024,
            )
            print(
                f"[E001_EFF] timing_mode={self.e001_timing_mode} "
                f"method_id={self.e001_method_id} request_id={self.e001_request_index} "
                f"decode_token_id={self.e001_decode_token_index} "
                f"metric=decode_token_ms value={inference_time_ms:.6f} "
                f"peak_gpu_memory_mb={self.max_gpu_mem:.3f}"
            )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cluster_labels = None,
        image_shape = 576,
        token_length_list = [],
        pre_prompt_length_list = [],
        e006_question_span = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cluster_labels = cluster_labels,
            image_shape = image_shape,        
            token_length_list = token_length_list,  
            pre_prompt_length_list = pre_prompt_length_list,
            e006_question_span = e006_question_span,
        )

        # E018-BEGIN: evaluate answer NLL by prefix + one-token teacher forcing.
        if labels is not None and getattr(self.model, "e018_enabled", False):
            raise ValueError("E018 forbids full-prompt labels; use collect_e018.py teacher forcing")
        # E018-END

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # E018-BEGIN: compacted cache length is not the original RoPE position.
        if getattr(self.model, "e018_enabled", False):
            from llava.model.e018_stage2_support import e018_generation_inputs
            return e018_generation_inputs(self, input_ids, past_key_values,
                attention_mask, inputs_embeds, kwargs)
        # E018-END
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        cluster_labels=None,
        image_shape=576,
        token_length_list=[],
        pre_prompt_length_list = [],
        e006_question_span = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # three conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same);
            # 3) the user must have set generation parameters in the model config.
            if (
                self.generation_config._from_model_config
                and self.generation_config._original_object_hash == hash(self.generation_config)
                and self.config._has_non_default_generation_parameters()
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                print(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            print(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                print(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                print(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = self._get_generation_mode(generation_config, assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
            )

            # 12. run assisted generate
            return self.assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                do_sample=generation_config.do_sample,
                logits_processor=prepared_logits_processor,
                logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if generation_mode == GenerationMode.GREEDY_SEARCH:
            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                cluster_labels= cluster_labels,
                image_shape = image_shape,
                token_length_list = token_length_list,
                pre_prompt_length_list = pre_prompt_length_list,
                e006_question_span = e006_question_span,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            return self.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                sequential=generation_config.low_memory,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample
            return self.sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        cluster_labels = None,
        image_shape = 576,
        token_length_list = [],
        pre_prompt_length_list = [],
        e006_question_span = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        self.model.generate_process_count = 0

        causal_inference_start_event = torch.cuda.Event(enable_timing=True)
        causal_inference_end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        causal_inference_start_event.record()
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cluster_labels = cluster_labels,
                image_shape = image_shape,
                token_length_list = token_length_list,
                pre_prompt_length_list = pre_prompt_length_list,
                e006_question_span = e006_question_span,
            )
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)      

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs['attention_mask'] = model_kwargs['attention_mask'].new_ones((model_kwargs['attention_mask'].shape[0], outputs['past_key_values'][0][0].shape[2]))
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        # causal_inference_end_event.record()
        # torch.cuda.synchronize()

        # causal_inference_cuda_time_ms = causal_inference_start_event.elapsed_time(causal_inference_end_event)
        # self.model.causal_inference_cuda_time += causal_inference_cuda_time_ms
        # FLOPs_avg = all_FLOPs /self.num_layers
        # print(f"total_causal_inference_cuda_time:{self.model.causal_inference_cuda_time}")
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
