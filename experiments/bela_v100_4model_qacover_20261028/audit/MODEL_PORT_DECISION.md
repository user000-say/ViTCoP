# Model port decision — 2026-10-28

Verdict: **BLOCKED** for G1–G3. No four-model accuracy was launched.

## What passed

- 16 cross-backbone tests and 23 original QA-COVER tests.
- Budget planner: LLaVA-NeXT-7B stress keeps 5.556% (94.44% removed, W5120); moderate keeps 11.11% (88.89% removed, W10240). These match the LLaVA-1.5 W1024 / W2048 removal rates. Qwen3-VL stress removes 90.12% (W4608). Qwen3.5 moderate is Stage1-only K128, 87.5% removed, and is not full BELA.
- Live configs: Qwen3-VL-8B is `qwen3_vl`, 36 decoder layers, DeepStack indexes 8/16/24, dtype bfloat16. Qwen3.5-9B is `qwen3_5`, 32 layers, 24 linear-attention and full-attention at layers 4,8,12,16,20,24,28,32, dtype bfloat16. LLaVA-NeXT-13B config reports 40 decoder layers.

## Why full evaluation is blocked

1. None of the four checkpoints is cached. The hub only has `liuhaotian/llava-v1.5-7b` and `llava-v1.5-13b`.
2. `tools/entry_bridge.py` selects already-aligned token rows. It does not install LLaVA-NeXT crop packing, Qwen DeepStack, M-RoPE, or a decoder prune. The handoff forbids scoring with the bridge alone.
3. The live LLaVA class is `LlavaLlamaForCausalLM` (1.5). lmms-eval has `llava_hf` for NeXT, and Qwen2-VL / Qwen2.5-VL adapters, not Qwen3-VL or Qwen3.5.
4. LLaVA-NeXT-7B `config.json` omits `text_config.num_hidden_layers`, so `model_config_probe` correctly refuses to invent 32. The 13B file does contain 40.
5. Qwen3.5 cannot use the LLaVA RDRIFT path: 24 layers are Gated DeltaNet. Any later run is Stage1-only and must be labeled that way. V100 has no native BF16.
6. LLaVA-NeXT-13B FP16 on a 32GB V100 is not assumed to fit. An OOM stops that arm.

## Not changed

QA-COVER thresholds stay at the frozen LLaVA-1.5 values. No new selector, no test-label sweep, and the derived Qwen3-VL 94.44% schedule was not queued.
