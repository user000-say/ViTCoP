# Blocking source-code integration contract
### G0 preconditions (MUST complete)
- Live BELA code root and git sha; exact E013/E014 helper sources and whether modified since upload.
- `ViT Layer 9` true source semantics: CLIP `hidden_states[9]` is state after 9 blocks if embeddings at 0, while `attentions[8]` is ninth block attention; verify CLIP variant.
- Current LLaVA projector takes which actual hidden features? Are intermediate layers *only for selection*? If not, record architecture.
- For cut after decoder layers 4,12,22 (E3) and 12,20 (L2): exact file path, function, call stack, current attention query positions, normalization, score, token-id gather, mask, position, use_cache and per-block KV length.
- Smoke with 3 inputs and run `tools/budget.py` validation of observed full `[32]` counts and original Patch IDs.
- STAR-Pro official overlay separate virtual environment; checksum frozen checkpoint and raw evaluation prompts. Need `METHOD=star_pro` and official QR scorer, not old THCP.
- If no clean migration of one opponent selector into BELA, stop 2×2 crossover, keep competitor as same-machine full-system baseline and native Stage1/Stage2 ablation instead.

### Robust physical cut contract (NOT equivalent to masking)
For each cut, emit `{after_decoder_layer, original_before, original_after, selector_callsite, Qattn_aggregation, head_aggregation}`. All original IDs within 0..575, unique, nested. `active_visual_by_layer` has exactly 32 integers. Check hidden states, `attention_mask`, `position_ids` and cached decoding. Fresh compute cost excludes selection if disabled; full real TTFT includes encoder when stated.

### Cross-over fairness
For STAR QR + BELA S2, do not run QR inside a separately trained visual projector; share the SAME full ViT projected original tokens, then only override candidate IDs. For BELA S1 + STAR S2, provide original Patch IDs and consistent token positions to the STAR S2 code; never use incompatible Attention logits without calibration. If adapters change language model weights or force a different attention backend, label it an invalid/fairness-blocked hybrid.

### Practical parameter guards
- Bounded Stage1 scan K=64 at W1024 L2; K=128 at W2048 E5; K=172 at W2048 E3.
- Later cut E3 is at L22, **not at L12**. Last 16 tokens survive L23-L32.
- Official STAR-Pro source may request full attention (flash fallback) and patched RoPE cache; a single run with faster kernels disabled must be compared on exact same conditions.
- Never state that W2048 guarantees equal end-to-end latency. E1 disproved it empirically.

### Dataset and metric guards
GQA testdev split, POPE F1 (popular/random/adversarial TP/FP/FN), MME Total vs Perception (their denominators and task lists DIFFER), MMBench circular evaluation, TextVQA normalized answer, SQA-IMG vs full science set. Save question_id, original image ID, score parser output. Do not mix STAR paper "MME=1421" (perception scale) with BELA Total "1797".
