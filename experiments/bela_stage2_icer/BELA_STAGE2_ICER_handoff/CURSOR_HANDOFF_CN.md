# BELA Stage2 最终设计与 Cursor Handoff（基于 2026-10-20 Gate1）

## 0. 结论先行：主方法怎么定

**不要再加一个自由浮动的 scorer，也不要立即上 learned future-utility predictor。** 先把 Stage2 定义成一个预算不变的两步决策：

1. **Core relevance**：原样使用当前已验证的 Q-Attn 或源代码确认过的 RDRIFT，产生固定 K 的基础集合 `S0`。
2. **Interaction-conditioned complementarity repair (ICER)**：只在 Top-K 边界附近做少量 donor↔dropper 交换；donor 必须补足当前集合对 Stage1 admission pool 的覆盖缺口，交换不得明显牺牲 query relevance；RDRIFT 只在非最终边界给 donor shortlist 一个临时偏好，最终边界关闭该偏好。

所有交换均保持 `K`、事件边界和 W 不变。Stage1 的 COVER/PCOV 描述符与 original patch IDs 必须跟踪到 decoder。这个版本是可检验的研究原型，不是已经证明有效的最终算法；若同预算精度不提升或时延成本抵消收益，就不把 ICER 放进最终主方法。

**不要把“互补性保护”写成永久保留一组 token。** 每个边界重新计算，避免早期 token 被无限保护。不要将 RDRIFT 写成已证明的未来价值预测器；当前它只是趋势提示。

## 1. Gate1 结果与现在的判断

本轮九任务实测：
- W2048 COVER+Q-ATTN: R4=95.3621, TTFT wall/cuda=124.87/124.77 ms
- W2048 COVER+RDRIFT: R4=95.7571, TTFT=125.66/125.56 ms
- W1024 PCOV+Q-ATTN: R4=91.8111, TTFT=118.99/118.84 ms

W2048 日程：`172×4 + 90×8 + 32×20 = 2048`，events after blocks 4 and 12 (verify exact zero-/one-based mapping in source).
W1024 日程：`60×10 + 28×10 + 12×12 = 1024`，events after blocks 10 and 20 (verify source; don't rely on prior shorthand).
Gate1 gives a small but not yet statistically validated RDRIFT gain (+0.395 R4) at W2048. Do not treat this as a universal win; task-level tradeoffs exist. The previous "98/96" figures were relative-retention R8 estimates using mixed Dense baselines, not raw accuracy, and are not the nine-task Gate1 R4. Freeze a single official score definition before final paper tables.

## 2. What the papers justify—and what they don't

- **STAR-Pro**: official LLaVA code uses pivoted-QR admission pool plus progressive decoder pruning; its default text-attention aggregation + Top-K is a strong baseline. Reproduce its published path separately; don't call a hybrid BELA schedule "STAR-Pro".
- **CoRePrune**: arXiv 2609.26484 argues removability depends on representation depth and the surrounding deletion set; offers perturbation and set-conditioned rescue analysis. The independent diagnostic functions in this package are mathematical reimplementations, not official authors' code. Do not claim its code was reviewed if unavailable; don't assume local perturbation predicts answer quality.
- **SCOPE / VisionDrop / evaluation critique**: motivate coverage and alignment controls. Cite these as related work, not as components that automatically guarantee gains.

## 3. Exact ICER behavior

At each already-scheduled pruning boundary:
1. Read the current active original patch IDs, actual Q-Attn score and (if selected) exact legacy RDRIFT score. Score array order must match active visual slots.
2. Produce baseline `S0 = TopK(rank_score, K)` using deterministic tie-break (original ID ascending).
3. Compute descriptor similarity within the **current admitted candidate pool**. Weight reference tokens using a small uniform prior mixed with normalized question-to-visual mass.
4. Consider only bounded candidate sets: top `donor_pool=24` by coverage gain plus optional positive drift; `drop_pool=16` lower-defensibility kept candidates.
5. Evaluate one-for-one swaps. Accept only if coverage improves, retained Q-Attn mass stays above `min_qmass_ratio=0.97` of the original Top-K mass, and utility exceeds a minimum.
6. Repeat at most 3 times. At final boundary, set drift contribution to zero.
7. Return slots sorted in current sequence order; caller must use the **existing, validated model pruning path** to update hidden states, visual position IDs, attention mask and cache. The helper itself does not mutate model tensors or cache.

This prototype's coverage is a nonnegative cosine/facility-location proxy. It is not QR and not CoRePrune's selector. Test `max_swaps=0` first as a parity baseline.

## 4. CoRePrune-style perturbation diagnostic

`bela/stage2_icer.py` includes `singleton_perturbation` and `set_perturbation`, for controlled comparison only:
- Inputs must be actual layer attention probabilities normalized over **all causally visible keys** and actual post-projection-to-attention `V` states from that layer; the current implementation's shape is `[heads, receivers, all_keys]` and `[heads, all_keys, head_dim]`.
- `visual_key_positions` are positions in the full key sequence; don't softmax only over visual keys.
- Receiver indices should be the question-token subset that is causally valid in the actual prompt.
- These are frozen-QKV local perturbations; they don't include later MLP/blocks or prove answer quality. CoRePrune is an ablation, not the default BELA method.
- The projected-`W_O` variant is not implemented here; first establish the paper-style diagnostic, then add `W_O` as a separately named ablation if useful.

## 5. Critical TTFT diagnosis: why 2x smaller W only saved ~6 ms

The reported TTFT values are:
- W2048 Q-Attn: 124.87 ms
- W2048 RDRIFT: 125.66 ms
- W1024 Q-Attn: 118.99 ms

So the W2048-Q vs W1024 difference is about **5.87 ms / 4.7%**, not a halving of total compute. This is plausible for several reasons:
1. W is `sum of visual token counts across decoder blocks`, not whole-model FLOPs. W2048 vs W1024 removes 1,024 token-layers; relative to the 18,432 dense visual token-layers, that's only 5.56 percentage points of that proxy, not 50% of total-model work.
2. Stage1 still runs the full vision encoder over 576 patches; pruning after the encoder does not reduce encoder FLOPs/latency.
3. Decoder per-layer work includes text tokens, Q/K/V/output projections, MLP, attention over text+vision, and fixed kernel/launch overhead. Only some terms shrink with visual count; attention is quadratic in total sequence length, while projections/MLP are approximately linear.
4. The selector/sort/gather and cache/mask/position handling cost is included in the reported generate timing, and can consume savings.
5. Current report does not show a component-level breakdown, and `W_trace` is not proof that every real decoder block used the intended sequence length. A hook that only changes score/IDs but not actual hidden sequence/cache could produce little speedup while still reporting a theoretical W.

**Do not claim speedup until actual per-block sequence lengths and the timing boundaries are verified.** The reported `fixed32_wall_median_ms≈1.29 s` appears to be full generation, while `TTFT≈0.12 s` is a different metric. Never compare those as if they were the same interval.

### Required profiler instrumentation

For each fixed prompt/image, batch=1, identical output config:
- `vision_encoder_ms`
- `stage1_selector_ms`
- `projector_ms`
- `decoder_prefill_ms`
- `stage2_selector_ms` (attention extraction/scoring + sort)
- `gather_reindex_ms`
- `first_decode_step_ms`
- `full_generation_ms`
- CUDA event and wall clock for each; p50/p90 over >=100 paired samples after warmup.
Use interleaved order e.g. `A B B A`, same GPU/model/dtype/backend/context length, synchronize correctly. Report selector-only, decoder-prefill-only, full TTFT (image start to first generated token), and full generation separately. `fixed32_wall` is not TTFT.

## 6. FLOPs: required definitions

Report three different quantities, never conflate them:
1. `W = Σ_l K_l` and `1 - W/(576*32)` — visual token-layer proxy only.
2. Analytical decoder prefill FLOP bracket at real sequence length `T_l = text_len + K_l`, including Q/K/V/O projections, MLP and causal attention. State assumptions (2 FLOPs per multiply-add, actual Llama dimensions, text length).
3. Measured/profiled whole-model operations where supported by profiler. PyTorch profiler may not correctly count FlashAttention fused kernels; report missing fused ops explicitly. Also measure vision encoder, selector, gather separately.

## 7. Experiment plan (do in this order)

### Gate 0 — correctness/integration (stop if any fail)
- Run original model with pruning disabled and compare logits/predictions to existing baseline.
- Record true visual sequence length and original IDs at every decoder block. Assert counts exactly match the declared schedule.
- Assert attention softmax is over all causally visible keys; prompt's question receiver tokens and original visual positions align.
- Verify position IDs/RoPE, attention masks, cache length, prefill vs decode behavior, multi-image/crop indexing.
- Run `max_swaps=0` and verify output IDs exactly equal the historical selector.
- Confirm RDRIFT source, formula, window, direction and tie handling; don't substitute a guessed formula.

### Gate 1 — selector ablation (fixed Stage1, schedule, W)
Run on same paired development panel:
A. Q-Attn Top-K (current baseline)
B. real historical RDRIFT Top-K
C. Q-Attn + ICER, `drift_weight=0`
D. Q-Attn + ICER, early-boundary drift donor prior
E. RDRIFT Top-K + ICER
F. singleton perturbation ranking (CoRePrune-style diagnostic)
G. bounded boundary-only set refinement (same donor/drop candidate cap)
H. random replacement control matched to number of swaps
Start with 500-1,000 image-grouped dev questions, then full 9 tasks only for finalists. Report per-task metrics, R4 and nine-task retention separately.

### Gate 2 — component ablations
- `max_swaps ∈ {0,1,3,6}`
- donor/drop pools `(8,8), (16,8), (24,16)`
- coverage only vs coverage + positive drift
- final boundary drift on/off (expected default off)
- relevance guard `{0.95,0.97,0.99}`
- uniform vs query-weighted coverage
- descriptor raw/normalized/centered
- optional CoRePrune singleton / set-distortion selector
Keep same W, IDs where possible, and identical pruning boundaries.

### Gate 3 — causal lifetime/survival swap
On held-out dev images, keep identical entry pool, final survivor IDs, pruning boundaries, K at every stage and exact W. Change only which extra tokens live through the intermediate stage. Compare paired answer correctness, target answer NLL, next-token KL and question-state drift. Record per-image paired win/tie/loss. Do not call this proof of future utility unless the proxy predicts the sign of held-out interventions.

### Gate 4 — schedule interaction (only after selector has signal)
Compare the best selector across current anchor, wide-short, narrow-long and STAR-Pro's exact published schedule; each has exact matched W where possible. Keep this distinct from selector-only ablation so that timing effects aren't attributed to the score.

### Gate 5 — full benchmark + efficiency
- Run all 9 tasks on final candidate and strongest fixed baseline with frozen prompts/scorers. Preserve raw prediction files.
- Dense baselines for ScienceQA/AI2D/TextVQA need same-protocol local reruns before using their relative retention; the earlier provisional R8 numbers used mixed literature and local denominators.
- Run >=100 paired samples after warmup on a dedicated GPU, A/B/B/A order. Report p50/p90 wall and CUDA timings, component breakdown, memory, W, analytical FLOP bracket, profiler FLOPs with fused-kernel caveat.
- Include TextVQA-OCR as a diagnostic even if the headline eight-task set excludes it; don't cherry-pick only high-retention tasks.

## 8. Suggested order and decision gates

1. First, baseline integration parity and true trace.
2. Then implement Q-Attn + ICER as an optional branch, `max_swaps=0` parity first.
3. Run the selector ablation on a small paired panel.
4. If accuracy gain is <0.2 R4 or unstable and latency overhead exceeds saved prefill time, keep Q-Attn/RDRIFT and focus on measurement/story rather than more score engineering.
5. If gains are stable, tune only the bounded few hyperparameters on development split; freeze config before full benchmark.
6. Only if survival-swap interventions show meaningful paired gaps, make "lifetime allocation" the main causal claim. Otherwise describe the method more modestly as interaction-conditioned evidence refinement.

## 9. Files in this bundle

- `bela/stage2_icer.py`: independent selector prototype + singleton/set perturbation diagnostic.
- `bela/latency_audit.py`: timing utility helpers to insert into actual model stages.
- `tests/test_stage2_icer.py`: unit tests for core invariants.
- `experiment_matrix.csv`: planned ablations and gates.
- `CURSOR_HANDOFF_CN.md`: detailed integration instructions.

## 10. Current known limitations

- No production LLaVA model repository was included in the current turn; therefore no exact `llava_arch.py`/decoder hook diff can responsibly be generated here.
- No GPU checkpoint was executed by this package generation. It does not report new accuracy, FLOPs or latency results.
- CoRePrune helper functions are independent equation-level reimplementations, not official author code.
