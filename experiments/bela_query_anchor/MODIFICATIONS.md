# QA-COVER modifications (2026-10-27)

The selector in `query_anchor.py` and `llava_adapter.py` is the package as delivered. Its swap rule, thresholds, fixed-K sort, flat-query fallback, and matched-random control were not retuned.

## What the package already did

- Native keep is an input. The operator never chooses the first 64.
- At most `max_swaps` outsiders replace low-query, redundant insiders.
- A flat query (`std < min_raw_std`) returns the native ids.
- Coverage loss, query gain, and novelty are gates. Saliency guard defaults to 0.
- `off` returns the sorted native ids and does not read the question.
- `matched_random` keeps the native removals and redraws only the additions.

## What was wrong relative to the live model

The package could not see a question. Stage1 runs inside the CLIP tower, and `encode_images` runs before `vitcop_prepare_inputs_labels_for_multimodal` splits the prompt. A global `LAST_QUESTION` would also leak across requests. The handoff forbids that.

## Edits

1. `request_context.py`: one `ContextVar` per forward. Re-entry raises. `end_question` resets it.
2. `question_protocol.py`: frozen template-piece mask (`BELA_QA_CONTENT_MASK=1`). It drops function words and punctuation from the tokenizer decode. It does not read answers or object labels. This is the recommended mask in the handoff, not a new score.
3. `e006_lmms_patch.py`: before `generate`, take question ids from `explicit_span`. Shuffle mode loads another image's question through `make_shuffle_map` (seed 17) and refuses a donor from the same `image_source` / `imageId`.
4. `vitcop_llava_arch.py:encode_images`: while the mode is on, pass the live projector and token embedding into the slot around the vision-tower call only. `off` leaves this function on the old two-line path. The embedding table is passed as a module; calling it would run `Embedding.forward` with no ids. Smoke on 2026-10-27 hit that and the call parentheses were removed.
5. `llava_stage1.py`: after Dual COVER and before the penultimate gather, call the adapter. Update `LAST_ENTRY_KEEP`. Keep `LAST_ENTRY_METRIC` as the full grid. Refuse `admission_cap > 0` in the same forward. `off` does not call the adapter.
6. `e018_stage2_support.py`: when `BELA_QA_TRACE=True`, append `qa_survival.jsonl`. No keep/drop decision changed.
7. `e014_boot.py` and `e010_common.py`: copy `vitcop_llava_arch.py` into the eval runtime. Without that, the eval process would keep the old `encode_images`.

## Not changed

- Dual COVER callback, RDRIFT, Q-Attn, event layers, K schedule, and the QCP operator.
- Anchor thresholds (`max_swaps=8` and the rest of `S1_QANCHOR_L2.json`).

## Smoke, 2 GQA samples, 2026-10-27

`BELA_QA_ENTRY_MODE=off` reproduced the stored L2 answers on doc `201307251` (`No`) and `201640614` (`Woman`). Audit: stage1 64, events `[[11,17],[19,10]]`, final 10, no QA trace.

With the anchor on, both samples swapped 8 ids and still answered the same two strings. After the layer-11 cut to 17, 5/8 and 1/8 of the new ids were alive. After the layer-19 cut to 10, 1/8 and 0/8 remained. That is the failure mode the handoff named: Stage2 can delete the new entry evidence. It is not a G1 result.

Traces now carry `sample_id` and `image_id` so later rows can be joined to official samples.

## Known limit

Projector cosine versus question-token embeddings is still an unverified alignment. The shuffle and matched-random arms exist to test that. CPU tests do not establish accuracy.
