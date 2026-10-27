# QA-COVER live call map (2026-10-27)

Verified against the current LLaVA eval path, not the handoff pseudocode.

1. `e010_eval_job.py` loads the runtime tree and calls `e006_lmms_patch.install`.
2. `e006_lmms_patch._generate_until` builds the llava_v1 prompt. When `BELA_QA_ENTRY_MODE` is `off`, this function does not read the question and does not set a request slot.
3. When the mode is `s1_anchor` or `matched_random`, `_begin_qa_request` calls the existing `e006_span.explicit_span`. The ids are the decoded user question, excluding the image placeholder (`-200`), the system prefix, and the `ASSISTANT:` tail. A frozen template-piece mask then drops function words. The slot is a `ContextVar`; a second begin on the same context raises.
4. `LlavaLlamaForCausalLM.generate` calls `vitcop_prepare_inputs_labels_for_multimodal`, which calls `encode_images` before the prompt is split.
5. `encode_images` (`LLaVA/llava/model/vitcop_llava_arch.py`) attaches the live `mm_projector` and `embed_tokens` to that same slot, then calls the vision tower. The modules are detached before the normal K-token projection, so the LLM still receives `mm_projector(penultimate[keep])`.
6. `clip_encoder.py` ELVA branch calls `experiments/elva_r3/llava_stage1.py:select_existing_forward`. Native Dual COVER still runs first through `select_checked` / `native_cover.legacy_callback`. Shapes at that point: penultimate rows `[576, 1024]`, Dual metric `[576, D]`, keep `[64]`.
7. `_apply_qa_cover` runs only if the mode is on. It calls `apply_llava_query_anchor`. The extra projector sees the full 576 penultimate rows and is timed separately. Returned ids stay 64 unique original ids in ascending order. `LAST_ENTRY_KEEP` is overwritten with those ids. `LAST_ENTRY_METRIC` stays the full 576 Dual descriptor.
8. The vision tower gathers `tokens_0.index_select(keep)` and `publish_stage1(final_idx)` publishes the same ids. Stage2 RDRIFT is unchanged: events `11:17,19:10`, W = 64×12+17×8+10×12 = 1024.
9. At the end of E018 prefill, `qa_survival.jsonl` records which swapped-in ids are still alive after each cut. This log does not change the selector.

`off` never enters steps 3, 5's attach, or 7. QCP and bounded admission are refused / forced off for these arms.
