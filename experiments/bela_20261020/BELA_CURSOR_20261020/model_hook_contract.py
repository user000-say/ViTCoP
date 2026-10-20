"""Model hook CONTRACT example ONLY. Not a ready LLaVA monkey patch.

Integrate in repository-owned block loop, not by replacing forward wholesale.

state = LifetimeState(plan, patch_ids_from_stage1)
for idx, decoder_block in enumerate(model.layers):
    block_no = idx + 1
    hidden_states = decoder_block(...existing model args...)
    state.record_block(block_no, actual_visual_count)
    if block_no in {e.after_block for e in plan.events}:
        scores = capture_existing_score(... currently aligned active visual IDs ...)
        keep_slots, kept_original_ids = state.prune(block_no, scores)
        # REQUIRED: invoke repository's own prune_and_gather on actual tensor state:
        # hidden_states, visual_token_positions, attention_mask, position_ids,
        # past_key_values = repository_prune_and_gather(..., keep_slots, ...)
        # Assert physically present visual token count == len(kept_original_ids).
audit = state.finish()
"""
