# NeXT-7B P0 source audit return (2026-11-02)

Read-only collection. **No hot-fix / no new ACC.** GPU 4–7 workers were not stopped.

## Active call map (s2qk / BELA_NEXT_V2)

1. `scripts/run_next7_s2qk_gated.sh` → `scripts/lmms_bela_next_entry.py` (lmms-eval `llava_hf`)
2. Entry sets `BELA_NEXT_V2_ENABLE=1`, `BELA_NEXT_ARM=…`, loads HF `LlavaNextForConditionalGeneration`
3. `bela_next_forward.apply_llava_next.apply_patch(model)` monkey-patches:
   - `llava.get_image_features` (stock Dense pack only; no prune)
   - `llava.forward` → ENABLE=0 stock; ENABLE=1 prefill → `run_native_stage1` then `_llama_forward_with_cuts`
   - `model.generate` → custom prefill+decode loop when ENABLE=1
4. Stage1: `native_stage1.run_native_stage1`
   - vision_tower `output_hidden_states=True` → projector on `config.vision_feature_layer`
   - `pack_image_features` (HF spatial_unpad+newline) + `native_layout.map_native_packing` / `pack_dual_depth`
   - Dual: `bela_next_compat.original_dual` → `dual_cover.dual_cover_one_crop` → `e013_stage1_support`
   - QA: `bela_next_compat.original_qa` → `bela_query_anchor.llava_adapter.apply_llava_query_anchor` → `query_anchor.select_question_anchored`
5. Stage2: after decoder layer L12/L20 (1-based): `stage2.select_and_prune` with **Q/K from layer INPUT** (current tree after 2026-11-02 fix); physical gather + DynamicCache prune
6. Scorer: lmms-eval task metrics; gate `scripts/gate_next7_arm_r4.py` vs `next7_dense6_absolute.json`

## Legacy W1024 call map (1.5-7B)

- Overlay: `experiments/elva_r3/apply_llava.py` + `llava_stage1.py` / `native_cover.py`
- Dual COVER: `e013_stage1_support` + CLIP attentions for saliency (production path)
- QA: `bela_query_anchor` + `e006_span.explicit_span` question slice
- Stage2: `LLaVA/llava/model/e018_stage2_support.py` + hooks in `vitcop_modeling_llama.py` (`_e018_callback` on post-RoPE Q/K)
- Configs: `P3_QA_QATTN.json` (both cuts Q_ATTN); `S1_QANCHOR_L2.json` (first RDRIFT, later Q_ATTN)

## Package layout

- `sources/NEXT/` — files from NEXT_FILES.txt
- `sources/LEGACY/` — W1024 historical sources
- `sources/RUNTIME/` — `inspect.getfile` transformers 5.20.2 snapshots (not monkey-patched; NeXT hooks live in apply_llava_next)
- `docs/` — this callmap, audit table, proven defects, run manifest
