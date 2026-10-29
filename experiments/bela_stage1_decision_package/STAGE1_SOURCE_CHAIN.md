# STAGE1_SOURCE_CHAIN — P0

```
e014_boot.py
  → e006_lmms_patch.install / generate_until
      → (QA on) _begin_qa_request + explicit_span + content mask  [ContextVar]
  → LlavaLlamaForCausalLM.generate
      → vitcop_prepare_inputs_labels_for_multimodal
          → encode_images
              → clip_encoder ELVA_R3 branch
                  → llava_stage1.select_existing_forward
                      → select_checked + native_cover.legacy_callback (e013 COVER)
                      → optional _apply_qa_cover → apply_llava_query_anchor
                      → LAST_ENTRY_KEEP / LAST_ENTRY_METRIC
                      → pen.index_select(keep)  # physical Stage1 gather
  → E018 Stage2 RDRIFT/Q_Attn on decoder events (arm overlay)
```

## Mapping to decision-kit arms

| Kit arm | Production realization | Disk status (2026-10-29) |
|---|---|---|
| S0 Native Dual COVER B0 | `L2_NATIVE` / `L2_LATE_64_17_10` | GQA+POPE (+ L2_LATE full extras) |
| S1 COVER→QA B8 | `S1_QANCHOR_L2` | **six tasks complete** |
| S6 matched-random | `S1_MATCHED_RANDOM` | GQA+POPE |
| S7 wrong-video/query shuffle | `S1_QUERY_SHUFFLE` | GQA only |
| S2 salient→QA | needs `BELA_STAGE1_INIT=salient_topk` | **to run** |
| S3/S4 joint | `joint_baseline` + host hook | **to run** (surrogate, labeled) |
| S5 query-only Top-K | `BELA_STAGE1_INIT=query_topk` | **to run** |
| S8 ablations | env threshold overrides | **to run** |
| P1 B-scan | `BELA_QA_MAX_SWAPS∈{0,4,8,12,16,24}` | B8 done; others **to run** |

## Stop conditions already checked

- Grid size locked to 576 for QA path.
- QA refuses stacking with `admission_cap>0`.
- `off` mode skips question attach and QA swap (`LIVE_CALLMAP.md`).
- Historical S1_QANCHOR_L2 metrics exist — do not regenerate unless scorer/commit changes.
