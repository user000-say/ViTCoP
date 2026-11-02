# Cursor 逐项工程审计回报

| 检查项 | 状态 | 真正运行的文件:函数 | run_id / git_sha | 对应证据 | 发现/修复建议 |
|---|---|---|---|---|---|
| E01 Dense 等价 | FAIL | `apply_llava_next.apply_patch` / hooked_generate | smoke `logs/smoke_a0_a4/SMOKE_VERDICT.md` gen_parity=false; git 见 01_RUN_MANIFEST | ENABLE=0 prefill_parity true but gen_parity false | 查 generate 旁路；未在本轮热修 |
| E02 AnyRes/crop 协议 | PASS | apply_llava_next 注释+entry：不强制 pinpoints；smoke crop_n=5 | smoke A0 | 与 Dense processor 一致（当前树） | 保持 |
| E03 spatial_unpad/newline | PASS | `native_stage1`→HF `pack_image_features`；`native_layout` | smoke pack_mode=spatial_unpad_newline | 非 simple_crop_concat | 保持 |
| E04 CLIP 双层与 Projector | UNKNOWN→provisional | `native_stage1` mid_layer=9, pen=-2；projector on config.vision_feature_layer | mid=9 per ELVA hs[9]=block9 | 未对照 1.5 CLS-attn saliency | Dual scores 用 pen L2 非 attentions（见 E05） |
| E05 Native COVER/slots | FAIL (pre-fix) / patched | `bela_next_compat.original_dual` | smoke slots_per_crop [6,6,6,7,7] after fix; previously 32×N crops | 历史每 crop slots=32 膨胀 COVER | 已改全局 slots 均分；未证明 ACC 提升 |
| E06 QA B8 | FAIL→partial | `original_qa`→`apply_llava_query_anchor` | smoke qa_swaps=0→8 after question slice | 曾把 USER/ASSISTANT 当问题；现切片原 token | 仍缺 1.5 content_mask；metric 为 raw concat |
| E07 Q_ATTN Query/QK | FAIL→patched | `apply_llava_next._llama_forward_with_cuts` + `stage2._layer_qk` | audit 86f09ecd：曾对 post-layer 二次 probe | 现改为 layer-input Q/K；text_start 仍偏「视觉后全部」 | s2qk A0 MME/MMBench 仍 <88% → SKIP |
| E08 物理剪枝与身份 | PASS provisional | `stage2.select_and_prune` keep non_patch+topk patches | smoke events 320→91→46 | newline 经 structural_keep_mask | 空行丢 newline 与 Dense 训练分布差（已知） |
| E09 RoPE/Cache/decode | UNKNOWN | compacted position_ids + cache index_select | fidelity dense-RoPE 曾掉到 ~82 | 未做逐层 KV 证明 | 保持 compacted；勿再盲切 dense RoPE |
| E10 generate/scorer | UNKNOWN | lmms_bela_next_entry + lmms-eval scorers | same tasks as Dense refs in next7_dense6 | 无 paired Dense regenerate 本轮 | NOT_YET_RUN 成对 Dense |

**本轮不热修、不开新超参。** PROVEN_DEFECTS / PROPOSED_FIX 见同目录。
