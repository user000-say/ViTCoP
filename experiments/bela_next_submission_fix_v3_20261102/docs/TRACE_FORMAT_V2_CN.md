# BELA NeXT 真实 trace JSONL 合约 — V2

本合约只约束诊断证据，不要求把实验实现重写成这里的类。每一行 **一个真实图像+问题+一次运行**。不含图像、模型权重和完整答案。为省体积，仅选代表性样本，但选定样本每步索引需要完整记录。`SYNTHETIC_TRACE_EXAMPLE.jsonl` 只作格式开发参照，`real_checkpoint_trace=false`，不得拿它当验收。

重要说明：`bela_trace_gate_v2.py` 是一个**证据一致性和边界校验器**，不是模型正确性证明；它不会自动读取线上进程，不会自动知道哪份 UID reference 可信。独立 reference 必须从未改动 Native NeXT Dense 采样，且附来源函数/commit/hash。对照当前 BELA 的实际 Crop 与 ImageProcessor 函数路径，不能仅拿 BELA 自己重算的 UID 当外部真值。

## 必需公共字段

- `schema`: 固定 `BELA_NEXT_TRACE_V2`
- `sample_id, run_id, checkpoint, source_sha256`；`real_checkpoint_trace`: boolean。真实样本还需要单独提交 `git_sha, processor_sha, model_config_sha, runner_sha, precision, attn_implementation, input_sample_hash`。
- `image`: `orig_size_hw:[H,W]`, `num_crops:int`, `patches_per_crop:[int,...]`。实际还须记录 `grid_hw`, `pixel_values.shape`, `vision_token_count_before_unpad`, `native_pack_len`, `newline_count`, `thumbnail_features_len`。
- `native_pack`: 原生 Native Dense *映射参考* 的 UID 列表，每个成员包括 `uid, kind, packed_index`。对 `patch/base_patch` 再加 `crop_id, local_patch_index, valid=true`；`kind` 可为 `patch/base_patch/newline/special/text`。
- `native_reference_uids`: **独立原生实现**得到的 UID 顺序，与 `native_pack` 按 packed_index 排序比对。`native_reference_provenance`: 实际原生函数、版本、SHA、输入 sample hash 和比较方法。若缺少它，校验器警告，审计不得把 D18/D25 判 PASS。

说明：`crop_local_patch_index` 为送入 CLIP 时该 Crop 内的原始 patch；`native_pack.packed_index` 则是 AnyRes 的 unpad/repack 后的 index，二者不可混用。`newline` 属于结构 token，不允许声明成 raw patch。在有额外图文文本 token 的 LLM 序列中，应在 `coordinate_checks` 中另给 `native_packed_pos → llm_seq_pos` 的映射。

## Stage1 + QA

```json
"stage1": {
  "eligible_uids": ["c0p0", "c0p1", "c1p0"],
  "before_qa_uids": ["c0p0", "c1p0"],
  "after_qa_uids": ["c0p1", "c1p0"],
  "k_global": 2,
  "crop_quotas": [1,1],
  "qa": {"enabled":true,"max_swaps_total":8,"swaps_attempted":1,
         "query_provenance":"Q hidden state / text embedding actual source",
         "accepted_swaps":[{"drop_uid":"c0p0","donor_uid":"c0p1"}]}
}
```

真实 trace 另加：`coverage_slots_global`, `coverage_slots_per_crop`, `cover_layer_ids`, `before/after_cover_scores`, `donor_pool`, `drop_pool`, `gain`, `novelty`, `coverage_loss`, `accepted/rejected_reason`。注意如果 QA 有意不接受交换，`accepted_swaps=[]` 合法，但需证明 `qa.enabled=true` 的代码确实运行。

## 最重要：所有索引都带 coordinate space 与独立参考

```json
"gather_events": [{
  "name":"layer11_visual_gather",
  "input_uids":["c0p0","c1p0","c1p1"],
  "indices":[0,2],
  "output_uids":["c0p0","c1p1"],
  "expected_uids":["c0p0","c1p1"]
}],
"coordinate_checks": [{
  "name":"crop1_local_to_native_pack",
  "reference_space_uids":["c1p0","c1p1","c1p2"],
  "indices":[0,2],
  "observed_uids":["c1p0","c1p2"]
}]
```

`gather_events` 判断 `output_uids` 是否与 **当次实际输入坐标系**用 `indices` 取出的 UID 一致；`coordinate_checks` 必须由**独立参考映射**提供 `reference_space_uids`，检查越界及“整数有效但错 Patch”。随事件提交 `source_file, function, line, in_shape, out_shape, coordinate_space_name, dtype, device`，且将 `torch.gather`, `index_select`, `nonzero`, `topk`, `permute`, `reshape`, `torch.cat` 导致的坐标转换逐项记录。

例如：同一个数字 `137` 在 raw crop、unpad 后、native pack、LLM merged sequence、L20 survivors **代表完全不同的元素**；不允许直接延用整数值而不转换 UID。

## Stage2、RoPE 与 KV

```json
"expected_prune_layers": [11,19],
"layer_prunes": [{
  "layer_idx":11,
  "input_visual_uids":["c0p1","c1p0"],
  "output_visual_uids":["c1p0"],
  "target_k":1,
  "sequence_length":100,
  "query_indices":[97,98,99],
  "visual_positions":[51,52],
  "score_source":"actual_post_rope_qk_with_mask"
}],
"position_tracking": [{"uid":"c1p0","before":52,"after":52,"policy":"preserve"}],
"cache_events": [{"layer_idx":0,"decode_step":0,"key_len":100,"value_len":100}]
```

- `layer_idx` 明确以真实代码的 0-based decoder block 序号为准，`11/19` 是 W1024 等日程的一个具体实现假设，需要核实是否是对应 block 计算后剪。不能因工具示例直接改变现役代码事件。
- `input_visual_uids` 与上一阶段的 `output_visual_uids` 必须集合一致，后剪只从前剪 survivors 选择，不能复活 UID。
- `query_indices` 为当前 Decoder **实际 Query 所在 token 的位置**，与 `visual_positions` 不重叠，且小于 `sequence_length`。实际还要记录 `all_text_UIDs`, `question_text_UIDs`, `attention_mask`, `head_count`, `num_key_value_heads`, `scale`, `observed scores`, `topk` 排序和 tie-break。
- `position_tracking.policy=preserve` 时，before 和 after 必须相等；如果有意识 renumber，报告 WARNING，另交 native/adapter 的 Attention 和生成数值测试，不能直接 PASS。
- `cache_events` 校验同层 K/V 长度、同一层逐 decode step 递增或解释 cache eviction；**不同层缓存长度不要求一样**。真实验收必须再检查第一和第二次生成的每层 `cache_position` / `past_key_values` 以及 decode logits。

## 使用

```bash
python bela_trace_gate_v2.py EVIDENCE/real_next_samples.jsonl \
  --out EVIDENCE/trace_gate_report.json
```

`PASS_INVARIANTS_ONLY` 只能表示该 JSONL 自洽。最终门槛 D01–D52 的 PASS 仍要人工比对真运行源码、原生 reference、numerical oracle 与真实模型数据。
