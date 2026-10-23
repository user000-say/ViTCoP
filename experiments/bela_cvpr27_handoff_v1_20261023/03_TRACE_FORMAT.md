# 严格只读剪枝 Trace / Prediction 合同

每个样本 `TRACE.jsonl` 必须是一行 JSON，至少含：

- `run_id`：git sha + config hash + checkpoint id；`sample_id`、`image_id`：任务内稳定标识。
- `tokens_per_layer`：**实际** 32 个 decoder 层上的活跃视觉 Token 数量，不能用配置推测。
- `cut_events`：每次视觉容量发生变化的事件，含 `after_layer_index`（零基，第 4 层后裁剪=3）、`patch_ids_before`、`patch_ids_after`、`selector_module`、`selector_function`、`selector_callsite`。
- Patch IDs 全部为原图视觉 Patch ID，不能把当前压缩序列的局部索引当作原始 ID。
- 第一次后保留的原始集合必须与第二次前的集合一致（允许顺序变，但不得恢复已裁掉 Patch）。最终 `after` 必须是 `before` 子集且无重复。

`python scripts/audit.py trace outputs/TRACE.jsonl --arm A2 --schedule config/schedules.json` 会对人数、W、每一层的 K、裁剪层位、patch 身份来源做严格检测。**但该 CPU 脚本不能证明 CUDA KV-cache / mask / position 正确性**：必须在 `CUT_CALL_MAP.md` 由 Cursor 按真实 forward 路径人工核对并附 3 个 smoke 样本的 log。

### Paired result JSONL

从**原始官方评测器**导出：每题 `sample_id`、`image_id`、`correct` (JSON Boolean)。本包绝不自建 TextVQA/GQA 评价器。POPE 还要 `gold_binary`、`pred_binary`（`yes`/`no`）和 `subset`（`random`、`popular`、`adversarial`）。样本 ID 或 image ID 不匹配时脚本必须拒绝比较，不静默删样本。

`paired.py` 的 POPE micro 值用于诊断；如果官方方法采用不同子集汇总算法，仍以官方评价器的正式 F1 为主，并在产出中写明定义差异。
