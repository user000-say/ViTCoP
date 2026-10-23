# BELA / ELVA — Cursor 唯一有效交接（V1.0，2026-10-23）

> **执行目标：现役 BELA 的可复核工程审计 + B1/B2 机制诊断 + 真实效率。不是第三套新模块。** 历史 Evidence Guard、Candidate-risk Probe / gate.py / selection.py 代码均视为未验证候选，禁止混用或直接接入生产。

## 首批任务（不得跳关）
1. 对现役源码 `git rev-parse HEAD`、`git status --short`，冻结 checkpoint/processor/eval 命令、数据集版本及 scorer SHA；填写 `templates/BASELINE_LOCK.md`。
2. 追踪 **真实两次剪枝事件**：调用路径、`score_fn`、用的是 Q-Attn / RDRIFT / 其他，score 从哪个 layer 获得，question-token aggregation，patch ID、position/mask/KV cache 的 gather。填写 `templates/CUT_CALL_MAP.md`。**不允许按历史讨论猜 RDRIFT 运行位置**。
3. 用默认关闭的只读 trace 开关导出 A2 3 个 smoke 样本、真实 `tokens_per_layer` 及每次剪枝的原始 Patch IDs。核对“开/关 trace”不改变模型输出；用 `scripts/audit.py` 验证。
4. 同 GPU、同 checkpoint、同 prefill 和固定 32-token 输出协议，对 A0/A2/A3 做重复 TTFT、Fixed32、selector 耗时与显存测量。名义 W2048 不等于延迟相同。
5. 只有 P0 通过，才运行 B1 172/90/20 (W1808) 和 B2 172/120/32 (W2288) 的 **GQA、POPE、TextVQA**，输出逐样本预测与配对统计；这些不是等预算优劣排名。
6. 根据 `02_RUNBOOK.md` 的决策树只选择下一步一个方向：C1 生命周期实验或同 K20 的身份选择实验。先交最小 diff 和停止标准，不直接上线复杂模块。

## 交付回执
`outputs/BASELINE_LOCK.md`, `CUT_CALL_MAP.md`, `TRACE.jsonl`, `RESULTS.csv`, `POPE_DIAG.json`, `PAIR_DIAG.json`, `LATENCY.csv`, `SOURCE_DIFF.patch`, `STOP_GO.md`。不存在的结果标 `NOT_RUN`；对话历史分数标 `USER_REPORTED`；外部论文标 `PAPER_REPORTED`。

## 本包离线工具（不依赖 PyTorch）
```bash
python -m unittest discover -s tests -v
python scripts/audit.py schedules config/schedules.json
# 真实trace导出后
python scripts/audit.py trace outputs/TRACE.jsonl --arm A2 --schedule config/schedules.json
# 用官方评价器提供的 correct 和真实样本 ID，不重写打分规则
python scripts/paired.py outputs/A0_GQA.jsonl outputs/A2_GQA.jsonl
python scripts/paired.py outputs/A0_POPE.jsonl outputs/A2_POPE.jsonl --pope
```

**阻断条件：** 现役剪枝调用不明，Patch ID/实际 W 不一致，评测版本不一或没有可复核的真实时间数据时，不宣称算法成功。对任何 Agent 建议都不默认接受。
