# Stop / Go：数据决定下一处代码修改

| 实验观察 | 决策 |
|---|---|
| P0 无法确认真正在用哪个 score 函数，patch ID/KV 不匹配 | STOP 所有新算法，先修调用审计 |
| B2 恢复 GQA/TextVQA 且 POPE 基本保留，额外预算开销可接受 | GO_C1_DESIGN；先提交支持 3 次剪枝的最小 diff 和缓存测试 |
| B1 独立提升 POPE 但 FP 增加 | GO_POPE_DIAG；不能直接解释成减少幻觉 |
| B2 无法恢复而固定 K20 出现可解释的身份错误 | GO_IDENTITY_ABLATION；现役 selector / Q-Attn / Random 多seed |
| B1/B2 都没有稳定的逐样本改善 | STOP_SCHEDULE_SEARCH；回查 Stage1 COVER / 第一次剪枝 |
| 新方法略增 R4，但 GQA/OCR 下降或实际 TTFT 明显恶化 | REJECT 新模块 |

不可通过选择最有利指标来改 GO 标准。下一版竞争力努力目标：GQA 尽量接近 59、POPE 稳在 85 以上、TextVQA 至少恢复至 A0 附近，并具有真实 latency-Pareto 改善；这些是**研究目标而非承诺结果**。

STAR-Pro 公告结果只能标 PAPER_REPORTED，直至同机协议验证。混合 MME Perception 与 Total 或混用不同 POPE F1 聚合口径时不做排名。
