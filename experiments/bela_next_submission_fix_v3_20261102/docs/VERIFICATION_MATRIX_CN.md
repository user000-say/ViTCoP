# 可验证性矩阵与交付边界

| 事项 | 当前状态 | 合格证据 |
|---|---|---|
| 源码导入和语法 | CPU PASS | `PATCHES/LOCAL_VERIFICATION_EVIDENCE.json` |
| 核心索引、全局 slots、QA、QK 数值合成测试 | CPU PASS，33 项 | 同上 + `tests/` |
| 预算 W5120（A0）算术 | CPU PASS | `next_configs/arms.json` 和 preflight |
| 固定五 Crop 与 tokenizer 同源硬失败 | CPU PASS | `test_index_cache_and_generation.py` |
| 与用户 GPU 的 transformers 5.20.2 源文件 SHA256 一致 | GPU 环境 UNKNOWN | `preflight.py --runtime` 日志 |
| 原生 Dense logits 对齐（开启/关闭补丁） | UNKNOWN | 同 checkpoint/inputs 的 logits maxdiff 与生成序列 |
| 原生 AnyRes 全量 UID/packing/unpad 等价 | UNKNOWN | 至少 10 幅多高宽比、完整 original UID/position trace |
| 真实 penultimate CLS Q/K vs eager | UNKNOWN | eager attention 数值 oracle |
| 真实 L12/L20 Q_ATTN vs causal attention | UNKNOWN | GQA, KV, padding, layer 序列数值 oracle |
| 精度 R4 与六任务 mean6 | UNKNOWN | 六任务绝对、相对 Dense、CI、split/scorer |
| 论文可复现与加速数字 | UNKNOWN | 多 seed、显存、latency、正确 token-layer 计数 |

只有 CPU PASS 的项目可以称为已修复并通过本地检验。GPU UNKNOWN 不能称为已修复有效或达到 95%。
