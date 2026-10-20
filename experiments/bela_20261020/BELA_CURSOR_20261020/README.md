# BELA 2026-10-20 代码 + Cursor Handoff

这是**可运行的独立算法核心与实验协议**，不是声称已经接好生产模型的完整 LLaVA repo patch。

## 快速启动
```bash
python -m unittest discover -s tests -v
python -m bela.cli --target 2048 --entry 172 --boundaries 4,12 --final 32
python -m bela.cli --target 1024 --entry 60 --boundaries 3,13 --final 16
python -m compileall -q bela tests
```

## 本版相比 E020.4 实际改动
- 新增 `bela/core.py`：严格整数预算枚举；按原始 patch ID 稳定 deterministic top-K；追踪各 block 实际 visual token；same-budget survival swap 生成器。
- 新增 `bela/tensor_adapter.py`：PyTorch 真实 sequence 维度 index_select（可处理非连续视觉位置，仍由模型负责 mask/KV/position）。
- 新增 CLI、单元测试、模型接线合同、Cursor 实验说明和 run manifest。
- 完整保留旧 E020.4 文件在 `legacy_e020_4`，**没有修改其算法或将它自动装进新主方法**。

## 重要约束
1. 系统级 Stage1 双层 ViT/COVER 与 Stage2 真实 decoder hook 尚需原 VLM 仓库实现/审计。新增代码不能独立生成答案。
2. `lifetime_scores` 是明确标注的、线性 cheap signal 代理，并不是对“未来答案收益”的已训练或验证估计器。
3. 本地单测不等于真实模型准确率、GPU FLOPs、TTFT 或 ablation 有正结果。
4. 端到端速度测试要同时计 selector overhead 和 decoder 真实耗时。
5. 先将原 RDRIFT/REF 正确接入匹配 baseline；再研究未来生存贡献。

请先阅读 `CURSOR_HANDOFF_CN.md`。
