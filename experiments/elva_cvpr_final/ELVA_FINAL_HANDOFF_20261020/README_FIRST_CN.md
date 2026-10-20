# ELVA_HANDOFF_20261020 — 先读这里

这是针对 2026-10-20 最新 R3 W2048 / W1024 分数修订的**最终收敛实验+代码交接包**。

**首先阅读 `FINAL_HANDOFF_CN.md`，实现 Agent 直接阅读 `CURSOR_AGENT_PROMPT_CN.md`。**

## 边界：不是已经打通的 LLaVA 模型

本包包含：新编写、单测通过的日程/身份/反事实代码（`elva/`），具体 `server_adapter/` 宿主前向接线合同、14条纯候选预算日程，以及完整Cursor执行书；`reference_only/` 是上传的旧E020.3和E014源码归档，只供**对照审查**，不能用旧版本覆盖昨晚 R3 的正在运行的仓库。

缺失：昨晚新 R3 V1-V5 的真实模型patch、checkpoint/服务路径、K60逐层trace、官方scorer原始日志。**不能声称本包给出了真实R3模型完整替换实现，不能声称实现了真正的GPU双分支反事实**。这是目前最小不可绕过的集成限制。

## 本机运行（Python >=3.10，从本目录）

```bash
python -m unittest discover -s tests -v
python -m elva.cli plans plans/schedules.json
python -m elva.cli official-r4 results_examples/official_scores.json
python -m elva.cli verify-trace plans/schedules.json E94_K80_48_16 results_examples/synthetic_trace_NOT_MODEL.json
python -m elva.plot_patch_maps results_examples/actual_ids.example.json /tmp/elva_demo_map.png
```

注意：示例 `official_scores.json` 是用户提供的原W2048 V5四项分数，**不是本次新实验**；`synthetic_trace_NOT_MODEL.json` 是合成正确性测试，**不是从真实模型捕获的trace**。热区示例也只是合成patch IDs。

## 真实宿主入口

- `elva/bridge.py` 的 `LivePruneCoordinator`: 当前源 forward在每个block begin/end处调用，提供真实host selector/gather回调，收集真实trace和patch ID。它**不负责**RoPE、mask、KV：这些必须调用原始模型对应函数。
- `server_adapter/integration_contract.py`: 与真实 R3 的选择器函数签名适配；还须在服务器编写实际接线diff。
- `elva/core.py` 的 `SurvivalSwap`: 实验前先检查Ga/Gb/S身份及成本。实际分支复制与真实loss只可由server runner实现。
- `elva/plot_patch_maps.py`: 从真实结果 JSON 的`pen`,`L9`,`dual` ID集合产出热点图。

## 目标与优先级

1. W2048，压缩率88.9%，当前 R4 95.90，目标 ≥97。先接原 RDRIFT，比较同W日程。
2. W1024，压缩率94.4%，当前 R4 92.71，目标 ≥94。先两组K80同W日程，失败再讨论K60/PCOV与final量。
3. 原四任务R4固定；新增ScienceQA-IMG、TextVQA、VQAv2、SEED-Bench-IMG、MM-Vet组成九项固定外部验证集；VizWiz可选。不得改变历史R4口径。
4. 截止 11/16 AoE，建议11/04–11/06冻结方法，优先真正的GPU证据与论文。

## 新增代码验证状态

11个 unittest（含CPU PyTorch的32-block模拟宿主张量操作）已跑通；E020.3原始包的41项测试本次独立复跑成功。它们验证的是合同与原外部算子**局部行为**，不是LLM准确率、真实RoPE/KV、GPU TTFT。
