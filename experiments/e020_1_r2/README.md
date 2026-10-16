# E020.1 R2 — 同预算集合交换的收敛验证包

这是对 E020.1 的代码审查修订，不是 E021，也没有新增 GPU 结果。
先读 `CURSOR_E020_1_R2_CN.md` 和 `RESEARCH_DECISION_R2_CN.md`。

## 主入口

`close_core.py`：真实标签 → current 权重与验证集阈值 → 核心 ACC → 确认与 TTFT。
核心为12个因子配置加最多4个强工作点诊断配置；full 历史头是可选消融。
`build_matrix.py` / `workflow.py` 保留历史兼容能力，不作为 R2 默认执行入口。

## 模块

- `features.py`：prompt-only v2 特征；current/no-context/shuffled 对照对齐。
- `exchange_numpy.py`, `module_torch.py`, `dispatch.py`：同K、同层位，一次最多交换1个patch；16动作确定性候选目录。
- `policy_protocol.py`, `policy_calibration.py`：候选契约、完整状态身份、验证集选择最大值后的接受阈值。
- `pairdata.py`, `replay_adapter.py`：真实配对分支、重算no-op、答案因果行对齐、可选endpoint机制诊断。
- `scorer_numpy.py`, `fit_models.py`, `diagnostics.py`：拟合、图像隔离、候选集收益与策略遗憾。
- `core_matrix.py`, `core_progress.py`, `run_matrix.py`：权重身份、依赖、断点恢复与证据状态。
- `factorial_interaction.py`, `audit_mme.py`, `paired_stats.py`：入口×日程交互、任务归因与图像配对统计。
- `measurement_anchors.py`：按实际原路径接入S0和外部方法；不凭W反造配置。

## 本地验证

```bash
python3 -m unittest checks checks_closure checks_r2 -v
```

所有本地生成标签都是临时合成夹具；不会作为VLM有效性证据。
真实teacher-forced后端、原模型和官方评分器不在当前工作区，必须在服务器接线。
`VALIDATION_LOCAL.json`记录检查结果及明确未测项。

旧v1权重不能改名复用：特征消融、候选搜索和策略校准语义已经变更。
