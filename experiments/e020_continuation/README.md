# E020 Continuation Exchange

先读 `CURSOR_HANDOFF_CN.md` 和 `RESEARCH_DECISION_CN.md`。

- 这是新的可移植算法模块及实进程调度工具，不是对未提供项目仓库的伪装补丁。
- 已有最强报告结果：COVER FIRST_SHIFT R4=96.50；本包没有新GPU ACC/TTFT。
- `module_torch.py` / `dispatch.py`：真实patch、固定K、首event交换；线上无答案/未来分支。
- `pairdata.py`：实际相同前缀的任务损失标签；需接项目prefix clone/suffix接口。
- `replay_adapter.py`：无mid-layer resume时，复用现有完整forward收集同前缀标签；含标准teacher-forced NLL。
- `diagnostics.py`：独立图像上的单步真实收益与采样候选headroom，不冒充ACC或多轮oracle。
- `scorer_numpy.py` / `fit_models.py`：1664参数小头、按图像隔离、同容量控制。
- `budget.py`：完成block数与零起算index明确转换。
- `build_matrix.py` / `workflow.py` / `run_matrix.py`：72配置容量，四任务筛查、确认、专卡测速、超时/重试/续跑。
- `paired_stats.py`：官方指标口径的配对图像簇统计。
- `runtime.example.json` 和 `bridge.example.json` 为故意不可执行的模板，必须填真实来源/路径/命令。
- `CONFIGURATION_CATALOG.json` 是实验计划，不是实验结果。ONE未导出时自动少2配置。

本地验证：`python checks.py`。本地无PyTorch，Torch parity项会skip；服务器必须运行该项及CUDA smoke。合成学习测试、调度器测试的数值均不计为模型实验。

只依赖 Python+NumPy 的离线工具；在线模块使用项目已安装PyTorch，不要求新模型训练环境。不要将整个模型attention切到慢速实现来获取本模块统计。
