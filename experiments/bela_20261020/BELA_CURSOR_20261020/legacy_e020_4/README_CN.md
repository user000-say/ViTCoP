# E020.4 完整研究模块包

阶段二：受限表示回收 × 交互预算/层位 × 实际首token成本。

先读 `RESEARCH_DECISION_CN.md` 和 `CURSOR_HANDOFF_CN.md`。这是完整可检查的策略/运行状态/诊断工具代码，**不是原模型仓库的已接线补丁**；原仓库、checkpoint、GPU和真实评测数据不在本工作区。没有新增VLM精度或TTFT结果。

新增：`current_recycle.py` NumPy参考、`current_torch.py`设备实现、`current_runtime.py`状态绑定、`paired_benchmark.py`配对测时、`paired_quality.py`官方评分回调的配对bootstrap、`validate_current_cuda.py`真实捕获对齐、`extend_matrix.py`扩展目录。

原版缓存算子、廉价选择器、成本核算、容量响应与测试一并保留。生产模型仍拥有selector、position/mask和KV处理。无需重训学习头。

本地复查：

```bash
python -m unittest discover -s tests -v
python extend_matrix.py
python -m compileall -q .
```

矩阵入口是 `extend_matrix.py`；`make_matrix.py`仅保留原基础目录，输出 `base_matrix.json`，不会覆盖扩展目录。

服务器捕获actual_stage.npz后：

```bash
python validate_current_cuda.py actual_stage.npz actual_parity.json
```

必须包含hidden/keep/metric/coords/groups/positions，不是生成伪造捕获。测试通过后还需运行原模型关闭路径回归及开启路径完整生成。PyTorch/CUDA本地未运行，不能据此声称设备对齐完成。

101配置是可调度研究目录，不等于已跑101轮；随机种子是重复证据，非新方法；四任务全展开最多404个筛查作业，不承诺未经pilot的总时长。确认使用独立image组，实际任务规模按吞吐分配。
