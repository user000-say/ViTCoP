# E019 Progressive Evidence

先读 **RESEARCH_DECISION_CN.md**，交给Cursor执行 **CURSOR_HANDOFF_CN.md**。

本包新增了真实算法代码、近同工作量日程生成、48配置/192任务矩阵和可执行子进程队列；不再只是48条任务台账。

核心模块：evidence_numpy.py、evidence_torch.py、selector_hook.py。
实验工具：schedules.py、build_matrix.py、run_matrix.py、workflow.py、summarize.py、promote.py。
接口与核查：project_bridge_contract.py、checks.py、analyze_reported.py。
optional_E018包含上一轮收益控制器代码，供并行研究使用，不是本轮主矩阵的依赖。

离线检查仅需NumPy/Python；真实hook需要项目现有PyTorch。无真实项目源码，Cursor需复用已经跑通的ACC/TTFT入口实现薄bridge。没有伪造的模型入口或新增GPU结果。

```bash
OPENBLAS_NUM_THREADS=1 python checks.py
python analyze_reported.py
```

runtime.example.json有意保留null，必须导出真实日程/来源/面板身份，不能当作已知工程配置。
合成测试只核查数学和调度，不证明VLM准确率、真实吞吐或论文新颖性。
