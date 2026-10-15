# 加量任务包入口

先读 EXTENDED_TASK_BOOK_CN.md；task_queue.json列出48项任务、依赖和门槛。代码沿用上一包，没有新增GPU验证。

# E018: counterfactual deferral controller

先读 **RESEARCH_DECISION_CN.md**，交给 Cursor 执行 **CURSOR_6H_4GPU_CN.md**。

这是本轮新实现的独立模块，包含可训练的 NumPy 控制器、PyTorch 在线后端和真实项目采样协议。
当前附件没有项目源码，因此不是修改原 E014/E017 仓库的已集成补丁。Cursor须在实际仓库实现ProjectAdapter并接入已有pruner。
不包含伪造的项目安装器、GPU结果或论文收益；PyTorch后端须在服务器通过数值对齐。

| 文件 | 用途 |
|---|---|
| controller_numpy.py | 区域历史统计、当前状态特征、按图分组、实际MLP训练、检查点、收益—成本决策 |
| controller_torch.py | 对应的设备端特征计算和冻结控制器推理，尚待服务器验证 |
| adapter_contract.py | 同前缀反事实分支协议与基本核验 |
| collect_pairs.py | 项目适配后执行真实配对采样，不猜测模型或KV接口 |
| merge_pairs.py | 合并相同配置的互斥分片，拒绝重复样本与配置漂移 |
| train_controller.py | 同容量zero/geometry/retained/history训练及独立图像组NLL重放 |
| audit_r4.py | 正确四任务R4与2×2交互；没有逐样本数据时不做显著性声明 |
| checks.py | 数学/训练/数据隔离/序列预算检查；另有条件运行的PyTorch对齐 |

依赖：离线训练/核查仅需Python与NumPy；在线后端需项目现有PyTorch。

```bash
OPENBLAS_NUM_THREADS=1 python checks.py
python audit_r4.py
```

运行真实训练/评估前，按执行书提供真实adapter与冻结配置。禁止把synthetic数据或checkpoint用于正式结果。
若需要修改已有模型文件，必须依据实际源码生成差异并保留原路径；不能根据旧聊天中的文件名和函数名猜补丁。
