# ELVA R2 — Stage1 收敛与视觉证据存活干预

先读 `CURSOR_HANDOFF_CN.md`，再读 `RESEARCH_DECISION_CN.md`。代码是本轮重新实现的完整增量模块；当前没有原模型仓库、旧 ZIP、checkpoint、逐样本结果或 GPU。不能把它称为已经修改并跑通的原仓库补丁。

研究目标保持为：不同深度需要怎样的视觉证据，在给定总视觉工作量下怎样分配存活机会。当前不预设动态一定优于固定，不重新启用旧 E020 交换头，不把多层 ViT、COVER、RDRIFT、merge 分别包装为原创贡献。

本轮可直接运行：

```bash
python make_catalog.py
python verify.py
```

Python 3.10+、NumPy；GPU接线另需项目原有 PyTorch/Transformers。CPU检查不下载模型。`elva/torch_ops.py` 是设备端实现，必须在服务器运行 `tests/test_torch_parity.py`；本环境缺 PyTorch，该测试明确跳过。

| 文件 | 实际作用 |
|---|---|
| `elva/budget.py` | 精确分段日程、逐层记账、有限目录的合法后续路径、单路径决策 |
| `elva/capture.py` | 同一 ViT 前向采集指定完成层，不要求返回完整 attention |
| `elva/evidence.py` | NumPy Stage1、原向量保护、跨层身份、注意力与 RDRIFT 参考 |
| `elva/torch_ops.py` | Stage1 与部分 QK 注意力的设备端对应实现 |
| `elva/integration.py` | 调用原生 selector/compact/inspect；不猜测模型 mask 或 cache 实现 |
| `elva/counterfactual.py` | 同预算临时存活交换、共同最终身份、两种 KV 干预、答案 NLL 因果移位 |
| `elva/policy.py` | 同一有限目录的离线标注与在线单次交换，强制共同最终集合 |
| `elva/learning.py` | 新目标的可选小型增益回归、按图像隔离、最大候选值阈值校准 |
| `elva/toy_adapter.py` | 随机因果 decoder，运行 prefill、KV decode、teacher forcing 和独立贪心生成；仅用于夹具 |
| `elva/benchmark.py` | 请求准备到首 token 的交错墙钟 TTFT 回调 |
| `make_catalog.py` | 预定义、去重的开发实验目录 |
| `run_queue.py` | 配置与来源指纹、三卡队列、断点复用、失败隔离、进程组超时回收 |
| `HISTORICAL_RESULTS.csv` | 用户报告的历史舍入汇总及重算 R4；不是原始日志 |
| `validation/` | 本地检查和明确标为 fixture 的因果模型结果 |

三档默认日程均使用从1开始的实际执行 block：

| 显示压缩率 | block 1–10 | block 11–20 | block 21–32 | W |
|---|---:|---:|---:|---:|
| 88.9% | 120 | 56 | 24 | 2048 |
| 91.7% | 90 | 42 | 18 | 1536 |
| 94.4% | 60 | 28 | 12 | 1024 |

默认日程是预注册候选，不是已经找到的最优策略。相同入口/末尾容量/剪枝次数的等 W 对照已列入目录。附件原来的短早期池方案另作跨入口 recipe 对照，不能单独归因层位。

正式九任务的完整名单当前不可核验。Cursor必须读取服务器已经冻结的 benchmark/scorer manifest；本包只默认列出已知的 GQA、POPE、MME、MMBench，禁止自行补齐五个名称或计算不完整 R9。

