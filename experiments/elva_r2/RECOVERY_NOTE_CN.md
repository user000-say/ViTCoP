# ELVA R2 恢复交付说明 — 2026-10-19

本次从当前工作区实际遗留的 `elva_r2/` 目录恢复交付。算法 Python 源文件和原中文 Cursor 执行书保持内容不变；重新运行 `verify.py` 后，更新本地验证记录，补齐源码展开稿、文件校验清单和 ZIP。没有用本次新写算法冒充上一轮原代码，也不声称与一个未找到的旧 ZIP 字节相同。

## 完成范围

- 16 个完整 Python 文件，包含算法模块、配置生成、队列、测试和复验入口。
- 83 个开发实验配置；配置数不代表实验已经在模型上跑完。
- `CURSOR_HANDOFF_CN.md`：完整接线、优先顺序、四卡安排、验收和回传要求。
- `CHANGELOG_CN.md`：相对此前要求的实现变更。
- `RESEARCH_DECISION_CN.md`、`HISTORICAL_RESULTS.csv`：研究取舍与历史汇总，历史汇总不是逐样本原始日志。
- `FULL_SOURCE.md`：逐文件展开全部 Python 源码，无省略号或删节。完整 JSON 配置位于 `configs/`。
- `MANIFEST_SHA256.json`：包内文件字节校验；`SOURCE_SHA256.json` 单列恢复时的全部 Python 源码摘要。

## 保留的核心实现

Stage1 对比 pen、L9、dual 及原 COVER；投影输入始终使用原 pen 向量，只改变选点。Stage2 保留三档精确工作量、同预算日程对照、原 QED/coverage/RDRIFT 接口与必要桥接；新增的临时存活交换固定共同最终身份，分别测试原生 KV 和支持集合对齐。有限后续日程检查严格预算，新学习目标在真实独立图像证据就绪后才启动，不能复用旧 E020 权重。

| W | block 1–10 | block 11–20 | block 21–32 |
|---|---:|---:|---:|
| 2048 | 120 | 56 | 24 |
| 1536 | 90 | 42 | 18 |
| 1024 | 60 | 28 | 12 |

已按逐层求和与 Fraction 反算复核这些预算。W 是视觉 token×层工作量，不等于 FLOPs 或 TTFT。

## 本次复验

`python verify.py`：48 项测试，47 项通过，1 项跳过，失败与错误均为零。PyTorch 路径因当前环境缺少 PyTorch 未执行；另有 1000 个小规模后续日程实例，与独立整数穷举逐例核对。测试覆盖身份与 payload、层边界、预算、同预算两分支、交换反向、KV 支持对齐、前缀不被修改、答案因果移位、学习闸门、来源指纹和队列恢复。

随机因果 decoder 的输出位于 `validation/toy_swaps_FIXTURE_ONLY.json`，仅作执行夹具；没有新的预训练 VLM 准确率或 GPU TTFT。`ENVIRONMENT_LOCAL.json` 记录本次本地依赖状态。

## 交给 Cursor

把整个 `elva_r2/` 目录交给 Cursor，先读 `CURSOR_HANDOFF_CN.md`。本地复验入口为 `python verify.py`。服务器沿用原仓库和环境，把原 COVER/QED/RDRIFT、模型前向/缓存裁剪与官方 scorer 绑定到接口；必须实际执行 PyTorch parity、真实关闭路径回归后再跑主实验。优先 Stage1 核心对照和固定日程，学习条件不满足时继续其他就绪任务。

当前缺原模型仓库、checkpoint、GPU 接入和逐样本服务器日志；这份交付是完整增量模块，不能称为已经接线并跑通的整仓补丁。旧记录提到的 SVG/PNG 或其他历史 ZIP 不在本次遗留目录中，本包不把重新绘制的图冒充原文件。

恢复时另核对了 [CVPR 2027 官方日期](https://cvpr.thecvf.com/Conferences/2027/Dates) 与 [Transformers 缓存说明](https://huggingface.co/docs/transformers/main/cache_explanation)。原交接的模型绑定和逐层缓存审计要求继续适用；未进行新的算法文献综述或改变既定研究方向。
