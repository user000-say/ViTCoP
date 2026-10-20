# BELA | Cursor 实验执行 Handoff（2026-10-20）

## 0. 状态与不可越过的边界

交付的是**经本地单元测试的完整 BELA 独立算法核心** + E020.4 原始模块备份，而不是已经修改过的 LLaVA/ViTCoP 生产推理仓库。当前交付资料不包含经核实的服务器 HEAD、完整 model forward、模型权重、逐样本预测、真实 GPU 计时日志。**不得把 standalone 通过写成模型级通过**。

工程目标：
1. Stage1 在同一底座上固定为 `W2048 dual + COVER` / `W1024 penultimate + PCOV`，仍以原 penultimate patch representation 进入 projector；保留原 patch ID。
2. Stage2 实现 exact token-layer budget + evidence lifetime selection；关掉 merge/recycle；首先复刻最强 REF/RDRIFT，再测 BELA 和 equal-budget survival swap。
3. 单一配置同时给出相同模型、相同环境下的官方质量和真实加速；不要拿两条不同配置的 Q 和 latency 拼接。

## 1. 工作树

- `bela/core.py`：严谨计算 `Plan` / `find_exact_plans` / `LifetimeState` / `stable_keep` / `lifetime_scores` / `equal_budget_swap`。
- `bela/cli.py`：枚举严格等 W 方案；不存在解就非零退出。
- `tests/test_core.py`：单元测试（预算、非法边界、重复事件、同分排序、缺失/NaN、swap）。
- `legacy_e020_4/`：原 E020.4 模块完整留档，仅供接口和历史对照，不自动合并入主运行路径。
- `bela/tensor_adapter.py`：实际 PyTorch sequence tensor 剪枝辅助；保持原文本顺序，支持非连续 visual positions，**不自动处理 KV/mask/RoPE**。
- `model_hook_contract.py`：接线合同及位置语义核对。
- `run_manifest_template.csv`：生产实验的结果记录字段。
- `validation_report.json`：此次可复核的本地检查（不含模型结果）。

## 2. 基线与目标

必须锁定模型 commit/checkpoint/tokenizer/image processor、精度、torch/transformers/flash-attn、GPU、驱动、推理 batch、视觉 crop、生成上限和评测脚本版本。

原模型 32 层 × 576 patch 时全量视觉 token-layer 为 18432。
- W2048：可算的示例为 entry172、block4 后 90、block12 后32（不是推荐最优日程）。
- W1024：可算的示例为 entry60、block3 后54、block13 后16（不是推荐最优日程）。
- 两工作点 `1 - W / (576*32)` = 88.8889% / 94.4444%；该比率仅是 visual-token-layer proxy，**不是整模型 FLOPs 减少百分比**。
- 入口 stage1 如有其他 visual special tokens/crops，基数必须重新核对；不要默认为 576。
- 历史 95.90@W2048、92.71@W1024、96.95@W2284 是过去汇总的待绑定历史信号，不是本次重新跑出的结果。
- 目标≈97 和≈94 指四任务 `R4=25*(GQA/61.9+POPE_F1/85.8+MME/1852+MMBench/64.1)`；论文叫 **Average Retained Performance (%)**，不得叫原始平均 accuracy。

## 3. Cursor 先做的真代码集成

### A. Stage1：轻改动，不重新发明入口

1. 找到 ViT encoder 输出 patch tensor 与原始 patch ID 的对应关系，比较仅 penultimate、仅 layer9、dual 选点 + COVER/PCOV；selector 输出整数 original patch ID，送往 projector 的仍是 penultimate 对应 hidden。
2. 将 ID 映射带到 LLM decoder；对 multi-crop 每个视图 ID 必须全局唯一，保留来源 crop/position。
3. 做 OFF-path bitwise/容差回归：禁用 BELA 时与原模型官方预测、logits 一致（允许设备非确定性设定下的容差）。

### B. Stage2：完成真实 decoder 接线

找到原模型 **prefill** LLM decoder block 循环。事件定义 `after_block=b`：先让第 b 个 block 使用裁剪前完整集合，结束后裁剪视觉 token，再执行 block b+1。避免 0-based hook 的 off-by-one。

一个样本初始化 `LifetimeState(plan, original_entry_patch_ids)`。
每个 block 真实执行结束后：
- 计算 `observed_visual_tokens`（根据当前实际视觉 token index 而非 shape 中估算）。
- `state.record_block(block_no, observed_visual_tokens)`。
- 若该 block 是事件边界，使用原 selector 输出与 `state.ids` 顺序严格对齐的 score；调用 `slots, new_original_ids = state.prune(block_no, scores)`。
- 把 `slots` 真实作用到当前视觉 hidden、position_ids、attention_mask、KV cache/gather、视觉索引与其他依赖视觉序列长度的对象，**必须沿用仓库已验证的剪枝路径**。不要只删 NumPy IDs。
- 如存在 prefill cache，确保 key/value 的序列长度与实际新 token 序列同步；别重排/改变 text token 和因果遮罩。
- 末端运行 `state.finish()`；保存 counts、W、每次保留的 original patch IDs，出现越界/错位立即失败而不是静默继续。
- decode 阶段不能错误重触发 prefill 剪枝。
- 如果模型的 HF forward 由普通 decoder layer 循环改为 flash attention 或 compiled graph，hook 的实际语义必须重新检验，不能按上述模式机械猜接线。

**注意：`LifetimeState` 是 audit + deterministic keep-index 核心，不拥有实际模型 hidden/KV 的编辑权。这是有意设计的 API 边界。**

### C. RDRIFT 强 baseline 与候选 BELA selector

先确认旧 RDRIFT 实际 `rank_drift`: now/old 的定义、ranking 方向、tie、窗口、作用层位、归一化方式。当前 `lifetime_scores` 提供透明 `0.6*now + 0.4*(now-old)` 代理，**不是声称与原模型的实现完全一致**。
候选未来生存信号可先用 cheap current attention + RDRIFT + entry prior + coverage；先固定/小网格冻结系数，报告 cheap selector overhead，再考虑在**按 image 分组的 dev**上用干预产生的未来质量差标签训练线性校准器。不要将未来 attention 当答案收益标签。

## 4. 按顺序进行的实验（所有比较先固定 W）

**Gate 0 / 模型接线审计（必须先过）**：
OFF-path、decoder 视觉 id 不变、实际 per-block trace 完全吻合、两个精确 W 可跑完真实生成、官方 scorer 可重新算、相同图像问题可复现。抽查 OCR 与 multi-crop 和双问题图像。

**Gate 1 / 基线复现**：
相同 commit/checkpoint 上运行旧 W2048 best、旧 W1024 best、当前 REF、RDRIFT，记录实际 W、真实 R4、四任务每列、CUDA TTFT。历史汇总不能替代这个 gate。

**Gate 2 / 生命周期因果性**：
固定入口 K、最终 survivor ID、全部事件层位、各事件容量和 W。只交换 interim extra token group A/B 的 survival 长度。要求同图、同问题对比，至少报告 paired win/tie/loss、类别分析、原预测与 official scorer。每组 extra set 尺寸一致，交换导致的选择器排序代价要记录。A/B 阶段有差异不能直接证明 estimator 准确；另对预测的 swap 胜负用独立样本检验。

**Gate 3 / 日程和 score**：
先做 matched budget strong fixed：无动态、Q_ATTN、历史 RDRIFT、BELA scorer；加 1-event 与 2-event、优选 boundaries、廉价 random/deep random 控制。每档 W 至少一组 count/profile matched。仅当不同样本的最优方案互补且开发集可预测，才允许加在线 policy routing；routing 对比最佳固定、同预算频率乱配、图像/问题交换路由。

**Gate 4 / 扩展评测**：
四任务 GQA/POPE F1/MME/MMBench 原评分器 + OCR、TextVQA、ScienceQA-IMG、VizWiz 等具代表性任务（以服务器真实可用官方基准为准），所有任务使用事先冻结的配置。Dev/Test 按 **image id** 分离，MME 同图双问不可拆。

**Gate 5 / 速度、算力**：
独占同型号 GPU, warmup 后交错 A/B/B/A，batch=1，固定 image resolution / context / output token count / dtype / backend，报告 prefill wall TTFT、CUDA events、p50/p90、峰值显存、真实 TTFT、end-to-end latency，分别计 vision encoder / selector / sort / gather / text+vision decoder / cache / generation；另用 THOP/torch.profiler/模型级可复核 FLOP 统计工具输出 prefill 整体 FLOPs，**明确 fused kernel FLOPs 是否漏算**。不同方案不允许不同后端。

## 5. Fail fast（淘汰线）

- 任一 W trace 失配、原 ID 身份错误、OFF-path 性能变化或真实 token 变更未反映在 KV/position/mask：停止 accuracy sweep。
- 重建的 BELA selector 不能在准确匹配的强固定 QATTN/RDRIFT 和随机选择对照上胜出：不宣称预测未来生存效用。
- 等预算 swap 无稳定、可重复的回答质量差，或预测 swap 结果无相关性：不写“证据生命周期效用估计被证明”。
- Quality 改善但 selector overhead 抵消 decoder 省时：报告真实 tradeoff，不宣称加速。
- W1024 用 merge 只有在纯 hard pruning 全部 gate 完成后，才作为补充消融独立开启，绝不混进主算法对比。

## 6. Cursor 要回传的最小可审核证据

`git rev-parse HEAD`、`git diff --stat`、`git diff --check`、修改文件 diff、检查点标识和哈希、环境快照、GPU 型号、完整命令、任何失败 traceback、raw predictions、官方 scorer 输出、逐 block counts/原始 IDs trace、CPU/GPU timing 原样本、FLOPs profiler 报告、配对 image IDs、四任务 R4 重算表。

提交工作产物建议目录：
```
server_results/
  environment.json
  integration_diff.patch
  parity_off.json
  schedule_trace_W2048.jsonl
  schedule_trace_W1024.jsonl
  quality_raw/
  official_scores.json
  timing_raw.csv
  flops_report.json
  survival_swap_pairs.jsonl
  summary.md
```

## 7. Cursor 运行的第一组指令

```bash
python -m unittest discover -s tests -v
python -m bela.cli --target 2048 --entry 172 --boundaries 4,12 --final 32
python -m bela.cli --target 1024 --entry 60 --boundaries 3,13 --final 16
python -m compileall -q bela tests
```

第二步定位原模型每次 prune 所在函数，贴出对应源码和原 RDRIFT 实现，**先完成 Gate 0 再跑大矩阵**。若服务器未附当前模型仓库，不要编造准确接线补丁或跑出假 benchmark；原仓库可用后再按这些接口逐行修改。

## 8. 决策

投稿方法故事暂定：**Stage1 可解释的 evidence admission + Stage2 严格等预算的 evidence lifetime allocation**；主创新是未来生存价值是否可由当前廉价信号预测和在全局预算下兑现。不能将 Layer9、dynamic K、attention drift、prune+merge 单独声称新颖。论文名暂用 BELA（待查正式同名），不是宣称全新术语。

CVPR 2027 官方时间：2026-11-10 AoE 注册，11-16 AoE 正文，11-23 AoE 补充材料。应并行撰写研究问题、实验协议、失败案例和相关工作。
