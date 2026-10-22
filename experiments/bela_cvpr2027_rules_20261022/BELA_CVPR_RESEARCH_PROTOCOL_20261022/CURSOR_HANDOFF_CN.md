# Cursor 执行 Handoff：先验证分布，再决定模块（2026-10-22）

## 交付文件如何接入现役工程
将 `AGENTS.md` 放工程根目录，将 `.cursor/rules/bela_cvpr_research.mdc` 放 `.cursor/rules/`。该规则设置 `alwaysApply: true`；仍需确认 Cursor 版本是否自动加载。所有读写操作仅针对当前工程，不得把旧 E014 ZIP 当成现役代码。

## 先只运行离线测试
```bash
python -m unittest discover -s tests -v
python tools/schedule_audit.py research/experiment_matrix.csv
```

## 阶段一：只读工程审计（P0，不允许盲目修改）
确认当前 commit、模型 checkpoint、vision encoder、token pruning hooks、prefill/decode 区分、物理 token 压缩、KV 结构、position IDs、mask、patch 原始 ID 映射；打印历史 A0/C0 输入和实际逐层 K；对四项任务保存 scorer/metric definitions。重跑 A0 主配置并与原始日志建立绑定。

## 阶段二：独立分支实现 T1 精确 W2048 日程
严格限制变更：只允许把第二次剪枝的 `(k_mid,k_final)` 改为预设组；不得更换 Stage1 排序、RDRIFT 公式、prompt、模型权重或 evaluator。根据矩阵逐项为每张样本收集：
- `sample_id`, dataset, question_id, image_id, prompt_hash, checkpoint_hash, git_commit
- `stage1_token_ids`, `kept_ids_after_first`, `kept_ids_after_second`
- `actual_K_per_layer`, `W_verified`, `selector_ms`, `prefill_ms`, `TTFT_wall_ms`, `TTFT_cuda_ms`, `peak_vram_mb`
- `prediction_raw`, `prediction_parsed`, `ground_truth`, `is_correct`, `POPE_TP_FP_TN_FN`
所有 arm 共用冻结样本列表。

先小样本 smoke（至少含单题、多题、特殊文本长度和 batch=1）；然后 A0/A2/A3 在 GQA+POPE 跑相同评测；结果不同才决定是否扩大到 A1/A4。不得用开发集挑选后又将开发集称为独立测试。

## 阶段三：同 K 深层选择器对照
在最佳 T1 轨迹上，只改变第二次剪枝的 `RDRIFT/Q-Attn/seeded_random/stride`，其它条件冻结。至少三个随机种子；汇总 GQA/POPE 单项、配对 win-loss 表、分组置信区间与选择器时间。若随机表现与 RDRIFT 无可辨差异且延迟更低，优先删除昂贵深层选择器，而不是继续调 drift。

## 阶段四：T2、T3 + 模块相互作用
在 T1 决策后执行 B0/B1/B2、C0/C1/C2/C3 的受控实验。与 STAR-Pro 的预算形状相同不等于运行 STAR-Pro 原算法；对后者需独立运行官方代码作为外部参照。
交叉比较 Stage1 (`COVER`/penultimate) × Stage2 (`Q-Attn`/`RDRIFT`) × 至少两种日程，分别报告主效应及交互项。
成功判断不靠 R4 的单个小数点，而是四关键指标的联合表现和真实 Accuracy–Latency Pareto。

## 风险禁止
- 不能把名义 W=2048 当作实际执行 W=2048；检查每层 trace。
- 不能把 64 个平均 Token 比作另一个方法“最终保留 64”。
- 不能以数学测试 PASS 宣称 GPU 或仓库集成 PASS。
- 深层 K=0 是数学上允许的诊断，但不表示现役模型可无修改支持。
- 千万别在没有证据时增加复杂度动态控制器或扰动修复，尤其不能再次复制 ICER 高延迟代价。

## 第一次向用户返回的报告格式
1. P0 审计来源、commit、评测与 MME/POPE 口径
2. A0 原始结果与实际延迟能否复现（不能则记录阻塞）
3. A2/A3 准确率四项或已完成两项 + selector latency
4. T1 是否支持“中层转移比深层 32 更优”的假说
5. 是否推进到 T2，或应转向入口候选/排序失效诊断
6. 更新投稿论文中可以/不可以陈述的贡献
