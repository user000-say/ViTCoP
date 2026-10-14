# E018 验证状态

日期：2026-10-14。此报告区分实际执行和待服务器执行，不以语法检查替代模型验证。

## 已在当前环境执行

`python -m unittest discover -s deliverables/E018_Continuation_20261014/tests -p test_e018_cpu.py -v`

9项全部通过：

1. 所有Python文件AST解析。
2. 移除E018挂接块后，两个完整主源码的AST与原E014一致。
3. 三个K档的post-block单次/两次日程W等价；平均64附近的控制预算计算。
4. 非法日程、端点、预算及零K拒绝。
5. 协议指纹允许lambda变化，拒绝cache/Stage1身份变化。
6. 图像分组与有限样本标定分位数。
7. 小型MLP对独立合成监督的拟合与特征mask行为。
8. 完整训练CLI→JSON标定产物；重复标签拒绝。
9. 安装器dry-check→备份安装→回滚→新版源码冲突拒绝。

合成监督仅用于验证训练程序，没有保存为真实控制器，也不是VLM实验结果。安装器依赖占位文件仅存在临时测试目录，没有用它们冒充真实E010/helper/builder。

`bash -n scripts/e018_env.sh` 与原e014_env.sh语法检查通过。CLI help/代码包与差分完整性检查通过，最终SHA256清单在`MANIFEST_SHA256.txt`。

## 当前环境不能执行，已提供测试入口

PyTorch、Transformers、模型权重与CUDA不存在。因此**没有实际执行**：

- `tests/test_e018_torch.py` 的9项真实修改SDPA小模型回归；
- 真实LLaVA模型的loader/MM preparation、旧路径身份、生成终止规则回归；
- HIST-COVER32所需未上传E013 helper的身份与对应逐题结果核验；
- 两种cache语义下真实模型的反事实标签；
- 真实控制器训练/未知图像预测指标；
- 新官方benchmark得分、完整模型TTFT或Fixed32。

不能把本包说成已经GPU验收或已经出现投稿增益。必须先按`CURSOR_HANDOFF_CN.md`完成服务器小模型与32～64题smoke，再启动正式长实验。

## 已做的语义审查

- 仅当前问题行、完整合法prefix的softmax；未对visual子块先归一来伪造mass。
- 原始patch ID与张量位置分离；每次删除保留所有文本，更新问题索引。
- 保留原始RoPE；decode按原始下一个位置而不是压缩cache长度。
- stage1 metadata只携带已实际选择的原patch来源与attention。
- 同状态反事实共享prefix；冻结当前D及未来keep IDs；检查相同final survivors。
- scrub_all检查最终cache key位置相同；layer_local不解释为纯吸收收益。
- 每组删除只执行一次；记录distinct block退休位置及实际hidden gather次数。
- 特征来自当前/过去状态；GT仅用于offline分支答案NLL。
- 当前候选MLP标定按image group；线上协议不符/未标定参数拒绝。
- COLLECT可重放既有控制器，先捕获实际状态后读取GT。
- STATIC/PROXY的Q_ATTN路径不计算未用的pairwise redundancy或全部controller features，避免给强简单对照人为加开销。
- 计时trace同步关闭，实际线上决策/reader与必要同步不移出timer；Fixed32不提前EOS；每次重算vision。

以上是代码审查结论，仍需真实Torch/GPU测试证实实现与当前服务器架构一致。
