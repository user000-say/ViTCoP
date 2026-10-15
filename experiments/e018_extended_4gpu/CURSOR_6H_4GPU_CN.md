# Cursor 执行书：四 GPU、6 小时墙钟预算

先读 RESEARCH_DECISION_CN.md；本书是执行约束，不是全量完成承诺。
本轮没有真实项目源码。请在现有仓库新建隔离分支，新增模块并实现 adapter，不覆盖历史路径。
先运行本包 checks.py；PyTorch parity 在服务器必须实际运行，不能接受 skipped。

## A. 第一优先级：恢复事实，最多45分钟先给状态

1. 找到产生 COVER/BASE × ONE/TWO 的四条真实命令、配置、commit、checkpoint与预测文件。
2. 输出 ONE/TWO 各自每一层的视觉token数、选择器、完成多少block后删除、是否多次推理、是否逐样本变预算、query span、位置ID及每层KV长度。
3. 用 audit_r4.py --metrics-json four_arms.json 重算原始四任务数值。文件格式为四个臂名映射到 GQA/POPE/MME/MMBench 四项原始数值，不能填已有综合均值。
4. 对 POPE 明确 accuracy/F1、数据split和当前分母85.8对应的原始baseline metric；不混用。
5. 明确 TWO 是否与 ONE 同成本。先给完整日程和 token-layer，再测真实延迟；不能由最后保留K推断预算。

若 TWO 是固定两次剪枝，称为 progressive；若预算或时机按输入变化才称 sample-adaptive。
如果包含按真值取优、答案后验挑分支或重复推理，立即标注并停止将该表用作在线动态证据。

如旧结果口径无法恢复，先修复复现，不在无基准上大规模跑新方法。

## B. 本轮只改一件事

沿用可信 TWO 的已有早期事件；在最后一次删除前，增加 early / late 二选一。
early 当前删至同一个最终K；late 多保留4层后删至同一个K。若剩余深度不够，先确定合法的一对边界并冻结，不自动改其他参数。
选择规则固定为已经验证的Q-ATTN，或者在保持当前TWO规则的情况下先复现；主比较不得同时更换选择器。
入口数量、Stage1 helper、原图resize、prompt、答案评分、后续cache策略全部固定。
当前表没有证明COVER必需；BASE与COVER作为因子控制，选择更简单且可信的入口作为主采样源。

层位统一写“已完成d个block”，不混用0-based layer index与论文L8。
controller_numpy.token_layers 可核算日程；代码检查示例4/8/12与172/96/32不是项目推荐配置。

## C. 项目适配器：真实接口不能猜

实现模块如 project_cf_adapter.py，暴露 build(config) 工厂，返回对象：

- iter_samples()：训练/开发源样本迭代器，按全局图像内容ID确定卡间分片。
- manifest()：至少包含 checkpoint_hash、stage1_config_hash、selector_hash、schedule_hash、prompt_hash、projection_hash、cache_policy、query_rule、cost_source、data_split。
- prepare_prefix(sample)、features(prefix)、clone_prefix(prefix)、run_branch(prefix_clone,sample,action)：遵守 adapter_contract.py。

features(prefix) 返回：base[55]、history[272]、retained[272]、extra_ms>0、canonical image_id、唯一sample_id、stride=17。
image_id必须跨问题、跨任务、跨Stage1臂统一，优先图像内容hash，不能只用各任务自己的整数qid。
sample_id包含任务、qid和Stage1配置，防止多臂覆盖。所有投影持久化，不能每个样本或每次加载重新随机。
data_split只可为 train_only 或 development_only；不得拿官方测试题训练/选择控制器。

run_branch 返回：prefix_fingerprint、answer_token_ids（Python整数list）、answer_nll、prediction。
检查答案分词一致、prompt只含问题和必要模板、答案mask完全排除prompt。
answer_nll需按同一参考答案的token数量归一；若多个正确答案，预先冻结选用规则，不按分支选择最有利答案。
NLL之外保留实际生成答案，后续用官方打分确认收益与正确率一致。

必须深拷贝或重建可变KV、hidden、mask、位置和选点状态；两支不能互相修改。
如果安全clone不易实现，离线可分别完整重跑early/late，并核对实际相同前缀的指纹；记录增加的采样成本。
不要为节约采样时间接受浅拷贝错误。全量hidden指纹/CPU校验仅离线核验，不能放进测速热路径。

Stage1 原图patch摘要只供控制器读取，不作为新token输入LLM，不改写幸存特征或attention logits。
单图24×24可显式构造4×4 region_id；多tile/多图/视频必须保留tile/frame身份，不能简单按flat索引取模。
本轮在线实现以batch=1为验收范围；不要宣称支持任意ragged batch。

设备端：加载controller_torch.FrozenController(...).to(device)，保持FP32权重；特征计算和控制器段禁用外部autocast。
已提供PyTorch特征函数与NumPy定义。先用相同捕获输入验证误差，再做CUDA smoke；不能只确认import成功。
只在最后删除点调用一次控制器。summary可在Stage1后计算一次，计入端到端开销。
设备返回的布尔决策到Python分支可能产生同步，实际同步时间也计入TTFT。
已有Q/K统计能复用就复用；若须额外投影或attention计算，单独记录开销。不要全局切换eager attention以获取完整矩阵。
保留既有pruner对KV/position的处理；本包控制器不负责擅自重写不同架构的cache。

## D. 四卡排期

| 时间 | GPU0 | GPU1 | GPU2 | GPU3（独占测速） |
|---|---|---|---|---|
| 0–45分钟 | 四臂配置与前缀核查 | PyTorch parity与模型smoke | 复现可信基线、冻结图像组面板 | 输入/边界核验，early/late成本表 |
| 45–150分钟 | 默认Stage1配对采样分片0 | 同源配对采样分片1 | 同图BASE/COVER小型因子控制；空闲后加采默认源 | 交错early/late、selector与摘要开销 |
| 150–210分钟 | 补训练/开发源配对数据 | 固定日程邻域小面板 | 64例固定token身份的延迟删除诊断 | 控制器FP32 CUDA smoke与固定生成长度 |
| 210–330分钟 | 冻结history策略独立评估 | 同容量retained策略独立评估 | 开发集选定强固定与随机分配控制 | 交错测试固定/retained/history真实总延迟 |
| 330–360分钟 | 汇总逐样本输出 | 补最关键缺失对照 | 官方指标与机制案例复核 | 漂移复核、导出原始计时 |

CPU离线训练与配对采样并行使用已有数据，训练4种输入模式，先固定seed0。若时间允许，补seed1/2仅作稳定性检查，不挑最优种子。
三卡配对数按实测速率决定：先16–32对pilot，测每对总耗时t_pair；采样上限不超过可用采样卡×时间预算/t_pair。
目标先128对诊断，再争取512–1024对；样本不足时只报告pipeline/初步诊断，不写“已证明机制”。
优先保证同一组图像上的关键配对和独立评估，不承诺半天跑完四任务全量。若吞吐低，缩减方法数量，不改变已经冻结的样本选择规则。

## E. 采样、合并、训练命令

在项目PYTHONPATH可找到adapter后运行；下面module/config是接口示例，需替换为实际已实现模块和冻结配置，不是可凭空运行的工程入口。

```bash
python checks.py
python collect_pairs.py --adapter project_cf_adapter:build --config frozen_shard0.json --output runs/pairs0.npz --limit 256
python collect_pairs.py --adapter project_cf_adapter:build --config frozen_shard1.json --output runs/pairs1.npz --limit 256
python merge_pairs.py --output runs/pairs.npz runs/pairs0.npz runs/pairs1.npz
OPENBLAS_NUM_THREADS=1 python train_controller.py runs/pairs.npz --out runs/seed0 --seed 0 --steps 300 --fraction 0.5
```

每卡使用独立进程/工作路径和显式可见GPU。不要在同卡并发精度与测速。
merge_pairs只允许同配置分片合并；BASE/COVER配置不同的因子数据分开存，跨Stage1迁移作为单独试验，不能偷偷混合为一份标注相同的训练源。
训练脚本按canonical image_id固定60/20/20分组。所有问题、Stage1臂和同图变体属于相同分组。
λ只在验证组校准；--fraction是开发集目标，不保证测试组同预算，必须报告实际占比和真实成本。
默认动作不为负的预测收益额外花计算，因此小于目标占比可能是合法结果；不要强制填满预算。
train_controller.py的report是离线NLL反事实重放，既不是官方R4，也不是在线测速结果。
训练日志里的打乱动作仅保持占比，不保证相同延迟；真实随机控制按prompt长度/当前token数分桶并报告成本差，必要时用相同实测成本包络比较。

## F. 三道门槛

G0：复现和语义通过。ONE/TWO定义、POPE口径、token数量、前缀与cache、PyTorch parity任一不清楚，先修复，不解释新模块收益。

G1：存在可用余量。early/late实际预测有互补，开发集同成本oracle优于固定日程包络，且这种NLL余量与真实正确率方向一致。
仅用每题min(NLL)的无预算oracle只能说明损失互补，不能当可部署精度或成本公平上界。
若等待的主要收益仅来自重新选点，在固定token身份的64例诊断中会暴露；据此调整机制解释，不写“吸收完成”。

G2：历史有可部署增量。真实history超过重新训练的同容量retained/geometry，并在冻结独立样本的同成本官方指标中兑现，控制器开销已计入。
如果history仅较zero强、较retained不强，不支持淘汰内容独立价值。若较retained精度强但用了更多等待预算，同样不能下结论。
小样本未检出增量不等于普遍不可能，但不应据此扩展更复杂模型或宣称CVPR主贡献已成立。

## G. 计时、统计与回传

GPU3关闭其他工作进程；冻结batch=1、dtype、attention backend、图像处理、prompt长度分布与生成设置。
预热后同步计时，交错顺序运行，至少3个测量轮次；分开报告encoder、摘要、控制器、prefill/TTFT与固定生成长度总时间。
固定生成长度的EOS策略必须一致；另保留自然结束的真实任务评估。中位数与p90附原始逐样本计时，不把selector微基准等同总收益。
成本表的测量不含ground-truth答案或两支oracle采样；线上每题只运行选中的路径。
早层长KV若仍留存必须记录；不得将后层token减少等价为所有层cache释放，也不声称信息已全部转入文本。

逐任务用官方脚本重算；配对置信区间以图像/任务原始关联单位重采样。MME须保留成对题及类别结构，POPE F1按重采样计数重算，R4每次从四个完整重算指标合成。
同图多问、跨臂对照一起分组，不能把相关问题当独立样本。多个种子只报告全部预定种子，不按测试最优选主结果。

回传目录至少包括：

- truth_manifest.json：四臂commit、命令、所有metric key与原始四任务值；是否同预算的判断。
- method_manifest.json：初始/逐层token数、检查点语义、最终K、query span、cache策略、投影hash和λ。
- pairs.npz与原始jsonl、canonical image_id、每样本early/late预测、NLL及官方评分。
- seed0/1/2各模式checkpoint与report；训练/验证/独立评估图像清单；不得只回最好一组。
- online_predictions、官方指标、配对差值与适当置信区间。
- latency_raw.csv和设备/进程/输入设置；完整token-layer与真实总延迟。
- git diff、复现命令、失败任务及门槛裁决。

最后结论只允许三类：继续history机制；history失败但普通控制可保留为对照；动态本身未胜固定日程，回到强固定方案。
不要把运行成功、NLL重放改善、额外计算提点写成“已完成符合CVPR标准的动态模块”。
