# E018：交互层续算收益研究代码

2026-10-14。交付基于你本轮上传的 **E014_AEE_REGENERATED_20261010.zip**。这是完整的修改文件及研究执行工具，不是只给一段数学函数。代码已实现；本地没有 PyTorch、Transformers、GPU 与模型权重，因此没有虚构真实模型测试或 ACC/TTFT。

## 1. 我的算法判断

我保留“随着交互减少视觉证据”的方向，但不把 attention 变小、互信息变小、hidden 相似或 QED-RS 分数低定义成“已经被充分学习”。这些量只能成为候选信号。更可检验的中心问题是：**在当前实际压缩状态下，让这一组证据继续参与后续层计算，是否值得它的额外成本？**

Stage1 提供覆盖较好的真实 patch 候选及来源记录；Stage2 决定候选证据的保留时长。论文的连接应由 Stage1×Stage2 实验和来源消融证明，不能把两个模块拼接后就宣布协同。

这不是宣称续算反事实或动态剪枝本身首次出现。前一轮文献审查已指出普通渐进剪枝、信息迁移、输出反事实和策略路由都已有相关工作。投稿机会取决于这里的具体干预目标、未知图像上的预测能力和同成本净收益，不能靠 E018 名称创造新颖性。

## 2. 四个算法模块

| 模块 | 本版实现 | 在论文中需要证明的事 |
|---|---|---|
| 证据入口 | 沿用原 Stage1；携带原 patch ID、CLS attention、交换来源和输出顺序 | 入口补点确实保护任务证据，而非只改变了预算或运行版本 |
| 当前状态 reader | 复用 SDPA 的 post-RoPE Q/K，只重算问题行的完整合法前缀 softmax | 原始视觉 mass、响应分布、接收端状态和历史是否能预测真实续算收益 |
| 集合级收益控制器 | 当前候选删除集的 38 维特征 → 32 隐单元 MLP → 标定上界 | 超过固定日程、简单 mass 阈值及去来源/历史消融 |
| 退休执行器 | 索引删除、原始 RoPE、问题 span 重映射、逐层 KV 位置账本 | 生成正确，收益与缓存访问机制可区分，完整 TTFT 有净收益 |

候选集先用 Q_ATTN，旧 QED 可切换；不要一开始又引入五套候选公式。QED-RS 的原始推导另给离线诊断实现，暂不作为“已经吸收”的标签或主方法。

## 3. 监督目标与部署决策

在 post-block t 的**同一实际 hidden/cache 状态**上冻结待删除的整组 patch D。构造：

1. now：现在退休 D；
2. later：保留 D 到端点 e，再退休同一组 D。

后续固定操作采用 now 分支选出的原 patch ID，并强制 later 分支使用相同的未来 survivor IDs。两支最终 survivor IDs 必须相同，删除集不能在分支间重新选择。

\[
U_t(D)=\overline{\mathrm{NLL}}(y\mid \mathrm{now})-\overline{\mathrm{NLL}}(y\mid \mathrm{later}).
\]

平均 NLL 按参考答案 token 数归一；可以为负，不假设越晚删除越好。第一项损失来自 prefill 最后一个文本位置预测答案第一个 token，后续用真实的单 token teacher forcing。禁止把完整 labels 张量与已缩短的 prompt logits 直接对齐。

GT 仅在 `e018_counterfactual.py` 和采集脚本中使用。线上 runtime 不接受答案。线上规则是：

\[
\text{continue iff }\widehat U_t^{\mathrm{upper}}(D)>\lambda\,\Delta W/(N_1L).
\]

\(\Delta W=|D|(e-t)\) 是视觉 token-block 工作量增量。它是标定预算的成本代理，**不是精确 FLOPs，也不是 TTFT**。最终需以实际完整模型计时匹配曲线。控制器需要少量监督训练，不能写 training-free。

按 canonical image ID 划分 train/calibration/test，同图不同问题、MME 两问、两次事件都不跨划分。标定使用每个图像组的最大欠估残差；它给出有限样本标定上界，但不承诺分布变化后绝对安全。若 rollout 改变第二检查点状态，应再采集实际策略状态并重新标定。

本版支持 COLLECT 模式加载上一轮训练好的控制器，在当前采集事件前按该策略运行。到达目标事件后先冻结状态，再读 GT。两个反事实分支的未来操作仍固定；它是局部干预目标，**不是完整策略的全局 oracle**。

## 4. 两次决策与退休端点

层号均为从 0 开始、完整 decoder block 之后。

默认固定控制：post-block 4 保留中间 K，post-block 12 保留最终 K。第一检查点若续算，原候选删除集在 post-block 12 退休，再在其 survivors 上选择下一组；第二检查点若续算，保留该组到最后一块完成后退休。

**这里对上一份研究方案做了实现澄清：** second checkpoint 的 continue 增加一个确定的终点处理位置，但不是第三次决策。两组删除，每组只能退休一次，所以最多两次 hidden-state gather。两个集合可能在 post-block 12 接连退休，日志分别记录 gather 次数和发生删除的 block 索引，不将它们混为一次算子调用。

在 `layer_local` 模式下，最后一块之后删除 hidden 视觉行，并不会清掉已生成的 KV；延后组仍可能通过较深层 KV 影响 decode。它不能证明信息已被文本吸收。`scrub_all` 则在退休时清理所有已填充层的同组视觉 KV，保留所有文本 KV 和原始位置。

## 5. 缓存语义是必要控制

| 模式 | 退休处理 | 能解释的收益 |
|---|---|---|
| layer_local | hidden/position gather；已处理层旧 KV 保留 | 更多后续交互与更深层 decode 可访问证据的综合收益 |
| scrub_all | 同时删除所有已填充层该组视觉 KV | 在最终 KV key 位置相同条件下，更深交互对文本状态与剩余 cache value 的影响 |

代码检查 scrub_all 两反事实分支最终 KV 的原始 key 位置一致。两支 KV **值**可以不同，这正是处理带来的影响。清缓存也有额外成本，必须在本模式的实际计时中计入，不能拿 layer_local 的速度替代。

如果论文只在 layer_local 有效，中心点就应是“视觉证据生命周期与缓存可访问性”，不能强写“充分学习后安全丢弃”。如果 scrub_all 也有效，才有更扎实的证据讨论接收端对续算信息的利用。

## 6. 同成本先验实验与动态预算

\(W=\sum_{l=0}^{31}N_l\)。原 Stage1 都是 172，最终 K 相同：

| 最终 K | 单次控制 | 两次控制 | 相同 W | 平均视觉数 |
|---|---|---|---:|---:|
| 32 | post8→32 | post4→102，post12→32 | 2284 | 71.375 |
| 64 | post8→64 | post4→118，post12→64 | 3020 | 94.375 |
| 96 | post8→96 | post4→134，post12→96 | 3756 | 117.375 |

这只能用于固定日程比较。以 102 为中间 K 的动态方法只要延后任意删除，其 W 就大于 2284。**不能据此宣传动态方案和 post8→32 同预算。**

动态第一轮可先研究 `4:80,12:32`，all-retire-now 的 W=2108，为续算留出余量；中间 80 只是待验证的预算配置，不是已经发现的最佳值。通过 calibration 集的 lambda/阈值选择，使平均实际 W 接近目标，再检查完整 TTFT。若这组日程本身过弱，就停止此设置或只增加一个经过预算计算的候选，不做大网格。

`E018_MAX_WORK` 是每请求硬上限，不等于数据集平均成本约束。将硬上限设为已用满预算的 all-retire-now W 会强迫所有请求立即删；这不是真正的动态分配。

平均64附近的入口预算控制应另做：146→32 的 W=2050、172→133(post2)→32(post8) 的 W=2050；MiCo 对应预算需要按其原实现核对。当前上传 E014 Stage1 写死172，**本版没有把未上传的 Stage1 146/128实现臆造为精确历史方法**。需用实际服务器入口源码适配，不能把 post0 剪枝叫 Stage1-only。

## 7. 本版代码清单

| 文件 | 用途 |
|---|---|
| `llava/model/language_model/vitcop_modeling_llama.py` | 修改后的完整 decoder 文件；5处隔离挂接 |
| `llava/model/multimodal_encoder/clip_encoder.py` | 修改后的完整 CLIP 文件；输出值/顺序不变，导出来源 |
| `llava/model/e018_contract.py` | 配置、成本、图像划分、协议指纹、有限样本分位数 |
| `llava/model/e018_stage2_support.py` | reader、controller、两组退休、RoPE/问题/KV、单 token decode |
| `llava/model/e018_counterfactual.py` | 同前缀捕获、分支续算、固定未来 IDs、答案 NLL、成本一致性检查 |
| `llava/model/e018_qed_rs_diagnostic.py` | 未评测 QED-RS 的离线数学诊断 |
| `scripts/prepare_e018_samples.py` | 使用真实已安装 multimodal preparation 导出 prefix |
| `scripts/collect_e018.py` | GPU分片采集，真实策略状态再采集，保存实际标签 |
| `scripts/train_e018_controller.py` | numpy MLP训练、组标定、来源/历史/receiver消融 |
| `scripts/evaluate_e018.py` | 独立于 GT 的 greedy 答案导出；交给原冻结官方 evaluator |
| `scripts/measure_e018.py` | 每次重算 CLIP→Stage1→projector→prefill 的 TTFT、严格Fixed32 |
| `scripts/make_e018_registry.py` | 实际 source/weights/data/evaluator SHA256及环境注册 |
| `scripts/install_e018.py` | 原源码 SHA检查、默认dry check、备份安装、回滚 |
| `tests/test_e018_cpu.py` | 9项已执行CPU检查 |
| `tests/test_e018_torch.py` | 实际修改 SDPA 的小模型/缓存/反事实回归，待服务器执行 |
| `E014_to_E018.patch` | 对照原文件的可审查差分 |
| `reference_E014/` | 本轮上传的原 E014 快照及历史工具 |
| `standalone_sources/` | 与 canonical 路径相同的两份完整修改源码 |

## 8. 必须诚实保留的边界

- 已上传的可执行源码是 E014；日志里的 E015/E016/E017 和 GLOBAL_AEE_FIXED16 不是完整新版源码。本版不能声称已修改那些未提供的文件。
- 精确 HIST-COVER32 依赖服务器 `e013_stage1_support.py`，附件没带。缺失时明确报错，**不会用 COVER proxy 替代**。E014 `FIXED_RISK` 也不能未经 ID/公式对照就改名成最新 GLOBAL_AEE_FIXED16。
- 附件缺少完整 LLaVA repo、builder、multimodal arch 与 E010 helper；必须安装到当前可运行服务器仓库。若源码 SHA 不匹配，installer 不覆盖新版本，Cursor 应按差分挂接到实际源码并做身份回归。
- 本版适配单图、batch1、无padding、固定CLIP patch几何、单token decode、SDPA、冻结eval。明确不支持 anyres、多图、beam、并行请求、训练完整VLM或 Qwen/DeepStack；不能把本代码的有限支持写成论文跨模型泛化。
- preparation 脚本对已安装 multimodal arch做签名与输出形状检查；BPE问题边界不精确时拒绝自动猜span，并要求使用原worker确认的输入token span。可用 `e018_tools.export_prepared` 接入原worker。
- 当前预备版实现 VLM冻结的小控制器，有真实训练入口，没有虚构已经训练好的checkpoint。线上 ONLINE缺模型或未标定时拒绝运行。
- 默认 trace 关闭；计时不执行 CPU tensor trace同步、不复用事先保存的视觉特征、不使用旧 `COMPUTE_EFFICIENCY` 计时分支。

## 9. 交互层前面还应做什么

必须在正式收益采集前完成的只有四类闭环；代码和小模型 smoke 可以同时开始。

1. **锁定真实入口和运行身份。** 首选实际 HIST-COVER32；确认原 helper、交换32个、输出172个、原 patch ID和顺序、无残留载体；保留 BASE和真正的 GLOBAL_AEE_FIXED16作为强对照。现附件不能验证最新 GLOBAL_AEE 身份，应保存服务器版本而不是用近似重建覆盖它。
2. **处理分数与评测矛盾。** PPT的 COVER POPE=85.32，最新文本为86.48；按给出的分母折算约0.338个R4百分点，不能当四舍五入。先查原逐题答案、evaluator SHA和数据划分。已有可信日志就复用，不要求盲目重跑全量。
3. **做一个小的证据入口诊断。** 同图同问在BASE/COVER之间统计补入、删出 patch对应的答案区域/OCR或关系证据命中、失败类型；可用已具标注的小样本。没有区域标注时只报告可验证的子集，不从attention图假造因果。它是连接 Stage1/Stage2的候选机制证据。
4. **建立强固定结果与完整计时。** 对齐 old路径、E018单次与两次固定路径的答案、survivor ID、问题span、原RoPE、prefill/decode1/decode2缓存；先32～64题smoke，正式准确率再全量。把reader、selector、同步与cache清理成本纳入完整模型TTFT。

然后再做有限的 N1预算比较，以及Stage1×Stage2的2×2因子实验。不需要继续开发更多 Stage1评分公式，不需要把所有 N1组合跑完才开始 Stage2反事实实验。

## 10. 论文结果的最低判断标准

继续研究的条件是：固定同W日程有差异且局部oracle显示可利用空间；续算收益在未见图像上可预测；在 matched latency/W 下完整控制器超过固定日程和简单mass；重要提升出现在细粒度任务；来源/历史消融能支撑所讲的连接。

如果去Stage1特征毫无影响，就保留Stage1作为可靠入口，不写来源驱动的创新。如果历史无用就删历史。如果mass阈值更强、更快，就承认控制器不成立。NLL改善还必须落实为官方任务得分，不能把first-token命中或局部oracle收益当benchmark准确率。

后续仍需要最近强基线、现代模型适配与跨模型/任务验证。只有LLaVA-1.5和四个旧数据集上的小涨点，不能据此判断已有竞争力的CVPR论文。本包的用途是把核心假设变成可以失败、也可以提供强证据的实验证据链。

具体服务器执行顺序见 `CURSOR_HANDOFF_CN.md`，验证状态见 `VALIDATION_REPORT.md`。
