# E020.3 · 四卡执行书

本轮争取两类实质结果：更高准确率的阶段二压缩算子，和能够兑现为端到端TTFT下降的工作点。不要只完成一个小测试就停，也不要让所有队列等待单一机制的成败。

## 资源与轮次

四卡默认0/2/4/6。GPU6独占测速，其余三卡运行开发筛查与准确率确认。按12小时队列准备，在第6小时输出检查点和完整续跑状态。若本轮只获准6小时，保持优先级与已完成结果，不宣称全表完成。

| 轮次 | 时间预算 | 三张实验卡 | 独占测速卡 |
|---|---|---|---|
| 接线轮 | 0–1小时 | off路径回归、真实特征对齐、锚点smoke | 原路径分解与TTFT锚点 |
| 发现轮 | 1–4小时 | B/C算子、D压缩前沿、E廉价选择并行；H容量响应穿插 | 缓存、聚合、gather、选择器各自开销 |
| 扩展轮 | 4–8小时 | 补齐有效配置、F跨入口/日程、G随机稳定性 | 有潜力候选与配对原路径交错测速 |
| 确认轮 | 8–12小时 | 锁定候选后的完整四任务确认与独立OCR | 真实端到端TTFT、p50/p90和漂移 |

时间只是算力安排，不是按点数或日期裁定科学成败。先用真实吞吐pilot确定开发面板大小，优先保证同条件配对。不能承诺79配置都在半天做完正式全量。

## 来源绑定：禁止替换历史方法含义

导出当前模型checkpoint、commit、prompt/问题span、attention后端、原COVER/BASE、FIRST_SHIFT/NEXT、REF/QED/RDRIFT、最终选择器和KV处理的实际源码与哈希。RDRIFT只指历史FIRST事件控制，不能因为名字就自己实现rank drift，也不能假设末次事件用了RDRIFT。

NEXT从原配置导入，保持实际中间K与层位。已知SHIFT重建例为32层、入口172、完成3个block后保留116、完成13个block后保留32。所有新profile使用“完成block数”，不是旧0基层号。

历史ACC仅在source/input/scorer/selection语义全部匹配时复用。新TTFT要与新候选在同一环境交错测；不直接复用不同日期的单个均值。

## 接入方式：只增加表示修改，不重搭模型

使用 `runtime_torch.RecyclingState`。每个新输入新建状态；entry_features是本次阶段一选中候选对应的已有视觉特征，坐标为原patch网格坐标，entry_patch_ids保持原身份。不得重跑视觉编码器获取相似度。

目录支持作为Python包导入，例如 `e020_3.runtime_torch`。多crop时patch IDs需为包含crop身份的唯一全局编号。

若有多个crop/图像，groups必须区分，防止跨crop错误合并；坐标不能改成压缩序列的序号。source_id应含输入身份及入口选点身份摘要，不能把另一臂的缓存复用到本臂。

两次原剪枝事件都调用状态的compress，即使本配置只启用其中一次聚合。先让旧selector生成真实keep_local，再对当前active视觉hidden调用compress。输出的keep和顺序保持旧selector含义；slots随之更新。

随后继续运行**原有**hidden/mask/position/KV gather。只在audit的off为False时，用 `torch_backend.inject_visual` 将新视觉向量写回原压缩结果的实际视觉位置。支持不连续视觉位置、保留全部文本。不要假定视觉区从序列第0位开始。

不要重置RoPE位置，不要把原patch ID当作当前local index，不要事后修改已计算的早层K/V。后续层根据新hidden正常生成自己的Q/K/V。过去KV的保留或压缩方式应与同profile原路径完全一致，并记录真实行为。

off路径不构建相似度缓存、不回写向量，必须与旧路径的保留IDs、logits、生成结果和实际token trace对齐。开启聚合之后特征有意改变，不再要求与原logits相同。

从真实first/final事件各捕获一份NPZ，含hidden/keep/affinity/coords和可用groups/weights，在服务器分别运行 `python server_preflight.py actual_stage.npz parity.json`。它要求真实CUDA，检查float32核的多alpha对齐，不会无GPU自动跳过。模型实际fp16/bf16与完整生成smoke仍单独完成。初次正确性通过后，各研究分支并行，不等待“oracle先涨分”。

## 新算子到底做什么

默认用阶段一候选已有特征的cosine矩阵，按输入计算一次，首次与末次事件按entry slots取子矩阵。入口172时fp32矩阵为118336字节；这只是缓存大小，不是完整峰值显存。计入矩阵构建与所有取片开销。

原selector给出集合S。每个将被删的token只在同crop、距离不超过2个patch网格单位、cosine至少0.8的幸存token中寻找接收者。未找到则正常删除。阈值是本轮预声明起点，不能写成已验证最优。

以接收者j为例，原权重为w_j，分配给它的被删token总质量m_j、加权中心mu_j：

`new_h_j = h_j + alpha * m_j/(w_j+m_j) * (mu_j-h_j)`。

alpha为0精确关闭；0.25/0.5测试有限扰动；1是充分加权平均控制。默认uniform权重，避免新增QK；query_weights消融只能复用真实已有非负问题attention，不能拿可能含负数的QED分数代替。若该接口不存在，单项标BLOCKED_FEATURE，其余队列继续。

默认不做norm restore。该操作作为单独控制，防止把向量尺度效应误归因为语义保存。随机构造接收者只从相同eligible集合选择，确保没有改变可聚合donor集合。

高cosine不保证字符/数量/属性等价。合并可能损害细节，特别是末次事件。因此first/final/both是独立实证问题，不预先宣布浅层merge深层prune正确。

算法中的一阶质量守恒检查仅适用于alpha=1、完整分配、未恢复norm的代数情形；模型不会自动使用这个质量，RoPE/softmax/后续非线性也会改变。禁止称其保持完整attention或未来答案。

## TTFT优化来自哪里

| Profile | 完成block后的事件 | W | 相对SHIFT |
|---|---|---:|---:|
| SHIFT | 3:116, 13:32 | 2284 | 参照 |
| SHIFT_K108 | 3:108, 13:32 | 2204 | −3.50% |
| SHIFT_K100 | 3:100, 13:32 | 2124 | −7.01% |
| FINAL11 | 3:116, 11:32 | 2116 | −7.36% |
| FINAL15_WMATCH | 3:102, 15:32 | 2284 | 相同W，计算形状不同 |

聚合本身会增加工作，不能凭K不变就说没有开销。要检验它能否补偿更窄或更早压缩的精度损失，并由减少的decoder工作偿还其开销。FINAL15仅匹配W，不能自动称同FLOPs或同延迟。

独立E支线仅替换末次选择器：POOLED_QK是问题reader的平均raw scaled QK，不是平均attention；KEY_ENERGY是平方key范数，属于query-agnostic对照；RANDOM为随机32。输入必须复用当前层已有Q/K，不重新跑投影。

POOLED_QK必须提供真实CPU已知的(min_reader_position,max_visual_position)，并确认所有reader在全部视觉key之后。若不满足，拒绝该项，继续其他队列。GQA头映射、RoPE及reader span绑定原实现。不能混入答案、未来文本，也不能用模板行替换原问题后仍称同一条件。

## 79个配置，分支独立

| 组 | 配置数 | 实际问题 |
|---|---:|---|
| A | 8 | BASE/COVER × SHIFT/NEXT × 原FIRST REF/RDRIFT锚点 |
| B | 18 | 两FIRST规则 × first/final/both × 三alpha，固定K与层位 |
| C | 12 | 六种控制 × first/final：语义、空间、随机接收者、问题权重、norm和宽松门限 |
| D | 12 | 四profile × hard-delete/first聚合/both聚合，直接搜索精度—TTFT前沿 |
| E | 12 | 两profile × 三廉价末次选择器 × 无聚合/first聚合 |
| F | 6 | BASE SHIFT和COVER NEXT上迁移三个聚合位置 |
| G | 6 | 两profile各补三个随机种子，连同E原种子形成稳定性证据 |
| H | 5 | 容量响应新增条件；原有重合条件复用，不训练学习头 |
| 总计 | 79 | 全展开最多316个四任务筛查作业 |

G是重复随机证据，不是六种新算法。确认阶段最多12配置、48作业；与筛查重复的配置不算新增方法。S0、ONE/EARLY和近期外部基线是额外已有参照，原条件不匹配时补跑，不能漏掉S0。

B同K不涨分并不自动关闭D：高预算下收益可能已饱和，聚合仍可能在低预算弥补损失。B/D不利也不关闭E或H。只以实际实现错误、数据来源不完整、资源上限阻止相关单项，不用一个小面板给整轮实验判停。

## 发现、扩展、确认怎样衔接

先锁定开发面板，按image ID与确认面板隔离，MME同图双问题同组。每个配置保存四任务明细、原生成和失败记录。开发面板用于排序和诊断，不能称正式R4结论。

若只能从同一官方评测集切开发面板，完整官方结果仍要报告，但不能把包含开发样本的全量称作独立确认；再用未参与选型的OCR或另一架构验证泛化。

先并行B/D/E/H，再补C/F/G。最多锁定6个互补候选，覆盖准确率最高、延迟最低、最有价值的中间点，配6个相应hard-delete/强固定参照，进入完整确认。若某低预算候选晋级，必须带同profile关闭聚合的控制；若廉价selector晋级，必须带相同聚合设置的原selector。

本轮质量目标优先是超过COVER SHIFT RDRIFT历史点或COVER NEXT REF，在相同TTFT下更准或相近精度下更快。不要只挑战较弱96.50，也不因开发R4多0.01就宣称有效。

H继续三预算100/116/132的同输入完整生成响应。`capacity_response.py`对可加诊断计算有限候选事后最优，包含全部中预算策略和平均预算约束；它使用答案，仅用于诊断，不能部署、不能称R4/F1上界。根据已有E020.2B输入侧信号重放预算分配时，预算置乱与同图换问题控制继续保留。

## 测速协议

保留已有1024输入面板、相同batch/后端/精度/预处理/计时边界，基线与候选按小块ABBA或随机交错。全部成本在完整TTFT内：vision、affinity、QK、sort、聚合、gather、KV处理及首token。新状态每个输入重建，不能只对候选缓存视觉编码或跨输入复用矩阵。

GPU6不叠加ACC。先分解旧路径选择器占比，再测新模块；若原选择器已很便宜，不继续微优化它，但E中已声明比较可完成。`benchmark_cuda.py`可辅助真实单路径计时，function应返回首token且清空上个输入KV；不得用CPU耗时冒充CUDA。

开发W下降3.5–7.36%不会直接变成同幅度TTFT下降。用实际固定阶段占比做Amdahl核算。若聚合耗时超过decoder节省，即使精度更高也不算降低TTFT；它可以保留为精度工作点，而速度支线继续。

## 独立备用队列

核心早完成后继续：独立OCR/细粒度任务；MME类别与同图双问；原始patch覆盖与聚合位移案例；冻结参数到另一预算；另一架构；当前hidden相似度与entry缓存相似度的同算子诊断；PruMerge/Libra-Merging/VisionTrim的同条件外部比较。不要用继续扫alpha填满剩余时间。

任何正结果应说明来自表示聚合、预算、末次选择还是其交互；任何负结果也要区分门限完全没生效、原位特征扰动过大、OCR混合、相似度过期和纯运行成本问题。已经失败的学习交换头不再重训。

## 必须回传

完整配置/来源绑定、全部状态；四任务原指标与预测；逐层真实token trace；每事件旧keep IDs、donor→receiver映射、接收率、非零质量比例、位移和norm分布；原始CUDA/wall计时与阶段分解；关闭路径对齐结果；开发与确认ID清单；最终Pareto表及失败案例。GPU分解、单模型工作点不能写成跨模型结论。
