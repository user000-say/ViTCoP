# Cursor执行书：E020.4

目标是找到真实精度—TTFT前沿。不要只审计旧日志停在那里，也不要先假定新聚合或动态K有效。六轮提供不同证据，多个分支可以独立推进。

## 必须先绑定的事实

沿用现有仓库，不重搭模型。记录commit/checkpoint、COVER/BASE、FIRST REF/QED/RDRIFT与原FINAL selector的实际源码哈希、SHIFT/NEXT配置、attention后端、dtype、prompt问题span、官方scorer与输入ID。RDRIFT不得按名字自行编造公式；NEXT从真实旧配置加载，不能猜层位。

当前SHIFT计算例为32个block、入口172、完成3个block后116、完成13个block后32；必须与实际token trace核对。最终32先固定。新profile只改声明项；层号全部用“已完成block数量”。S0原模型、ONE/原EARLY、历史最强点和匹配hard-delete均保留。

## 接线

旧缓存版继续使用runtime_torch.RecyclingState；新I/J组使用current_runtime.CurrentState。每个请求新建状态，groups必须含image/crop身份，coords为原patch网格，positions为原序列位置，patch_ids必须全局唯一。两次事件都调用compress，即使其中一个alpha关闭。先调用真实旧selector取得keep，然后调用compress。

metric_source=current_hidden时直接传当前active视觉hidden，禁止重算视觉编码器。current_key_mean_head时从本事件刚完成block已经计算的**pre-RoPE K**取视觉位置，对KV heads求均值，形成[N,head_dim]；不可再次执行k_proj，不得把与此不同的post-RoPE向量也记成同名实验。若hook拿不到pre-RoPE K，将该项标BLOCKED_FEATURE，hidden支线继续。此低维metric是实验对照，不保证比hidden好。

输出hidden随后交给原模型的hidden/mask/position/KV压缩流程。用旧torch_backend.inject_visual只写回保留视觉位置，保留全部文本和原RoPE位置。不能擅自改已生成的早层KV，不允许在计时外预建特征、缓存或读取同一请求多预算结果。原工程若把hidden和KV生命周期绑定为不可干预，应先实现正确事件hook，不能只改一个离线张量就称接线成功。

默认新算子uniform权重、alpha=.5、cosine>=.8、radius=2；与旧版权重或norm消融不能混用。生产debug=False；关闭alpha不读取metric、不建立cache、不回写表示。开启后模型结果允许改变，不要求等于hard-delete。

## 六轮执行

| 轮次 | 工作与独立分支 | 必须产出的证据 |
|---|---|---|
| 接线轮 | 关闭路径回归、真实阶段张量对齐、RDRIFT/NEXT实际绑定；独占卡剖析旧TTFT | IDs/logits/生成对齐，真实token trace，阶段成本 |
| 表示轮 | A强锚点、B同K first/final/both回收；I当前hidden与key分支 | 改表示是否改善真实生成，donor接收率/位移，原始预测 |
| 前沿轮 | D更窄/更早压缩的hard/回收配对；E廉价末次selector独立跑 | 每工作点质量和真实wall/CUDA TTFT，不能只报W |
| 机制轮 | C聚合控制、J prefix-only、冻结参考保留ID诊断；H三预算响应继续 | 区分身份变化、表示变化、位置混合、预算变化；同图跨问题响应 |
| 扩展轮 | F跨入口/NEXT、G随机重复；关键点进入OCR/计数/关系任务 | 参数冻结迁移，任务失败面，随机波动与同图配对区间 |
| 确认轮 | 开发选型后锁定精度/速度/折中候选及各自控制；独立image组与第二架构 | 官方重评分、独立质量区间、交错TTFT及完整Pareto表 |

前三张卡运行准确率/捕获队列，一张卡独占TTFT，默认可用0/2/4/6。不要让速度线等待聚合涨分，也不要让同K负结果阻止低预算恢复实验。按真实吞吐pilot估计总时长和开发面板大小，不能照旧口头承诺半天全跑。到资源检查点回传已完成/运行中/失败/缺接口及预计剩余工作。

## 101个唯一配置

| 组 | 数量 | 作用 |
|---|---:|---|
| A | 8 | BASE/COVER × SHIFT/NEXT × FIRST REF/RDRIFT |
| B | 18 | 同K同层位、两FIRST规则、三聚合位置、三alpha |
| C | 12 | 空间/语义/随机接收者/问题权重/norm/门限控制 |
| D | 12 | 四预算/层位profile，各有hard-delete与两回收路径 |
| E | 12 | 两profile、三廉价末次选择、回收开关 |
| F | 6 | 跨入口/日程迁移 |
| G | 6 | 随机种子重复，不是新方法 |
| H | 5 | 补齐真实容量响应；重合中档复用 |
| I | 18 | 三profile × 两当前metric × 三聚合位置 |
| J | 4 | 两profile × 两聚合位置的prefix-only控制 |

基础对照不能漏S0和最强固定K32，已有记录只有来源完全匹配才复用。外部方法与独立模型不计入101；新增必要的公开基线复现不应被目录数字限制。

## 工作量与TTFT

| Profile | 事件 | 视觉token-layer W | 相对SHIFT |
|---|---|---:|---:|
| SHIFT | 3:116,13:32 | 2284 | 0 |
| SHIFT_K108 | 3:108,13:32 | 2204 | -3.50% |
| SHIFT_K100 | 3:100,13:32 | 2124 | -7.01% |
| FINAL11 | 3:116,11:32 | 2116 | -7.36% |
| FINAL15_WMATCH | 3:102,15:32 | 2284 | 相同W |

`work_verified.json`由Python正算和反算核对。W不是FLOPs、更不是TTFT。比较需要真实文本长度、后端、固定视觉编码开销、算子和同步成本。FINAL15仅匹配W，不能称同延迟。

当前矩形匹配仅将相似度对数从N²变为(N-K)K，不自动降低完整成本；要记录feature维度和两次算子总开销。current_hidden可能比一次入口低维cache更慢。优先保留已有高效attention；部分QK、额外reader和sort都进入完整计时，不为获取分数强行将整个模型转eager。

`paired_benchmark.benchmark`接受两个完整真实首token函数，必须清空请求KV并统一预处理边界。ABBA/BAAB同输入交错，计时前后同步，保存raw wall/CUDA。重复相同输入时禁止只给候选缓存vision或图像预处理。主TTFT包括约定完整边界，阶段CUDA分解另列；报告p50/p90及平均值，不用一个无方差均值判断。

## 必须做的机制拆解

1. 固定K和层位比较hard与回收，检查实际anchor是否改变、donor接受率、位移与norm；零接受率是no-op，不解释成算法无潜力。
2. 在低预算/更早层位比较匹配hard控制，不能只比较旧强点。
3. 使用匹配hard参考两次事件的原始patch IDs离线冻结集合，当前路径只允许改hidden。若冻结集合不在active中应报错，不允许静默替换。该实验有未来参考，只作诊断，不纳入部署主表。
4. J方向控制区分回收是否依赖向前序位置注入后序特征；默认方法是显式表示干预，不能声称计算语义不变。
5. H同样输入低/中/高三预算完整生成；有oracle空间也不等于信号可预测。只有动态策略在独立图像、同预算直方图置乱、同图跨问题置乱下都得到支持，才晋升。
6. 随机final与KEY_ENERGY是控制；深层随机论文的层位不能直接套到当前13-block处。

## 评分、选择偏差与近邻

开发/确认按image ID隔离，MME双问同图不可拆。H生成缓存必须严格共享输入、prompt、seed和原selector语义；不是同一路径的选择不能重放冒充真实动态结果。

`paired_quality`要求官方scorer回调从重复采样index重评分。回调必须保留重复multiplicity；不能用ID字典覆盖重复样本，不能平均逐样本POPE F1，也不能用accuracy代替MME。分层按任务/官方类别；image cluster ID要包含任务/类别，跨任务同源图像若统一分析需另定聚类协议。

不从101配置的开发最高值直接报显著性。锁定互补候选后冻结参数，独立确认；各任务原值及区间都要报告，R4只是辅助汇总。速度改进和质量改进必须相对同一个声明的匹配基线；不能拿弱配置比精度、拿重配置比速度拼成同时改进。

至少复现或可信对齐FastV/PyramidDrop、SparseVLM、Libra-Merging/VisionTrim，以及ADSC/均匀和深层随机强控制；用它们可用的高效实现与公平后端。不要求等外部所有代码接通才跑本轮表示干预，但主论文结论必须等强近邻比较。

## 回传清单

原仓库commit和绑定源码、实际配置、split/image ID清单、全部状态；逐任务真实生成/官方评分；token trace；每事件原始keep/donor/receiver、接收率、位移、norm；真实NPZ对齐报告；完整TTFT raw wall/CUDA与阶段分解；开发和确认的Pareto表；冻结身份诊断；无效项与原因。

本地只有NumPy与语法检查，PyTorch/CUDA必须在服务器做；没有新的模型分数或延迟。禁止把此包的测试数字写成VLM实验结果。
