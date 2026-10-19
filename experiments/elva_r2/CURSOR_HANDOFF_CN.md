# Cursor 执行书 — ELVA R2，2026-10-19

你接手的是已有真实VLM剪枝工程。立即复用现有环境、checkpoint、scorer和运行入口，不另建演示工程。用户已经授权代码修改、GPU实验、只读审计和结果整理；正常可逆操作无需逐项向用户确认。不要启动旧学习头续训，也不要因为某一可选模块失败而让所有GPU停工。

## 0. 本轮必须交付什么

先把Stage1的“倒数第二层”与“第九层+倒数第二层引导选点”收敛，并为Stage2长阶段日程得到真实同预算结果。并行采集临时存活组的等预算干预，判断是否有不同于旧E020最终身份交换的可利用信号。交付真实生成、官方分项分数、逐层token轨迹和完整TTFT，不只给最好R4。

本包CPU数学/执行模块已经运行检查；`torch_ops.py`尚需服务器验证。没有旧仓库路径被假定为真实，先做一次源码绑定，之后持续运行，不反复停下来重新要求背景。

## 1. 先绑定真实工程，限时完成最小审计

在原仓库独立分支修改，记录原commit和diff。用 `rg` 定位：

```bash
rg -n 'RDRIFT|rank_drift|current_weight|QED|Q_ATTN|COVER|vision_feature_layer|select_layer|past_key_values|cache_position' .
```

形成 `server_binding.json`：模型/checkpoint指纹、processor/tokenizer、prompt模板、官方scorer、数据集/图像清单指纹、代码commit、attention后端、缓存模式、原始可裁视觉token数与decoder深度；`native_selectors`记录原Q_ATTN、QED、COVER、RDRIFT的真实路径/函数/参数/源码摘要。不可用任意新算法替代原COVER或QED。

先完成以下最小预检，不把它扩成无期限审核：

- 真实图片分别走旧原路径与新关闭路径，比较预处理、原pen payload、IDs、首token logits和生成；容差以相同dtype/后端的重复运行噪声为准。
- 原 RDRIFT 是 FIRST 的 `rank_drift(current_weight=0.6)`，末次Q_ATTN；确认高rank代表更重要、history与当前按原patch ID对齐、候选域大于K。若只在原K内部排序再取K，则集合没有变化，应立即修复并单列为实现修复。
- 同一source函数可能因参数不同产生不同方法。去重用实现+有效参数+输入变换，不能只看名字或只看函数地址。QED若等于某个attention+coverage配置，合并重复臂。
- 原始576和32仅为当前示例。若实际不符，停止使用这些预算表并按真实trace重新生成，不按深度比例直接搬到另一架构。

把本包作为 `research/elva_r2/` 等独立模块放入原工程。下面的NumPy实现用于参考与离线，不允许把CUDA hidden/score搬CPU后称为低成本在线策略。

## 2. Stage1接线：先只改选点，不改输入向量

原单次ViT前向继续运行。对实际视觉block列表使用 `LayerCapture`：完成第九层对应blocks[8]；D个block时，标准CLIP hidden_states[-2]对应完成D−1层。核对原feature_select还做了哪些norm/CLS丢弃/reshape，最终payload必须与原pen基线在相同IDs上逐元素对齐。

`elva/torch_ops.py:select_stage1`送入原投影层的始终是 `pen[keep]`；dual仅改变选择几何和评分。原COVER通过薄回调绑定，其groups/crop/coords从context传入。保护图像分隔符、newline及其他结构token，不把它们计为可删patch。

L9与pen saliency必须使用同一原始CLS评分规则、head聚合和有效key归一化。若需新读CLS行，复用同次前向的Q/K，只算该行；不要为了取分数打开整个模型的output_attentions或切到eager。相同评分规则不保证跨层尺度一致，记录两层质量/熵/分布，beta敏感性用于检验尺度偏好。

执行：

```bash
python -m unittest discover -s tests -v
python make_catalog.py
```

服务器必须让PyTorch parity测试实际执行，再用真实CLIP捕获与原forward对齐。当前本地的层号检查是原生hook模拟，不能冒充真实CLIP验证。

Stage1核心五组：

| 组 | 几何/评分证据 | COVER | payload |
|---|---|---|---|
| V1 | pen | 关 | 原pen |
| V2 | L9 | 关 | 原pen |
| V3 | dual，beta=.5 | 关 | 原pen |
| V4 | pen | 原COVER | 原pen |
| V5 | dual，beta=.5 | 原COVER | 原pen |

先在W=1536、共同90/42/18、10/10/12层背景下跑五组。随后在W=1024共同60/28/12背景复核五组，W=2048用于确认。全部固定Stage2为原Q_ATTN。不要按每个benchmark分别选择beta或入口。V1/V3/V4/V5组成真正的二乘二。

必要消融已列入目录：双层beta=.25/.75；W=1536的“只变几何”“只变saliency”；L9/pen同原ID核查。真正向量融合本轮关闭，不用融合结果代替选点归因。若V5没胜V4，就保留V4，继续Stage2，不让多层故事阻塞有效实验。

## 3. 长阶段与精确预算

计数是进入每个实际执行block之前的可裁视觉patch数。事件发生在完成对应block之后；不是在该block运行前删。

| 压缩率显示值 | 完整32层trace的三段 | W |
|---|---|---:|
| 88.9% | 120×10，56×10，24×12 | 2048 |
| 91.7% | 90×10，42×10，18×12 | 1536 |
| 94.4% | 60×10，28×10，12×12 | 1024 |

最强档在完成block10后60→28，完成block20后28→12。入口就是60，不得在前面偷偷增加128/172的热身block。

同入口和最终容量、同两次事件、同W对照：

| W=1024日程 | 时长 | 作用 |
|---|---|---|
| 60→28→12 | 10/10/12 | 用户希望的长阶段候选 |
| 60→28→12 | 8/16/8 | 同中宽，改变寿命分布 |
| 60→32→12 | 10/8/14 | 同首次边界，改变中宽/末次边界 |
| 60→20→12 | 12/8/12 | 延长较宽前段，压窄中段 |

另两档按精确整数倍/半整数倍生成，已在目录中逐层验证。它们不是“只改变层位”的纯干预；宽度和寿命协同变化，结论应称日程形状比较。

旧80/48/16@3/10/19，以及另两档旧recipe，单列跨入口/最终容量的同W系统比较。历史SHIFT/RDRIFT复现另列历史锚点，W=2284的96.79不能直接复制进新档位结果。

Stage2先以Q_ATTN、原RDRIFT完成同条件比较；补原QED与原attention+coverage。Stage1与RDRIFT的桥接包含pen/dual×Q_ATTN/RDRIFT的长日程二乘二及旧recipe迁移。

## 4. 原工程decoder与KV接线

将 `BoundRuntime.before_block` 接在每层真实forward前，将 `after_block` 接在同层forward后。adapter返回原patch IDs，不能返回rank位置或完整序列列号。compact必须保留所有文本/结构位置、原视觉位置和native后端。

位置、mask、每层past KV以及新decode的cache_position必须根据当前原生实现维护。保留原位置不等于所有cache长度相等；禁止对所有层统一按当前hidden长度截断。普通部署沿用native KV策略，并导出每层实际视觉KV数。

每样本记录pre-block trace，`Trace.finish()`严格验证总W和深度。单图多crop、padding和结构token另记。发生任何预算超限或身份复活，标记失败，不将它算入同压缩率成绩。

## 5. 临时存活交换：与固定实验并行，不依赖旧权重

只使用真实已经剪枝的prompt前缀。`adapter.inspect/fork/prune/advance/align_cache/score_suffix`必须绑定实际模型。若没有resume接口，可完整重跑：两支在begin之前执行完全相同配置，冻结相同IDs，在begin处施加不同干预；保证drop发生在正确边界。不要用未剪枝轨迹代替真实前缀。

适配器inspect至少包含：完成层数、当前原patch IDs、每层视觉KV支持、文本/结构原位置、视觉ID到原位置映射、绑定真实前缀的fingerprint。fork深拷贝所有会变的cache、mask、position、metadata及RNG；eval+确定性解码；原prefix内容在两支之后必须不变。

在 `policy.propose` 上完整标注同一目录，默认16个动作含no-op、g=4，单轮。common/inserted/removed先用活动局部索引生成，再映射原ID交给Swap。最终F在决策前固定；两组在窗口结束都退出。调用 `collect_catalog` 时feature_names包含完整变换版本，proposal_rule使用原选择器实现/参数指纹，不保留占位值。

先在少量开发图像上完整目录pilot，报告真实吞吐，再扩大到128张或可用时长支持的独立图像。每张多问按图像隔离；重复/近重复图像按canonical decoded image hash归组。先native KV，再对同一小面板做aligned_support，不能把两个目标的标签混在一起。

每个状态额外做真实不交换重复、A/B顺序反转。标注NLL时只对答案token计均值，logits[t]预测label[t+1]；第一答案token来自最后prompt位置。特征只来自问题和begin之前已计算状态。teacher forcing生成的答案行不得进入Q_ATTENTION或控制器。

本包随机decoder的 `fixture=true` 数据不能训练VLM头。真实适配器须在源码绑定后显式设置 `is_fixture=False`；这个标志不能替代真实模型、输入、输出和源码记录。

## 6. 只有新目标值得学，才启用G块

先交付四个诊断：真实no-op重算噪声；有限候选的收益分布；当前attention/RDRIFT与新标签的独立图像预测表现；NLL变化与真实greedy生成变化是否一致。事后最佳动作仅是候选目录诊断，不作部署算法，不作全局最优上界。

如果当前目录收益接近噪声，先区分候选限制和干预本身；不自动扩网络。若已有超噪声收益，再拟合 `fit_pairs` 的一个小头。decoder当前特征和加L9版本保持相同特征槽位/参数容量，去掉L9时填零；context项可单独消融，不把多输入当机制证明。

验证集使用完整候选目录，按每状态最大预测值校准阈值，并保留no-op。全部不交换就标为NO_SUPPORTED_DECISIONS，不进入大规模学习表。训练/验证/测试按图像隔离，最终评测图像和标签不得用于训练或阈值选取。

三个正式比较：自然末次重选的原路径；强制F的无交换锚点；同F的学习交换。必须同时记录动作率、真实正收益率、负收益尾部、相对有限候选最佳动作的regret，以及最终官方生成分数。

线上只执行所选分支。CPU参考的循环、数组转换不能直接当成CUDA优化；将候选特征打包成一个device批次运行，计入所有开销。缺训练信号时，固定日程和Stage1矩阵照常继续。

`run_queue.py --allow-conditional`还要求真实adapter和新证据gate；旧npz或旧schema不会解除gate。冻结F锚点不需要权重，但仍需要干预适配器。

动态层位是后续独立分支：调用 `feasible_suffixes` 在真实spent预算上生成合法后续日程。必须另采对应suffix动作标签，不能把临时组交换头拿去预测层位/数量。当前不为这个分支阻塞主结果。

## 7. 实验容量与四卡半天安排

当前目录83个去重配置、最多332个四任务作业，其中74个不依赖新学习条件，9个属于条件比较（3个冻结F锚点、6个新头）。这是储备目录，未承诺半天跑完，更不是已完成结果。完成同名source别名去重后数量可能减少。

| 块 | 配置数 | 问题 |
|---|---:|---|
| A Stage1 | 15 | 三预算×五种入口证据 |
| B 等W日程新增 | 21 | 形状×Q_ATTN/RDRIFT，复用A中的3个锚点 |
| C 旧recipe | 6 | 同W跨入口/尾宽对照 |
| D 选择器 | 6 | 原QED、attention+coverage的缺失对照 |
| H Stage1迁移 | 6 | dual入口在旧recipe及两选择器的稳健性 |
| E 深层随机 | 9 | 同一策略三种种子×三预算 |
| F beta | 6 | 双层权重的必要敏感性 |
| I 几何/评分分解 | 2 | 避免混淆多层saliency与多层coverage |
| J Stage1×Stage2桥接 | 3 | 补齐长日程dual×RDRIFT |
| G 条件比较 | 9 | 暂留价值头及冻结F锚点 |

四张卡沿用服务器实际可用编号；若仍是0/2/4/6：

| 时间目标 | GPU0 | GPU2 | GPU4 | GPU6 |
|---|---|---|---|---|
| 开始至约45分钟 | 原路径/新关闭路径与Stage1接线 | 来源绑定、原特征检查，随后A块 | 存活分支adapter与小pilot | 独占测速预热与固定锚点 |
| 后续约2小时 | A块W1536、W1024核心 | B/C/D固定对照 | 完整候选真分支；导出吞吐与噪声 | A/B对应输入交错TTFT |
| 后续约2小时 | 核心候选官方确认 | H/I/J、必要beta/随机 | 有信号才拟合；否则继续固定或失败定位 | 原生强策略、入选方案开销分解 |
| 最后一小时左右 | 补缺失官方任务 | 独立OCR/细节面板或未完主比较 | 配对分析、同图多问、结果打包 | 独立重复测速和漂移检查 |

时间由pilot吞吐修订。先保证五组Stage1主对照完整，再推进最有解释力的Stage2对照；不要把83×4全部直接跑全量。完成核心后继续就绪储备，不在几个小测试结束就停工，也不让可选学习耗尽所有卡。

## 8. 队列与测量

将真实runner封成argv JSON，必须接收config/task/out/run_key/binding五项。示例仅示意路由，`project_runner`需替换为原工程实际入口：

```json
["python", "project_runner.py", "--config", "{config}", "--task", "{task}", "--out", "{out}", "--run-key", "{run_key}", "--binding", "{binding}"]
```

```bash
python run_queue.py --binding server_binding.json --runner runner_argv.json --gpus 0 2 4 --hours 6 --out results/elva_r2
```

runner结果必须含config_hash、binding_hash、run_key、task、samples、backend、cache_mode、scorer_sha、官方metrics、以及覆盖所有样本的trace_audit。不能用汇总R4冒充四任务完成。缺来源或trace的旧结果会被拒绝复用。

TTFT从统一请求准备边界到真实首token输出，包含图像预处理（或明确披露统一缓存边界）、ViT、额外捕获/评分、投影、prefill、选择与索引、控制器、同步。用同输入、交错顺序、独占卡，至少三个重复轮次；报告wall与CUDA口径、p50/p90、峰值显存、每层KV及组件开销。微型算子计时不能替代端到端结果；不要为每个组件增加同步后拿它当主TTFT。

冻结入选配置后，尽量沿用历史1024输入面板测量；吞吐不足先pilot再完整。冷热模型、图像缓存、backend、batch size、dtype、频率与并发策略一致。所有额外成本记在同一方案，不能用不同baseline分别宣称提点与加速。

## 9. 正式评测、消融与回传

正式三预算固定W=2048/1536/1024。W、入口K、最终K、FLOPs、KV和TTFT分别报告。不能仅把最终K16/32/64称为三个整体压缩档。

从服务器读取已经冻结的九任务清单、scorer版本和dense分母；当前资料未给出完整名单，禁止自行补五个名称。两种冻结方法×三预算×九任务是54个主结果作业，dense九任务另计，共享的真实输出可以复用。近邻至少优先ViTCoP原S0、STAR-Pro和适配条件最接近的一种；统一评测与预算，无法复现要标明，不混论文抄表数值为本机实测。

独立确认包括视觉细节/OCR和另一架构。另一架构按真实N/L重算，不复制LLaVA层位。统计按图像聚类，MME同图双问一起重采样并调用官方scorer；不平均逐题F1，不补零缺失任务。已反复用于调参的四任务不能兼作唯一独立确认。

回传一个结果ZIP：

- 真实工程diff、commit、binding、配置与全部运行状态；
- 官方分项、逐样本prompt/answer/prediction、canonical image key；
- 每次事件的原patch IDs、层位、K、选择器、动作与全部pre-block trace；
- 原始TTFT重复记录、GPU环境、缓存策略和开销；
- 新目标的完整候选标签、真实no-op记录、分割、模型manifest和动作诊断（若已启动）；
- 一页结论：Stage1是否定稿、哪条Stage2超过何种同W强对照、是否真降TTFT、哪些假设被否定、仍缺哪些任务。

10月23日给首轮裁决，10月25日固定主要候选，10月30日冻结方法与参数，11月6日完整稿，11月13日可提交稿。CVPR注册/正文为11月10/16日AoE。写作、近邻和主表同步推进；不要等学习分支成功才开始论文。

若真实结果只支持固定日程，就撤回动态/信息饱和主张；但固定日程也必须通过对外结果与机制证据争取论文价值，不能预设为保底录用方案。
