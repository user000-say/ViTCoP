# BELA / CVPR 2027 — Stage1 结构必要性与参数搜索决策实验 Handoff
**日期：2026-10-29 | 文档性质：可执行研究协议；不是GPU实验结果**

## A. 本轮决策（不得由 Cursor 自行扩充算法）
冻结现役最强基线：Dual-depth Layer9+penultimate → Native COVER K64 → QA-COVER(B=8) → RDRIFT, L2 `64×12→17×8→10×12`，W1024。
并冻结 E3 W2048 `172×4→120×8→24×10→16×10`。若现役代码不符合此描述，以源码为准并先停机报告。

**当前唯一核心问题**：Native COVER 初始化与 Query 有界修正是否具备不可替代的作用，还是可由单次公平的联合选择等价或更好地实现？
不是将“两次 COVER”或“8次交换最优”当作已证明事实。Stage1 单元和 Stage2 仍作为一体化模型，不新加第五模块。

## B. 导出中已报告的可比结果；必须回溯运行日志
| LLaVA-1.5-7B W1024 | GQA | POPE F1 | MME | MMB | SciQA-IMG | TextVQA-OCR | mean6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L2 Native |55.57|82.21|1651.78|61.34|68.91|53.41|93.56%|
| L2 + QA B8 |56.52|86.16|1715.68|61.77|69.61|53.06|95.34%|
| CDPruner K32，本机另一推理路径 |57.00|87.67|1677.4|59.79|67.53|53.08|94.41%|
| STAR-Pro T32，本机另一推理路径 |57.19|86.67|1697|59.97|65.34|51.54|93.53%|

以上为历史聊天转述的本机报告，不等于本包在GPU复现。CD/STAR同名义W不自动意味着FLOPs、TTFT或attention kernel相同。
QA的POPE TP/FP/FN/TN = 3671/350/829/4150；Native 3312/245/1188/4255。
平均接受交换：GQA 100533/12578=7.9928；POPE 71953/9000=7.9948。最后存活：GQA 11048/100533约11%，POPE约10%。后层不存活≠已证明信息被早层吸收。
W2048 QA完整六任务、统一TTFT、视频/新架构准确率：在本导出中均没有可验收的最终结果。

## C. P0 — 先审生产源码，阻止伪对照
Cursor 只读记录生产 commit、checkpoint SHA、mm_projector、attention backend、dtype、ViT实际层索引（0/1基）、真实每层位置编码和K/V长度、数据集版本与prompt的SHA、采样ID、生成参数。
重点定位并摘录真实文件/函数/行号：
1. Native COVER 的构造输入、目标函数、相似度空间、循环次数、停止条件与完整输出Patch ID；是否有实际显著性约束。
2. Layer9和penultimate是分别评分还是先融合；`selected["metric"]` 是单视图还是两套视图；索引是否对应真实576个Patch。
3. QA `query_scores` 构造：仅问题文本还是包括模板/答案选项；`mm_projector`是否先对576全量投影；QA是否用旧样本位置编码；`saliency_guard_slots`是否为0。
4. QA参考实现中 `float(std)`、`int(flat.argmax())` 和 `int(drops[..])` 可能每轮触发GPU→CPU同步；用profiler确认生产路径是否实际执行。
5. Stage2 RDRIFT真正执行的剪点/评分，剪枝之后cache物理形状、位置及原Patch ID映射；不可把方法名当代码证据。
6. Native COVER 和 QA是否重复创建[N,N]矩阵；记录层缓存能否复用，缓存不应改变选择结果。
**错误时停止GPU精度矩阵**：生产与冻结锚点不一致、身份重复/错位、生成缓存不一致、评分器与历史记录不一致，先修复并重新确立基线。
三份必交文件：NATIVE_COVER_EXACT.md，NATIVE_VS_QA_OBJECTIVES.md，STAGE1_SOURCE_CHAIN.md。

## D. P1 — 参数必要性，而非测试集调参
最先独立开发图像分组，不与正式评价图像重叠。GQA以image_id聚合拆分；POPE各子集及正负样本按image_id和类别分层；记录随机种子。若已看过全量标签，需说明此前成绩探索性，用真正新图像集/新模型作确认性评测。
**主效应B扫描**：B={0,4,8,12,16,24}，其它参数完全冻结。
**其它默认参考参数**：donor_pool=16, drop_pool=12, min_raw_std=.008, min_query_gain=.08, min_novelty=.05, max_coverage_loss=.005, coverage_weight=.5, saliency_guard_slots=0。
在B8附近，使用预先固定的稀疏/正交设计（如24组）考察query_gain {.04,.08,.12}、novelty {0,.05,.10}、coverage_loss {0,.005,.01}、coverage_weight {0,.5,1}、donor_pool {16,32}、drop_pool {12,24}、guard_slots {0,4}。不要将约两千种全因子组合全部跑六任务，也不要通过同一正式测试集选择参数。
对每张图离线输出每一次被接受和被拒绝的候选信息及得分：raw/std、候选ID、真实净新增ID、Query gain、Novelty、单步ΔCoverage、最终全局ΔCoverage、是否达到B硬上限、后续选中的Patch ID。
本包测试了历史参考代码的行为：默认门槛下任一接受交换有J≥.0015；这只证明代理目标单步正增益，**不证明语义真实收益**。B越大每步允许-0.005覆盖下降，不能写成整个B次优化只下降0.005。

**晋级原则**：先看真实的配对GQA/POPE与MMBench/TextVQA保护，再看选择器时间，最后选1个挑战B8的参数进入正式六任务。仅代理目标更大或换入更多，不构成晋级。
**停止**：如果B12/16在独立开发集上仅提高代理分数、没有可靠任务收益，保留B8；如果B4不劣且显著减少选择器成本，支持简化为B4。谨慎解释微小改进，不能写“全局最优B”。

## E. P2 — 不可省略的结构对照（全部同输入、同RDRIFT、W1024）
| Arm | Stage1 | 作用 |
|---|---|---|
| S0 | Native Dual COVER, B0 | 标准基线 |
| S1 | Native COVER→QA B8 | 已有主方法 |
| S2 | Dual salient Top-K 初始化→**完全同一QA B8** | 查询修正存在时，原生COVER是否有边际价值 |
| S3 | 同双深度特征＋同Query分数，一次无约束联合覆盖选K64 | 最直接的“为什么不一次”质疑 |
| S4 | 同输入一次联合贪心K64，额外限制相对Native新增≤8 | 排除S1获益只因B=8的集合可信区域，而非执行两次 |
| S5 | Query-only Top-K64 | Query本身是否足以解释收益 |
| S6 | 同净交换数、相同被删除身份，随机换入（≥5 seeds） | 排除普通集合扰动 |
| S7 | 同一图像内错配另一个真实问题、同样原问题用于回答 | 检验真实问题相关性；不得依赖非唯一question_id |
| S8 | S1去Novelty / 去Coverage guard / 仅Query gain；用相同B8 | 子模块独立贡献 |
控制变量：同一N=576、入口K64、LLM输入Patch ID顺序、同样ViT张量、相同Query proxy、checkpoint和eval。S3/S4目标系数在独立开发集选定，不得故意给差的联合优化权重。S3/S4参考脚本只是简化控制，正式强基线应另复现CoIn/SCoRe/MMTok/CoverPruner/QCPruner中的有代表性方案，区分原论文代码与自建近似版。
**关键判据**：
- S1优于S2：在同一问题修正器存在时COVER仍有必要性的证据。
- S1优于S3但不优于S4：很可能“信任区域/修改额度”而不是“必须串行两次”有效，论文应当统一为单模块且解释约束。
- S3或S4不劣于S1且更快：允许采用一次选择；不为了叙事保留冗余串联。
- S1不优于Query-only或随机：否决独立机制故事。
“S1优于S3”也不能证明S1全局最优；必须与S4及强算法对照。

### E.1 同样交换总额下，“一次/两次/三次QA修正”是否只是记账差别
固定初始Native K64，使用**同一份缓存Query分数和视觉矩阵**，比较`[8]`、`[4,4]`、`[3,3,2]`，所有方案**累计最多8个接受交换**。在每次重新调用时真实审计是否重新归一化评分、重建矩阵、重置覆盖基准或改动saliency guard。另单列`[16]`作为**更大总交换预算**，不许拿它与3次累计8交换来证明次数的优劣。

本包历史参考实现的合成CPU试验，在`N=144,K=32`、冻结评分输入下，一次B8、两次B4+B4、三次B3+B3+B2得到相同最终Patch ID集合；**仅说明某类静态贪心路径里“执行次数”与“可接受交换总数”可等价，不代表所有数据或现役生产结果都相同**。如果生产三种方案在冻结输入下无差别，Method应描述为一个“有界查询修正过程”，而不是宣称多次完整COVER带来独特收益。

## F. P3 — H1/H2/H3的分离验证（不得全部归因于Stage1）
2×2：{Native, QA B8} × {简单Q-Attn, 原生RDRIFT}；完全固定入口K64、W1024与剪点。两种Stage2的实现除真实评分器外一致。
逐样本配对bootstrap（按image聚类）给出每个任务的差值CI及交互：
`[QA_RDRIFT − Native_RDRIFT] − [QA_QAttn − Native_QAttn]`。
这个差中差是该性能尺度上的交互，不直接证明Token吸收的因果机制。再做W2048 E3迁移，不要只保留好看的W1024结果。

针对早期“换入Token最后多被删除”的线索：记录进入时、首个剪点与最终存活集合的ID、读写attention和相应输出；设计同预算强制换入Token在首剪点退出的反事实与对应的同数替补。若GQA/POPE仍改善，说明不需要最终存活，但**尚不能**直接声称“信息已经被吸收”；还需要独立的早期信息干预。若强制删除完全抹去增益，说明Stage2存活规则也重要。
POPE要逐题输出TP/FP/FN/TN、yes率、random/popular/adversarial分组；使用独立开发集校准同一输出阈值后重测，以排除简单yes偏置。
同时提供GQA对象/属性/关系分类及同图异问，Dense正确→压缩错误的保留率，以及随机剪枝敏感样本表现。

## G. P4 — 真实效率与复现
同GPU+同attention kernel+相同样本+batch1+同warmup和测量次数（例warm10/meas50），固定生成1、2、8、16、32、64 tokens。记录图像预处理、完整ViT、全Patch projector、Native COVER、QA各候选搜索、LLM prefill、decode、端到端TTFT/Fixed32、峰值显存与peak reserved。
profiler重点检查`float(std)`、`int(flat.argmax())`和每轮Python循环导致的CUDA同步次数；新选择器不能以忽略其自身代价的TTFT宣称更快。
矩阵内同W不自动对应同FLOPs；动态分辨率/视频N很大，完整N×N相似度矩阵可能使算法不具扩展性。长序列实现先通过内存门禁，必要时使用块式coverage计算并验证选点与参考一致。

## H. 论文图规范（必须从真实张量及原Patch ID生成）
Figure 1：准确的Stage1+Stage2框图、K和剪点；不得使用未经源码核验的数学目标。
Figure 2：Layer9 / deep相似度差异、原生候选集合IoU、答案恢复样本与失败样本；所有空间图需对应同一图像变换及Patch网格。
Figure 3：B={0,4,8,12,16,24}的配对GQA/POPE、覆盖损失、cap-hit、净新增Token和真实selector时间；标95% CI、开发/测试分开。
Figure 4：同图不同问的Native/QA/Joint/Matched-Random Patch ID热图；选择样本规则事先固定，包含成功和失败案例，不能伪造对象框。
Figure 5：S0—S8结构消融和S1对S3/S4的Accuracy–Latency Pareto；所有测量必须同机。
Figure 6：RDRIFT剪点存活、QA候选净新增与首层/深层分布；注意力相关性不是因果证据。
此包的`plot_real_trace.py`必须获得真实`query_scores`及网格顺序；测试合成输入仅用于验证绘图代码，不可作为论文机制图。

## I. 产物接口（保留真实数据，杜绝伪结果）
**predictions.jsonl** 每行：
`{"sample_id":"gqa|val|img_hash|row_idx","image_id":"img_hash","question_id":"original_may_repeat","gold":"yes","pred":"no","correct":false}`
GQA优先存官方逐题`correct`，不使用本包简单字符串比较代替官方exact_match；POPE为yes/no。`sample_id`必须唯一，历史POPE的question_id有重复，不能用作主键。

**traces.jsonl** 每行：
`{"sample_id":"...","n_patches":576,"native_ids":[...],"final_ids":[...],"max_swaps":8,"swaps":8,"records":[{"removed_patch":1,"added_patch":22,"query_gain":0.2,"coverage_delta":-0.001,"donor_novelty":0.3}],"query_scores":[...576 floats...]}`
导出原生COVER的真实相似度、重要性、RDRIFT剪点Patch ID时存独立大文件避免JSONL膨胀；任何热图均保留原图与张量SHA/维度。

**命令（先以真正导出的数据替换路径）：**
```bash
cd bela_stage1_decision_package
python test_stage1.py
python bela_eval.py audit --trace REAL_TRACES.jsonl --out audit.json
python bela_eval.py compare --metric pope --baseline L2_POPE.jsonl --candidate QA_POPE.jsonl --bootstrap 2000 --out pope.json
python bela_eval.py compare --metric gqa --baseline L2_GQA.jsonl --candidate QA_GQA.jsonl --bootstrap 2000 --out gqa.json
python plot_real_trace.py --trace REAL_TRACES.jsonl --sample-id UNIQUE_ID --height 24 --width 24 --out figure.png
```
本工具仅核查轨迹与已有预测，不会加载LLaVA、自动连接生产仓库或凭空生成准确率。正式图表必须标明原始图像、程序commit和样本ID。

## J. 阶段晋级与停止门禁
1. P0源代码和样本身份审计通过，复刻冻结基准 → 才能运行P1。
2. P1开发集只晋级一个B改进选项 → P2结构必要性对照始终必须运行。
3. 正式GQA/POPE的Δ必须配对统计支持；严格保护MMBench/TextVQA/ScienceQA，不能以mean6掩盖退化。
4. 与一次联合选点的比较要同时报告图像感知能力、反事实Query错配和选择器耗时。S1如果不具结构必要性，就简化Method而不是堆术语。
5. P3 RDRIFT若在新QA入口下无净收益，不能把历史旧设置上的增量直接当最终系统证据。
6. 如果QA只提高yes倾向而不能在敏感样本及GQA提供真正收益，收缩论文主张。
7. 与外部方法的论证按同模型、同数据、同计时系统呈现；论文数字与本机复现分列。

## K. 论文与截稿纪律
CVPR 2027：11月10日AoE摘要/登记、11月16日AoE正文、11月23日AoE补充材料。
官方要求OpenReview资料、匿名、正文8页、LLM Usage Statement。9月15日之后首次公开工作可能属于同期工作，但被其影响仍必须恰当引用讨论。
已经确定的Introduction/Related Work/实验协议/图表骨架可以并行写；新核心机制不能在条件实验完成前写成实证发现。

## 可信文献与源码入口（2026-10-29核查）
- ViTCoP，AAAI 2026：https://ojs.aaai.org/index.php/AAAI/article/view/39594
- SparseVLM，ICML 2025：https://proceedings.mlr.press/v267/zhang25s.html
- CDPruner，NeurIPS 2025：https://proceedings.neurips.cc/paper_files/paper/2025/hash/2433fec2144ccf5fea1c9c5ebdbc3924-Abstract-Conference.html
- MMTok，ICLR 2026：https://proceedings.iclr.cc/paper_files/paper/2026/hash/4f8745c0ca5c951a92844ad7668cfca5-Abstract-Conference.html
- SCoRe，CVPR 2026：https://openaccess.thecvf.com/content/CVPR2026/html/Xu_SCoRe_Salience-Coverage_Reduction_for_Vision_Token_Pruning_in_Vision-Language_Models_CVPR_2026_paper.html
- CoIn，CVPR 2026：https://openaccess.thecvf.com/content/CVPR2026/html/Du_CoIn_Coverage_and_Informativeness-Guided_Token_Reduction_for_Efficient_Large_Multimodal_CVPR_2026_paper.html
- AgilePruner，ICLR 2026：https://github.com/cvsp-lab/AgilePruner
- CoverPruner，EMNLP 2026：https://aclanthology.org/2026.emnlp-main.349/
- STAR-Pro，2026 arXiv和LLaVA官方实现：https://github.com/EasonAI-5589/starpro
- QCPruner，2026预印本：https://arxiv.org/abs/2609.19990
- DUET-VLM，CVPR 2026（两阶段压缩架构的直接先例）：https://openaccess.thecvf.com/content/CVPR2026/html/Singh_DUET-VLM_Dual_stage_Unified_Efficient_Token_reduction_for_VLM_Training_CVPR_2026_paper.html
- Hi-Lo Prune，CVPR 2026（明确使用信息吸收/转移）：https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Hi-Lo_Prune_Look_at_What_Youll_Lose_before_Pruning_with_CVPR_2026_paper.html
- Information Horizon，CVPR 2026：https://openaccess.thecvf.com/content/CVPR2026/html/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.html
- Visual Sufficiency Landscapes，2026-10-18预印本（同图异问、问题条件下的剪枝脆弱性，不应重复主张）：https://arxiv.org/abs/2610.24968
- Hi-Lo Prune，CVPR 2026：https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Hi-Lo_Prune_Look_at_What_Youll_Lose_before_Pruning_with_CVPR_2026_paper.html
- CVPR2027官方Dates：https://cvpr.thecvf.com/Conferences/2027/Dates
- CVPR2027官方Guidelines：https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines
