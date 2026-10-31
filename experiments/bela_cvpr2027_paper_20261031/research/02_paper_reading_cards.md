# 15 篇范文：从原文中学习论证，再独立写 BELA

本轮基于实际 PDF 重读写作相关的连贯章节，重点为摘要、引言、方法定义及针对性的分析/实验说明。不是只读摘要，也未声称逐字覆盖所有附录。以下页码均指所读 PDF 的物理页；版本、PDF 总页数、哈希及来源见 `papers/paper_manifest.json`。标为“适配”的内容是给 BELA 的建议，并非原论文的结论。

卡片沿用 CCFA `ccf-paper-to-exemplar` 的八项职责：故事、摘要、引言、方法、证据、引用、可迁移写法与不可搬用的内容。此处不复制原论文完整摘要或独特表达。

## P01 SparseVLM — ICML 2025

[原始出处](https://proceedings.mlr.press/v267/zhang25s.html)。所读上传版含 arXiv:2410.04417v4；定位：p1–2 Abstract/§1，p3–4 §3，p7–8 §5。

- **Story Pattern**：先建立长视觉序列的成本，再将问题聚焦到文本怎样指导选择。
- **Abstract Moves**：从计算负担到免训练定位，再交代 raters、比例适配与回收。
- **Introduction Moves**：p1–2 用同图不同问题说明选择需求变化。
- **Method Moves**：先定义跨模态评分所需对象，再依次解释评分者选择、稀疏比例和回收。
- **Evidence Moves**：§5 分别检查 raters 与回收，而非仅给总成绩。
- **Citation Patterns**：背景、压缩范式与具体基线分别引用。
- **Reusable Techniques／适配**：BELA 也应解释每个信号在何时可获得、为何进入下一决策。
- **Do-Not-Copy Boundary**：问题引导和渐进剪枝已有先例；所读版本讨论 FlashAttention 兼容方式，不能照抄其他论文的绝对不兼容评价。

## P02 ViTCoP — 所读 arXiv:2601.17818v1

[原文](https://arxiv.org/abs/2601.17818)。定位：p1–2 §1，p2–3 §2 Insights，p3 §3，p7 §4.4。正式出处元数据另核 AAAI 版本。

- **Story Pattern**：成本与冗余之后，用观察支持不同阶段承担不同职责。
- **Abstract Moves**：压缩必要性、两类取舍、分阶段方法及效率证据形成一条线。
- **Introduction Moves**：先讲能力，再讲视觉 token 开销、冗余与具体不足。
- **Method Moves**：从 Insights 过渡到视觉端、浅层协作、深层精炼。
- **Evidence Moves**：显著性代理和阶段贡献分别讨论。
- **Citation Patterns**：架构背景与各剪枝范式分开给出处。
- **Reusable Techniques／适配**：让 BELA 的阶段分工由信息和决策权限解释，不按模块名字报幕。
- **Do-Not-Copy Boundary**：其 K-norm 相关性与层级观察属于其配置；不能替代 BELA 的 RDRIFT 定义和诊断。

## P03 AutoSelect — 2026 预印本

[原文](https://arxiv.org/abs/2603.07135)。题名 *The Model Knows Which Tokens Matter: Automatic Token Selection via Noise Gating*。定位：p1–3 §§1–3.1，p5 §§3.2–3.4，p9–10 §§4.4–4.6。

- **Story Pattern**：通过容量分配视角统一 Scorer、噪声门控与 Denoiser。
- **Abstract Moves**：冗余与预算问题之后，交代训练和推理的不同路径。
- **Introduction Moves**：先指出局部代理，再提出固定预算下的全局分配问题。
- **Method Moves**：每个模块对应一个明确障碍：梯度、信息限制或分布适配。
- **Evidence Moves**：分别分析选择质量、门控和推理开销。
- **Citation Patterns**：由已有优化范式推进到具体技术依据。
- **Reusable Techniques／适配**：用一个明确原则组织四部分；交代选择表示与实际前向表示。
- **Do-Not-Copy Boundary**：AutoSelect 训练额外模块；其通信容量、噪声和梯度解释不能迁移成 BELA 的理论。

## P04 VisionZip — CVPR 2025

[正式原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.pdf)。定位：p1–3 §§1–2.3，p7–8 §4。

- **Story Pattern**：提出视觉序列必要性问题，用观察引出 dominant/contextual 两类 token。
- **Abstract Moves**：先讲长度和成本，再讲冗余观察与紧凑表示。
- **Introduction Moves**：具体 token 数量让部署问题可感知。
- **Method Moves**：代表主内容与补充上下文各有明确操作。
- **Evidence Moves**：§4 回到冗余原因与选择表现解释。
- **Citation Patterns**：高分辨率模型、冗余证据、压缩基线分工清楚。
- **Reusable Techniques／适配**：分别说明视觉代表基础与条件更新为何必要，并回到实际选择行为。
- **Do-Not-Copy Boundary**：其 merging 改写表示，BELA 当前描述是选原始位置；不能混同。不能将所述视觉聚合现象推广为所有 patch 均无细粒度证据。

## P05 PyramidDrop — CVPR 2025

[正式原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Xing_Conical_Visual_Concentration_for_Efficient_Large_Vision-Language_Models_CVPR_2025_paper.pdf)。正式题名为 *Conical Visual Concentration for Efficient Large Vision-Language Models*。定位：p1–3 §§1–3.2，p6、p8 实验。

- **Story Pattern**：由“每层都需要所有视觉 token 吗”组织观察与设计。
- **Abstract Moves**：成本、过早删除风险、深度分析、逐段压缩相衔接。
- **Introduction Moves**：在提出方法前先交代层位与删减比例的受控分析。
- **Method Moves**：§3.1 先说明实验变量，再在 §3.2 推出日程。
- **Evidence Moves**：区分训练使用、推理使用和高分辨率评测。
- **Citation Patterns**：KV 压缩、视觉 token 方法与架构各有定位。
- **Reusable Techniques／适配**：解释入口与后层容量时保留层位变量及其计数。
- **Do-Not-Copy Boundary**：其“深层更可删”的结果不是 BELA 新发现，也不能证明所有换入者的信息已被吸收。

## P06 DivPrune — CVPR 2025

[正式原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.pdf)。定位：p1–3 §§1–3.3，p5 §4.2，p8 §§4.5–4.6。

- **Story Pattern**：从保留集合仍有重复，推进到 Max-Min Diversity 的集合问题。
- **Abstract Moves**：背景之后直接给数学决策对象。
- **Introduction Moves**：解释重复为何在严格预算下更有害。
- **Method Moves**：先定义输出保持目标，再给多样性代理及求解。
- **Evidence Moves**：比较预算，并单独讨论距离计算引入的开销。
- **Citation Patterns**：MMDP 作为已有数学原语归属清楚。
- **Reusable Techniques／适配**：BELA 必须区分目标、代理、求解器和实际预算。
- **Do-Not-Copy Boundary**：集合内距离、全体位置覆盖、任务相关性并不等价；不能因新颖性高就称覆盖改善。

## P07 CDPruner — NeurIPS 2025

[正式原文](https://proceedings.neurips.cc/paper_files/paper/2025/file/2433fec2144ccf5fea1c9c5ebdbc3924-Paper-Conference.pdf)。定位：p1–3 §§1–2，p4–5 §3，p8–9 实验。

- **Story Pattern**：两个单一选择准则的取舍，导向条件多样性。
- **Abstract Moves**：先成本，再具体不足，再条件化集合建模。
- **Introduction Moves**：同图多问题示例连接 relevance 与 diversity。
- **Method Moves**：先定义 DPP 核，再说明问题相关性如何改变核和子集概率。
- **Evidence Moves**：不同模型、预算和多轮场景分别检验。
- **Citation Patterns**：方法家族与 DPP 原始依据分开引用。
- **Reusable Techniques／适配**：先给清楚的集合操作，再解释各因素怎样改变决策。
- **Do-Not-Copy Boundary**：条件多样性已有直接先例。BELA 的区别必须落在受约束交换与代表性变化，而非“同时考虑问题和多样性”。

## P08 VisPruner — 所读 arXiv:2412.01818v2

[原文](https://arxiv.org/abs/2412.01818)。正式书目采用 ICCV 2025。定位：p1–3 §§1–3.2，p6 方法总览及实验设置，p7 比较。

- **Story Pattern**：先分析评分偏移与分散，再提出视觉侧选择。
- **Abstract Moves**：以调查结论作为方法的直接依据。
- **Introduction Moves**：从可见失败例推进到可测的代理偏差。
- **Method Moves**：保留重要内容与去除重复分别承担任务。
- **Evidence Moves**：用位置分布和控制比较检验评分解释。
- **Citation Patterns**：因果掩码、位置编码和现有剪枝方法各给依据。
- **Reusable Techniques／适配**：问题代理只作为候选依据，说明覆盖门控约束它的哪种后果。
- **Do-Not-Copy Boundary**：不能把论文对特定 attention 指标的分析推广为所有问题信号均不可靠；其后续名称/版本不重复计数。

## P09 DART — EMNLP 2025

[正式原文](https://aclanthology.org/2025.emnlp-main.505.pdf)。定位：p1–3 §§1–3.2，p7–8 §5。

- **Story Pattern**：随机对照暴露排序的不足，再提出 duplication 的决策关系。
- **Abstract Moves**：提出可检验的反常现象，然后交代 pivot 与去重。
- **Introduction Moves**：由反例具体说明 token 的价值随其他保留者变化。
- **Method Moves**：定义评分对象与 pivot 后解释重复关系。
- **Evidence Moves**：pivot 身份、数量、层位和真实速度分别分析。
- **Citation Patterns**：现象依据、算子与对手归属分开。
- **Reusable Techniques／适配**：保留 QA 的匹配随机与错配问题对照，说明它们各排除什么解释。
- **Do-Not-Copy Boundary**：原文特定设置的随机优势不能当作所有 attention 方法的定律；集合相互作用也不是 BELA 首次提出。

## P10 FitPrune — AAAI 2025

[正式出处](https://ojs.aaai.org/index.php/AAAI/article/view/34366)。所读 PDF 含 arXiv:2409.10197v2。定位：p1–2 Introduction，p3–4 Method，p6–7 实验。

- **Story Pattern**：把何时剪、剪多少统一成预算下的分布拟合。
- **Abstract Moves**：把统计目标和快速生成 pruning recipe 联系起来。
- **Introduction Moves**：先说明手工试验成本，再提出可操作的研究问题。
- **Method Moves**：先目标与预算，再统计量，最后求解流程。
- **Evidence Moves**：同时展示分布变化、预算日程和校准数据规模影响。
- **Citation Patterns**：MLLM 结构与注意力、剪枝家族分层引用。
- **Reusable Techniques／适配**：入口选择和日程共同纳入预算，参数与统计来源写完整。
- **Do-Not-Copy Boundary**：免训练不意味着不需校准；BELA 是否有历史统计同样必须据实交代。

## P11 VFlowOpt — 所读 arXiv:2508.05211v2

[原文](https://arxiv.org/abs/2508.05211)。正式书目采用 ICCV 2025。定位：p1–3 §§1–3.1，p8 §§4.5–4.6。

- **Story Pattern**：重要性、回收、日程围绕视觉信息流保持组织。
- **Abstract Moves**：把推理过程与策略优化分别说明。
- **Introduction Moves**：明确需要适配的是不同模型的策略，而非泛称“已有方法不好”。
- **Method Moves**：解释 attention 与信息熵的职责，再连接渐进处理与优化。
- **Evidence Moves**：以表征差异和优化成本补充最终成绩。
- **Citation Patterns**：训练式与免训练方法有清楚边界。
- **Reusable Techniques／适配**：先选一个中心问题，再解释四部分与它的依赖关系。
- **Do-Not-Copy Boundary**：信息流指标是其优化依据，不能把 BELA 的低末层存活率解释成相同的已测现象。

## P12 MMTok — ICLR 2026

[正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/file/4f8745c0ca5c951a92844ad7668cfca5-Paper-Conference.pdf)。定位：p1–5 §§1–3.1.3，p8 §§4.2–4.3；p5 公式与上下文已视觉核对。

- **Story Pattern**：先统一 coverage 定义，再实例化文本、视觉与联合目标。
- **Abstract Moves**：从冗余到统一准则，而非逐模块介绍。
- **Introduction Moves**：用同图异问题、同问题异图解释双模态需求。
- **Method Moves**：明确 source、target、相似度及融合校准。
- **Evidence Moves**：另检查图像贡献较高的任务，限制平均分的解释。
- **Citation Patterns**：覆盖函数和次模贪心保证保留经典出处。
- **Reusable Techniques／适配**：写清 QA 覆盖谁、谁代表谁、在哪个度量空间比较。
- **Do-Not-Copy Boundary**：联合多模态覆盖与标准贪心界已有先例；QA 的局部交换不能直接继承该近似保证。

## P13 AgilePruner — ICLR 2026

[所读原文](https://arxiv.org/abs/2603.01236)。定位：p1 Introduction，p4 §4.1，p6–7 §§4.1–4.3，p10 实验说明。

- **Story Pattern**：定量分析范式行为与适用条件，再检验分析能否指导已有方法。
- **Abstract Moves**：重点落在经验发现及其设计后果。
- **Introduction Moves**：先承认两类方法各自作用，再研究相对优势。
- **Method Moves**：明确 erank、attention entropy 测量什么。
- **Evidence Moves**：hallucination、复杂度分组与 inverse 对照承担不同证据职责。
- **Citation Patterns**：指标来源与应用方法分开归属。
- **Reusable Techniques／适配**：模块涨点之后，寻找能区分解释的中间量和反向对照。
- **Do-Not-Copy Boundary**：更多多样性不自动更少幻觉；不能把 erank 或 attention entropy 直接当作 BELA 已验证的机制。

## P14 LLaVA-PruMerge — 所读 arXiv:2403.15388v6

[原文](https://arxiv.org/abs/2403.15388)。正式书目采用 ICCV 2025，注意版本差异。定位：p1–2 §§1–2，p4–5 §§3.2–3.5，p7 §4.4。

- **Story Pattern**：视觉前缀成本、稀疏注意力观察、自适应采样与补充信息。
- **Abstract Moves**：将固定大量 token 的问题联系到自适应选择。
- **Introduction Moves**：先区分参数规模与前缀长度的成本。
- **Method Moves**：选哪些与怎样更新所选表示分别解释。
- **Evidence Moves**：采样、补充模块、量化和理论计时各有边界。
- **Citation Patterns**：IQR、聚类、模型背景保留来源。
- **Reusable Techniques／适配**：区分 BELA 的选择特征与送入 LLM 的真实表示。
- **Do-Not-Copy Boundary**：其表中 roofline 时间包含理论估计；不能当作同设备实测加速，也不能把合并等同于索引剪枝。

## P15 MoB — NeurIPS 2025

[正式原文](https://papers.neurips.cc/paper_files/paper/2025/file/6818dcc65fdf3cbd4b05770fb957803e-Paper-Conference.pdf)。题名 *Why 1 + 1 < 1 in Visual Token Pruning: Beyond Naïve Integration via Multi-Objective Balanced Covering*。定位：p1–3 §§1–2，p4–6 §3，p9 实验分析。

- **Story Pattern**：从多目标组合未稳定占优的现象，推进到有假设的几何分析和预算分配。
- **Abstract Moves**：研究问题、理论关系、实际算法、证据依次推进。
- **Introduction Moves**：先展示需解释的现象，再说明方法为何对应它。
- **Method Moves**：符号、假设、误差关系、覆盖几何和算法分层。
- **Evidence Moves**：与目标取舍对应的对照回到分析。
- **Citation Patterns**：引用既有覆盖与模型依据，明确本工作分析位置。
- **Reusable Techniques／适配**：把“两步的职责”落实到决策对象和约束，再给对应证据。
- **Do-Not-Copy Boundary**：视觉保留与问题对齐的取舍已被系统研究；BELA 的覆盖差分不能冒称同等的输出误差理论。

## 给写作者的共同结论

这些论文的有效写法是：明确一个对象或矛盾，给足以支持当前推断的观察，再定义真正改变决策的操作。它们的顺序和篇幅不同，不能机械套固定段数。

BELA 应借用这种“问题—依据—操作—结果”的关系，用真实代码、实验与数学性质填内容。最接近的对手应在引言研究缺口段就出现；方法章节再用公式表达具体差别。正式研究新颖性还须核对下一份最近邻文档中的 5 篇补查，不能只与早期 attention 排序比较。

## 2026-10-31 写作时的使用记录

本轮按 `ccf-paper-to-exemplar` 使用上面的页码卡，没有重新下载 15 篇 PDF。写进正文的限制直接来自这些卡片和 `06_closest_work.md`：不把 SparseVLM 写成一律不兼容 FlashAttention；不把 PyramidDrop 的层位观察当成 BELA 的发现；用匹配随机和错配问题分开两种解释；不把 MMTok 的联合覆盖或次模界套到局部交换上；VisionZip / PruMerge 的表示合并与 BELA 的索引选择分开；AutoSelect 的训练式门控不写成 BELA 的理论。方法公式以生产代码为准，不以卡片里的建议句代替算子。
