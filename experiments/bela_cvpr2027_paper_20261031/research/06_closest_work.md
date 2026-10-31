# 最近邻核查：差异必须落到实际算子

这是方法定位文档，不是任何方法的精度排行榜。15 篇写作范文之外，本轮另取 SCoRe、CoIn、QCPruner、STAR-Pro、CoRePrune 原始 PDF 作针对性核对；其范围与来源见 `papers/closest_work_manifest.json`。这些补查没有计入 15 篇范文数量。

## 1. 三篇核心范文中的直接重叠

| 工作与定位 | 已经研究的对象 | BELA 应核对的具体差异 |
| --- | --- | --- |
| [MMTok](https://proceedings.iclr.cc/paper_files/paper/2026/file/4f8745c0ca5c951a92844ad7668cfca5-Paper-Conference.pdf)，p4–5 §3.1 | 以平均最大相似度覆盖文本与视觉目标，校准后加权联合，贪心选集合 | 我们从已有集合出发做受约束交换；必须说明 q 与 G、初始化及覆盖可行性，不能称覆盖函数首创 |
| [MoB](https://papers.neurips.cc/paper_files/paper/2025/file/6818dcc65fdf3cbd4b05770fb957803e-Paper-Conference.pdf)，p4–6 §3 | 视觉保留、问题对齐及其耦合，在假设下给几何关系并作预算分配 | 我们以集合更新权限与每步覆盖变化约束表达适配；不能说首次揭示二者取舍，也不能借其假设声称输出误差保证 |
| [CDPruner](https://proceedings.neurips.cc/paper_files/paper/2025/file/2433fec2144ccf5fea1c9c5ebdbc3924-Paper-Conference.pdf)，p4–5 §3 | 用问题相关性调制 DPP 核，优化条件多样性 | 我们需要给出对全部原始位置的代表变化，以及既有集合中的删除损失；“问题+多样性”本身不是差异 |

## 2. 2026 年补查与必须正面回答的问题

### SCoRe — CVPR 2026

[正式 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_SCoRe_Salience-Coverage_Reduction_for_Vision_Token_Pruning_in_Vision-Language_Models_CVPR_2026_paper.pdf)。本轮核对 p1 Abstract/Introduction、p3 Motivation、p4 §4；p4 公式与伪代码已视觉核对。

论文将视觉代表性组织为显著性加权 k-center，并针对若干显著性与覆盖解耦策略提出批评。这不是“两阶段一律错误”的证明。

BELA 应回答：Native 是否已经优化全局代表性？QA 在每步是否重新检查群体覆盖？与一次统一选择相比，有限修改到底带来什么结果和代价？若只称“先粗后细”，不足以区分。

### CoIn — CVPR 2026

[正式 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Du_CoIn_Coverage_and_Informativeness-Guided_Token_Reduction_for_Efficient_Large_Multimodal_CVPR_2026_paper.pdf)。本轮核对 p1、p3–4 §§3.1–3.5，p4 原页已视觉核对。

其 informativeness 结合视觉显著性与跨模态对齐，再与基于体积的集合选择结合。三类信号共同参与选点已有明确先例。

BELA 的差异必须指向初始化、允许更改的身份范围、每步覆盖可行性和后续执行。不要只列“也考虑显著性、问题和覆盖”。其 volume-based coverage 与平均最近代表覆盖不是同一函数。

### QCPruner — 2026 预印本

[原始 PDF](https://arxiv.org/pdf/2609.19990)。本轮核对 p1 和 p4；正式书目保持预印本身份。

其 facility-location 用 query utility 同时加权视觉目标和候选代表，greedy gain 在完整视觉群体上计算。剪枝位置与相应成本应按各 backbone 协议核对，不能默认全部发生在 LLM 前。

BELA 不能称首次考虑问题条件群体覆盖；应说明自己的 q 是交换方向、覆盖是变化约束，而非把同一问题 utility 放到群体两侧。还需核实这一规则差异是否有受控结果支撑。

### STAR-Pro — 2026 预印本

[原始 PDF](https://arxiv.org/pdf/2609.05916)。本轮核对 p1 与 p3，尤其 Abstract、贡献及 Related Work。

其已提出前端 feature-coverage 候选池与后续嵌套渐进精炼，并报告层平均预算。不能再将“覆盖候选+Decoder渐进”整体结构称为 BELA 首创。

BELA 应把双深度实际选择依据、同容量问题适配、覆盖变化门控与真实 RDRIFT 规则写清。若差异只有实现或超参数，需把贡献限于相应方法与分析，不能扩大到新的阶段原理。

### CoRePrune — 2026 预印本

[原始 PDF](https://arxiv.org/pdf/2609.26484)。本轮核对 p1 与 p4，包含同身份不同深度、固定深度不同删除集合的干预定义。

其已明确研究删除效果对深度与集合上下文的依赖，并设计渐进与集合条件精炼。不能将“低重要性不等于随时可删”“删除效应不是独立的”重新命名为我们的新发现。

BELA 可进一步研究自己的入口更新为何改善固定渐进系统，但需用不同于一般条件删除的具体操作和证据作区分。低末层存活率本身还不能超越其已做的干预分析。

## 3. 要填的对照表

Cursor 应依据代码和原文完成以下列，而非只写优缺点。正文可压缩到最接近的几种，完整表留作者记录。

| 比较维度 | BELA 当前可知/待核 |
| --- | --- |
| 决策位置 | 视觉编码后入口选择，以及指定 Decoder 剪点 |
| 相似度空间 | Native 与 QA 的实际张量来源待核；不能默认相同 |
| 问题信号来源 | 参考支持外部 raw 或均值 embedding；生产路径待核 |
| 选择对象 | 基础 K64 与同容量身份更新；后续嵌套存活 |
| 覆盖目标 | QA 为原始群体平均最近代表相似度；Native 待核 |
| 约束与权限 | 单步覆盖下界、问题增量、新颖性、B 次上限 |
| 求解过程 | Native 待核；QA 为候选池内逐次交换 |
| 理论对象 | 参考交换的代理与集合界；没有下游输出/准确率保证 |
| 更新模型参数 | 当前方法不更新参数；是否依赖校准统计另核 |
| 实际开销 | 完整视觉塔、投影、相似度与搜索、Decoder、cache 均需纳入 |

## 4. 新颖性与表述决策

适合当前材料的中心是：**让问题条件下的局部身份更新受到既有视觉代表关系的约束，并在固定渐进预算中验证其作用。**

这是一项具体方法与作用解释，不自动是一条此前无人发现的定律。若作者已有更深入的控制实验，明确指出此前方法没有区分的现象，并给出能反驳其他解释的证据，才升级为对应机制发现。

避免三种扩大：把 surrogate coverage 当答案证据守恒；把正向组件消融当独立因果机制；把一次初始化加一次 QA 调用当“恰好两次最优”。也不要因最近邻研究过一般原则就否定具体规则与受控结果的价值。

当前书目还含 CoverPruner、HiPrune、Hi-Lo Prune 等相关条目。若正式正文要对它们的方法作具体判断，须返回原始全文核对；本轮不把这些未重新全文检查的条目算入 15 篇范文或新增机制证据。
