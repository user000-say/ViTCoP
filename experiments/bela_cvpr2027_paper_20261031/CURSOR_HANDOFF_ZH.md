# BELA：交给 Cursor Agent 的 CVPR 论文写作 Handoff

交接日期：2026-10-31。交付目标：中文非实验稿的真实 LaTeX 源码、实际编译的 PDF，以及与正文分离的作者核对记录。

## 1. 你接手的任务

你是这篇论文的执行与整合负责人。请在现有 BELA 研究上完成实质写作，直接修改 `paper/`，按 CCFA-Skills 的规程推进。研究对象是视觉语言模型的视觉 token 剪枝；本轮完成摘要、引言、相关工作、方法和结论，实验章节继续不启用。已有真实实验结果可以进入摘要与引言。

作者希望把四个有效的组成部分写成一个有解释力的研究，而非四项提分操作的罗列。核心任务是找到真实算法改变了什么决策、这些改变为何相关、已有证据支持到哪一层，并把它们组织成连贯论文。主张必须服从生产实现和结果记录。

已有 `paper/main.pdf` 是此前实际编译的 v03 中文底稿，供你比较；它不是本轮修改后的完成稿。现有 Method 在 Native COVER、特征构造与 RDRIFT 上仍有接口级定义，必须查真实代码后补全。不要把这些接口误认为完整可复现算法。

## 2. 开始前，先读取这些材料

| 材料 | 作用与可信范围 |
| --- | --- |
| `research/01_experiment_evidence.md` | 作者汇报的结果、对照、负结果及精度口径 |
| `inputs/experiment_source_excerpts.md` | 当前对话上传记录中的原始作者汇报摘录，带原文件行号 |
| `inputs/results_reported.csv`、`inputs/evidence_ledger.json` | 结构化结果；并非本轮重跑或原始逐样本预测 |
| `inputs/qa_cover_reference.py` | 从上传记录原样提取的完整 QA-COVER 参考算子；不等同于现役生产版 |
| `research/02_paper_reading_cards.md` | 15 篇原文的写作学习卡、页码和适配建议 |
| `papers/paper_manifest.json` | 15 篇原文的来源、版本线索、阅读页码与所读 PDF 哈希 |
| `research/03_mechanism_blueprint.md` | 四部分机制定位、可核对推导、中心主张及不成立的推断 |
| `research/04_section_blueprint.md` | 各章节与段落的论证职责、建议正文表达 |
| `research/05_code_audit_and_acceptance.md` | 源码核查问题、最小证据补强、审稿和交付标准 |
| `research/06_closest_work.md` | 对最近邻的具体差异核查，另含 5 篇针对性补查 |
| `paper/` | 唯一持续修改的论文工程；保留已有有效内容和引用 |
| `inputs/CVPR2027_author_kit_original.zip` | 作者上传的原始模板，以此核对格式文件 |

当前包没有完整生产仓库、模型权重、原始预测文件，也没有重新运行模型。若当前 Cursor 工作区已打开 BELA 生产仓库，直接读取它；必要时只向作者询问一个明确缺失的仓库路径，同时继续文献核查及其他不依赖该信息的写作。不得搜索作者其他会话或未提供的文件。

## 3. CCFA-Skills 的具体执行方式

参考仓库：[mikubaka88/CCFA-Skills](https://github.com/mikubaka88/CCFA-Skills)。本交接参考的固定提交为 `91ca9e2801698466f934b557a75ce46db8ba46b5`；源码定位见 `workflow/ccfa_source_map.md`。先实际读取所需技能及相关依赖；不能只凭名字声称已经执行。

每个专业任务先应用 `ccf-humanization`，再应用 `ccf-common`。同一任务复用已经读取且仍适用的规则。用自然语言指定技能即可。若本机未安装，可读取固定提交下的完整相应目录及其依赖，在项目内按规程工作；注明采用了源码规程，不声称已完成客户端安装。

按以下顺序执行；每一步交付到相同的项目路径，完成后继续下一步，不把阶段性交付当作终点。

| 步骤 | 主责技能 | 必须完成的工作 | 写入位置 |
| --- | --- | --- | --- |
| A | `ccf-pipeline-orchestrator` | 确认实际工作区、生产代码版本、材料来源与本轮范围 | `research/project_state.md` |
| B | `ccf-paper-to-exemplar` | 在写正文前复核 15 篇原文的相关连贯章节；已有学习卡作导航 | 更新 `research/02_paper_reading_cards.md` |
| C | `ccf-idea-optimizer` | 按代码和结果确定一个中心主张；区分性质、观察、解释 | 更新 `research/storyline.md` |
| D | `ccf-paper-writer` | 先方法，再相关工作、引言、结论，最后摘要；直接改 LaTeX | `paper/sec/` |
| E | `ccf-paper-reviewer` | 评审科学论证与最近邻差异，给出具体问题，暂不改稿 | `research/review.md` |
| F | `ccf-paper-writer` | 修复成立的问题，核对跨章节主张与符号一致性 | 原文件就地修改 |
| G | `ccf-integrity-auditor` | 对照原始记录核数字、对照关系、引用语境与实现 | `research/claim_audit.md` |
| H | `ccf-submission-checker` | 实际编译、渲染并检查模板、匿名、引用和排版 | `paper/main.pdf`、`delivery/verification.md` |

这是写作与核查任务，不启动新的 GPU 实验、不重构生产算法、不新增方法模块。证据设计放在作者记录中；已有的有效实验直接复用。技能切换不代表建立了独立审稿人或独立 agent。未经实际独立执行，不报告“多位独立审稿人一致认可”。

## 4. 先严格读论文，再写正文

15 篇范文为：SparseVLM、ViTCoP、AutoSelect、VisionZip、PyramidDrop、DivPrune、CDPruner、VisPruner、DART、FitPrune、VFlowOpt、MMTok、AgilePruner、LLaVA-PruMerge、MoB。

这些全文已经用于本次交接的定向重读；具体覆盖页码在 manifest 中，未声称逐字重读全部附录。你应根据所写章节返回原文，检查连贯段落、公式和相邻解释，特别复核：

- ViTCoP §§1–3：能力与成本怎样推进到冗余，再由观察推出阶段分工；
- PyramidDrop §3.1：同一研究问题怎样通过控制变量分析变成设计依据；
- DART §§1、3 与 AgilePruner §4：反向对照怎样限制“重要性”“多样性”等解释；
- MMTok §3.1、MoB §3、CDPruner §3：已有工作究竟优化了什么对象；
- VisionZip §4、AutoSelect §§3–4：怎样解释设计、区分训练与推理路径、分解真实开销。

每篇学习卡要能定位真实原文。范例学习保留其论证关系，不复制独特句子、不搬用其科学发现、不照抄所有缺陷归纳。正文的比较按当前版本核实；例如，不能照搬某篇 Related Work 就断言 SparseVLM 一概不兼容 FlashAttention，其所读版本已讨论兼容实现。

`paper/references.bib` 已有 54 项，不受 15 篇范文数量限制。引用按照论证需要增补；不通过未引用书目或无关背景凑数量。先核最接近工作，再补基础模型、算子和任务出处。

## 5. 必须承接的实验事实

作者已明确报告四个组成部分的消融都有正效应。接受这一作者汇报；要为各项找到真实的成对配置、结果文件和数值，不能自行补出四张消融表或交互效应。

当前可追溯的最强对照是：LLaVA-1.5-7B、576 个原始视觉 token、32 层 Decoder，`64×12 + 17×8 + 10×12 = 1024`。原生 L2 与 QA-COVER 入口容量、后续日程和名义 W 相同。

| 指标 | 原生 L2 | QA-COVER |
| --- | ---: | ---: |
| GQA | 55.57 | 56.52 |
| POPE F1 | 82.21 | 86.16 |
| MME | 1651.78 | 1715.68 |
| MMBench-EN | 61.34 | 61.77 |
| ScienceQA-IMG | 68.91 | 69.61 |
| TextVQA-OCR | 53.41 | 53.06 |
| 六任务归一化平均 | 93.56% | 95.34% |

六任务五项提升、TextVQA 下降。均值是逐任务相对 Dense 的比值再求平均，不是六项原始分数平均，也不是 95.34% 的答题正确率。1024 对应层平均 32 个视觉 token，视觉 token-layer 比完整配置少 94.44%；这不是实测 FLOPs、延迟或显存降幅。

匹配随机：GQA 55.96、POPE F1 83.34；错配问题：GQA 55.68，POPE 未完成。问题身份的证据应按任务分别表述。GQA、POPE 换入 token 最终存活率约 11.0%、10.2%，可作为现象描述；单靠它不能证明信息已被吸收。完整协议与局限见证据文档。

## 6. 推荐主张与四部分的关系

推荐以“**在保留视觉代表基础的条件下，让问题信号对入口集合做有限、可控制的适配**”组织论文。标题候选：

> BELA：覆盖约束下的问题条件渐进视觉 Token 剪枝
>
> BELA: Coverage-Constrained Query Adaptation for Progressive Visual Token Pruning

这是一条需要与生产代码对齐的方法主线。它并不预设“两次覆盖普遍优于一次”。

| 部分 | 应解释的本质改变 | 当前合理身份 |
| --- | --- | --- |
| 双深度选择描述 | 改变判断两个 patch 是否可互相代表的依据 | 表示与度量设计；深度互补的具体机制需诊断支持 |
| Native COVER | 从单点分数转向有限集合对原始视觉群体的代表基础 | 初始化作用；精确目标待生产代码核对 |
| QA-COVER | 让加入收益与移除造成的群体覆盖损失共同决定交换，并限制问题代理改动集合的范围 | 最有依据展开的核心更新原则 |
| RDRIFT / 渐进精炼 | 将“允许参与早期计算”和“继续占用后层预算”分开决策 | 深度相关选择；具体 RDRIFT 原理取决于真实公式 |

“两个 COVER”应拆成：**构建集合**与**评价交换对集合的影响**。如果生产代码证实前后就是重复优化同一目标，必须据实改写主张。QA 自身还执行至多 8 次交换，所以“总共恰好两次最优”不是现有算法已建立的结论。

数学解释采用 `research/03_mechanism_blueprint.md` 中的移除损失—加入补偿分解与有限更新界。它们解释操作的含义；不能包装成新的下游性能定理，也不能把平均覆盖约束写成每个答案证据都受保护。

已有工作已研究多模态覆盖、条件多样性、多目标取舍、分阶段覆盖与精炼、深度和集合条件下的删除效应。我们的差异必须落在实际决策规则、约束和受控效果，详见最近邻文档。

## 7. 各部分的写作要求

具体段落任务在 `research/04_section_blueprint.md`。必须保持以下一致性：

- **摘要**：剪枝必要性 → 严格预算下的问题 → 整体方法与关键操作 → 真实结果。最后写，避免四个缩写逐一报幕。
- **Introduction**：VLM 能力 → 大量视觉 token 与注意力二次项 → 冗余和已有方法 → 最接近工作后的剩余问题 → 方法原则 → 实现和证据 → 三条贡献。
- **Related Work**：按决策问题组织并正面对照最近邻；每段的区分都能回到方法公式或协议。
- **Method**：符号和表示路径 → 实际基础构建 → 问题条件更新 → 交换机制解释 → 实际 Decoder 评分和日程 → 完整开销。避免只给 `Phi`、`NativeSelect`、`RDRIFT` 抽象接口就宣布可复现。
- **Conclusion**：收束已经解决的问题、实际支持的结论和具体边界。

先写充分内容，再按篇幅压缩；当前不把空缺的实验页硬塞满。正文不出现审稿流程、内部验收状态、作者待办或尚未完成实验的预期结果。作者问题集中写到 `research/open_questions.md`。

## 8. 模板、声明与最终验收

`paper/cvpr.sty`、`preamble.tex`、`ieeenat_fullname.bst` 与原始模板逐字节一致，哈希在 `inputs/evidence_ledger.json`。保持 `10pt,twocolumn,letterpaper` 和 review 模式；中文支持保留在主文档层。不得改模板字号、页边距或压缩行距来满足篇幅。

[CVPR 2027 官方 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines) 在本次核对时规定：主文最多 8 页，图表包含其中；引用文献和 LLM Usage Statement 可额外占页；声明位于主文之后、参考文献之前。以正式提交时最新官方规则复核。保留并据实际使用情况修订 `sec/llm_usage.tex`；不要把内部任务文件作为匿名论文外链。中文稿用于作者修改，英文投稿阶段另行审校，不能声称中文工作稿已经完成全部投稿手续。

从本包根目录可运行：

```bash
bash paper/build.sh
```

最终必须交付：

1. 就地更新后的 `paper/` 完整源文件；
2. 修改后实际编译的 `paper/main.pdf`；
3. `research/open_questions.md`：仅列真实缺口及影响，注明哪些代码已核对；
4. `delivery/verification.md`：实际执行的编译、PDF 视觉检查、引用与主张核查结果；
5. 可移交的完整源码 ZIP，明确主文件和编译方法。

最终核查 Method、摘要和引言的每个具体结论都能追到代码、结果或文献。发现差异就修改正文，不用新名字掩盖差异。缺失一个定义只约束依赖它的内容，其余工作继续完成；若最终仍有缺口，准确说明，不能将底稿或局部完成结果冒充本轮完整完成稿。

## 9. 给 Cursor 的启动消息

```text
请读取本项目 CURSOR_HANDOFF_ZH.md，并按其要求执行。
先读取所需 CCFA-Skills、研究证据和 15 篇论文原文相关章节，
再核对实际 BELA 生产代码，固定与证据相符的主张，直接修改 paper/。
本轮完成中文摘要、引言、相关工作、方法和结论，实验章节暂不启用。
沿用所附 CVPR 2027 模板，完成审稿、事实与引用核查，实际编译并检查 PDF。
不要停在计划或提纲；持续完成可执行部分，具体缺口集中记录。
```
