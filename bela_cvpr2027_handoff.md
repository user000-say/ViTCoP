我同意现在必须加速收敛，但我不建议为了赶进度，随便选择一个看起来新颖的模块。我认为 BELA 已经到了可以确定研究主线、集中开发一次实质性升级的阶段。

在重新对照你的全部近期实验，以及 STAR-Pro、QCPruner、AgilePruner、HAP、CoRePrune、Hi-Lo Prune 等 2025—2026 年工作后，我做出一个明确决定：

> BELA 接下来不再优化 Token 数量和剪枝层位，而是优先优化“视觉证据的选择质量”，并把 Stage1 的覆盖信息与 Stage2 的问题相关性结合起来。
>
> 具体而言，保留现有 Dual COVER、RDRIFT 和 L2/E3 日程，集中开发一个轻量的 Query-Aware Coverage Preservation（QCP，问题感知覆盖保护） 机制。
>
> 但这不是再加一个通用的 Query+Coverage 评分公式。我们要针对 BELA 已观察到的低预算失效：在 Decoder 的某次剪枝中，原来被 COVER 选中的视觉证据是否失去代表，并且这种失去是否与当前问题有关。

我把它确定为当前唯一主开发方向，而不是已经验证有效的最终模块。

这次与前几轮最大的不同是：我们不再用一个实验决定要不要换整个研究方向，而是在已经确定的 BELA 主体内完成一次有明确消融、止损条件和投稿目标的优化。

## 一、为什么我现在优先选择这一部分？

先看最新 B1–B3 的结果。

|
方法

|

GQA

|

POPE

|

相比 L2 的结果

|
| --- | --- | --- | --- |
|

L2 原生

|

55.57

|

82.21

|

当前锚点

|
|

B1 固定 COVER32

|

53.09

|

74.53

|

两项大幅下降

|
|

B2 入口排序替代 Stage2

|

52.16

|

79.09

|

两项下降

|
|

B3 后层保留16

|

55.15

|

80.96

|

仍然下降

|
|

CDPruner K32

|

57.00

|

87.67

|

两项均领先

|

这批实验没有证明“Stage1 一定是主因”，但已经排除了三个值得停止投入的简单方向：

* 只调整 Token 生命周期；

* 只使用 Stage1 排名；

* 只把预算从前层移动到后层。

更有意思的是，B1 和 CD32 都固定保留32个 Token，但 GQA 相差3.91分、POPE 相差13.14分。

即使考虑实现路径的潜在差异，这也足以让我们高度关注低容量下的视觉证据选择质量。

同时，B2 又证明：不能为了改善入口而取消 RDRIFT 原生的图文交互选择。

所以我不支持“只做 Stage1”，也不支持“只做 Stage2”。

我的优先研究对象是：

Stage1 所代表的视觉信息，在 Stage2 逐步删除 Token 时，是否仍然被剩余集合覆盖。

这比继续把 RDRIFT 的权重调大、加两枚保护 Token，或者增加一个 erank→K 规则更符合已有结果。

## 二、近两年论文给我们的真正启示

我重点核查了与 BELA 直接竞争的几条技术路线，而不是把所有剪枝论文一律当作相同方法。

|
方法

|

已有核心机制

|

对 BELA 的约束

|

我的决策

|
| --- | --- | --- | --- |
|

DivPrune

|

利用视觉多样性降低冗余

|

单纯 diversity 已不新

|

不新增纯多样性模块

![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com

|
|

STAR-Pro

|

视觉覆盖入口与注意力渐进精炼

|

与 BELA 整体架构最接近

|

必须保留同预算对照

![](https://www.google.com/s2/favicons?domain=https://easonai-5589.github.io\&sz=32)

Efficient Large Vision-Language Models

|
|

QCPruner

|

问题条件下的双侧覆盖

|

Query+Coverage 已被系统研究

|

不直接复制其覆盖目标

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org

|
|

AgilePruner

|

通过有效秩、注意力熵研究图像复杂度与选择策略

|

基于复杂度动态分配 K 不足以构成核心贡献

|

复杂度只作分组分析

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org

|
|

CoRePrune

|

条件删除、集合依赖和扰动分析

|

删除集合修复已有先例

|

避免昂贵的迭代恢复

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org

|
|

Hi-Lo Prune

|

删除前先让保留 Token 吸收相关信息

|

信息吸收/合并不是新概念

|

暂不改动视觉状态传递

![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com

|
|

HAP

|

对注意力头进行问题对齐质量加权

|

修正平均注意力头的想法已有工作

|

可用于诊断，不作为主模块

![](https://www.google.com/s2/favicons?domain=https://aclanthology.org\&sz=32)

ACL Anthology

|

还有一篇 Visual Sufficiency Landscapes 很值得关注。它研究同一预算下，不同视觉子集对问题回答的充分性，并发现同一图像对应不同问题时，可靠子集的性质可能不同。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org

这进一步支持一个方向：不能只根据图像复杂度讨论保留多少 Token，还要考虑选出的视觉证据是否足以回答当前问题。

但这些论文也意味着：BELA 不能把“问题感知”“覆盖”“安全删除”单独作为新颖性。我们应当把贡献限定在一个更具体的对象上：

在不增加 Token-layer 预算、不重新运行 Decoder 的条件下，利用已经计算过的双深度视觉表示，估计一次渐进删除造成的问题相关覆盖损失。

这与已有的 query-conditioned coverage、set-conditioned refinement 存在交叉。能否成为独立贡献，取决于实现和增量消融，而不是名称。

## 三、我决定开发的 BELA 升级机制

唯一主开发候选 · QCP

Stage1：Dual COVER

保留原始 ViT Layer9 + 深层表示、原始候选 K64

原始 Patch ID + 多深度覆盖特征

Stage2：原生 RDRIFT

保留已有跨层图文注意力和排序，不再使用 B2 入口静态排名

QCP：剪枝边界的覆盖损失约束

在同一个 Top-K 候选集合内，评估少量边界交换是否能保住当前问题需要的视觉证据

原始 L2 / E3 预算不变

不增加存活 Token，不恢复已删除 Token

具体设计不是再给所有 Token 计算一次复杂的 DPP。

设 Stage1 入口候选为 S0S_0S0，当前剪点之前的集合为 SSS，RDRIFT 推荐的保留集合为 AAA，原始双层视觉相似度为 sijs_{ij}sij。

我们先构造覆盖：

C(A)=∑i∈S0qimax⁡j∈Asij.C(A)=\sum_{i\in S_0}q_i\max_{j\in A}s_{ij}.C(A)=i∈S0∑qij∈Amaxsij.

其中 qiq_iqi 是当前问题对入口候选证据的需求权重，必须来自已有的图文交互统计；不能调用答案、未来层或者额外完整前向。

如果候选交换将 u∈Au\in Au∈A 替换成 v∈S∖Av\in S\setminus Av∈S∖A，定义：

ΔC=C(A−{u}+{v})−C(A).\Delta C=C(A-\{u\}+\{v\})-C(A).ΔC=C(A−{u}+{v})−C(A).

只有当候选 vvv 具有足够的当前问题相关性，并且交换能改善覆盖、不会造成过大的 RDRIFT 重要性损失时，才允许替换。

一个可测试的决策形式是：

ΔC>λ \(r(u)−r(v)\)+τ\Delta C > \lambda\,\(r(u)-r(v)\)+\tauΔC>λ\(r(u)−r(v)\)+τ

其中 rrr 是归一化后的原始 RDRIFT 分数，λ\lambdaλ 和 τ\tauτ 只允许在验证集确定，不能针对测试答案调参。

我会明确要求 Cursor：

* 用未被 QCP 修改的当前剪点注意力计算 qiq_iqi，不能使用任何未来信息。

* 对双层视觉相似度先做非负归一化，避免覆盖增益出现定义错误。

* 只允许边界附近的少量同容量交换。

* 首个版本只修改一次剪点，不同时修改多个层位。

* 与原生 RDRIFT、随机同数量交换、已有 ICER 做对照。

* 如果实现需要显著增加 CUDA 同步、矩阵计算或 TTFT，就不把它纳入最终模型。

这里还必须特别警惕旧 ICER：**QCP 是否真的比 ICER 更准确、更快，是正式晋级条件。**如果做不到，它就只是旧机制的换名版本，应删除。

### 为什么不改 Stage1 的 K64 选点？

因为我们已经付出了 A、Layer9-only、B1 等实验的代价，却没有发现一个稳定、通用的入口改进。

当前最有价值的资产，是 BELA 已经生成的 K64 多深度候选。直接替换 Stage1 会同时改变入口分布和 Stage2 的交互状态，增加新的混杂因素。

因此这次冻结入口，改进第一次有问题的集合裁剪。如果机制失败，就不再延伸出第三种入口算法。

## 四、下一轮只跑一组主实验，不再无限探索

我建议 Cursor 立即开始实现 QCP，验证时只保留三个对照臂。

|
实验

|

配置

|

目的

|
| --- | --- | --- |
|

C0

|

L2 原生 Dual COVER + RDRIFT

|

锚点

|
|

C1

|

L2 + QCP

|

检验覆盖损失纠正能否提升 GQA/POPE

|
|

C2

|

L2 + 等次数随机边界交换

|

排除单纯扰动或交换带来的收益

|

已有 ICER 结果也必须作为历史对照；若代码接口兼容，再补同条件版本，而不是默认把它当成独立新贡献。

首轮只跑 GQA、POPE、MMBench、TextVQA。 同时保存 POPE 混淆矩阵、Patch ID、实际交换次数、覆盖变化与 TTFT。

建议采用以下晋级规则：

|
指标

|

L2 当前值

|

QCP 首轮目标

|
| --- | --- | --- |
|

GQA

|

55.57

|

至少 56.37

|
|

POPE F1

|

82.21

|

至少 84.21

|
|

MMBench

|

61.34

|

不低于 60.84

|
|

TextVQA

|

53.41

|

不低于 52.91

|

这些是预先设定的筛选阈值，不是承诺可以达到的性能。

我要求比原来 `GQA +0.5 或 POPE +1` 更严格一些，因为已经连续失败多轮。主方案至少应当让 GQA 和 POPE 之一达到上表目标，另一项不能明显下降，而且必须优于随机交换，并具备合理的延迟成本。

如果 GQA、POPE 同时改善，就直接晋级六任务，不再新增机制。

如果仅有微弱改善，或者随机交换能取得相同收益，就停止 QCP，不继续调整权重。

### 为什么要看 POPE 的 FN，而不只是 F1？

你们已经发现 L2 的 POPE 错误明显偏向 FN。

这次 QCP 要验证的是：

在相同 K 下，是否能够恢复部分真实存在对象的识别，而不是简单提高模型回答 Yes 的概率。

因此需要同时看：

* FN 是否下降；

* FP 是否明显上升；

* Yes-rate 是否朝正常范围移动；

* GQA 对象存在、属性和关系类问题是否同步改善。

如果 POPE 上升只是因为模型更愿意回答 Yes，而 GQA 没有改善，就不能将结果解释为视觉证据得到了保护。

## 五、论文现在就要开始写，而不是等最终 ACC

今天是 2026 年 10 月 27 日，距离 CVPR 2027 正文截止还有20个日历日。官方确定论文登记截止为11月10日、全文截止为11月16日，均为 AoE。

![](https://www.google.com/s2/favicons?domain=https://cvpr.thecvf.com\&sz=32)

cvpr.thecvf.com

我建议以以下时间表执行：

|
日期

|

必须完成

|
| --- | --- |
|

10月27–28日

|

QCP 代码实现、边界测试、GQA/POPE 先导；同时开始论文 Introduction、Related Work、实验设置

|
|

10月29日

|

C0/C1/C2 首轮结果，决定是否保留 QCP

|
|

10月30–31日

|

晋级者完成六任务 W1024；不能通过则立即冻结负结果

|
|

11月1–3日

|

在 W2048 E3 上验证最终模块，检查能否维持原有优势，并做实际延迟测试

|
|

11月4–6日

|

完成关键消融、主表及机制案例，冻结全部算法代码

|
|

11月7–10日

|

完成论文初稿、作者资料、注册与摘要

|
|

11月11–16日

|

论文反复修改、结果核验、匿名检查和提交

|

这样只给新算法一次明确的开发窗口，不会一直拖到截止前。

## 六、如果 QCP 也失败，论文应该怎么收敛？

这里我想强调一个与你的要求可能不同、但必须坚持的判断：

即使 QCP 失败，也不应该再引入全新的 Stage1 或第三个模块。

因为 BELA 在 W2048 已经有一个相对不错的 E3 工作点，mean6 为96.85，GQA 57.76，POPE 85.46，同时 MME/MMBench 表现有优势。

而在 W1024 上，L2 的 GQA 55.57、POPE 82.21 是真实短板。

如果我们无法有效修复低预算失效，应当诚实把它写成方法边界，在 W2048 上完整证明 Dual COVER 与 RDRIFT 的增量、实际效率和强基线对比，而不是靠最后一轮大幅改架构冒险。

这仍然有创新性风险：STAR-Pro 已经有视觉覆盖和渐进精炼，CoRePrune 已经研究集合删除。只有 E3 的较高平均值，并不足以保证 CVPR 竞争力。

所以如果 QCP 不成立，投稿应该继续，但必须把主张降到实验实际支持的范围内。

## 七、给 Cursor 的直接指令

下面这段可以直接转交，替代上一轮开放式探索任务。

编辑

# BELA / CVPR 2027 — 最终模块冲刺指令

**决策：锁定 BELA 主线，停止 B1/B2/B3、admission、Layer9-only 和新日程搜索。不得以 CDPruner 组件作为 BELA 投稿贡献。**

## 1. 冻结基线

* W1024：L2 原生 Dual COVER + RDRIFT。

* W2048：E3 原生配置。

* 严格保持模型、视觉特征层、输入 Patch ID、原始剪枝日程及评测协议不变。

* 已有结果不得覆盖，所有新实验使用独立配置与输出目录。

## 2. 唯一新候选：QCP

实现轻量的 Query-Aware Coverage Preservation。

在原生 RDRIFT 计算保留集合后，使用 BELA Stage1 已有的双深度视觉特征和当前问题交互信号，检查边界删除是否造成较大的问题相关覆盖损失。

仅允许同容量候选交换，禁止新增 Token、恢复历史已删除 Token、未来层访问或额外完整 Decoder 前向。

先确认真实剪点代码及 RDRIFT 统计来源，再实现 QCP。第一版只修改一个剪枝边界，不改变日程。

## 3. 对照

* C0：L2 Native。

* C1：L2 + QCP。

* C2：L2 + 等次数随机交换。

* 必须与已有 ICER 方法比较准确率和实际开销。

## 4. 工程测试

包括：

* Patch ID 唯一性及空间顺序；

* 剪枝前后集合包含关系；

* 选择结果的确定性；

* 所有边界情况，包括零有效交换、相同分数、空候选、重复候选；

* KV Cache、RoPE、attention mask 兼容性；

* 关闭 QCP 后与 L2 逐项一致；

* W1024 严格保持不变；

* TTFT、Fixed32、选择器耗时和显存。

## 5. 首轮评测

优先 GQA、POPE，随后 MMBench、TextVQA。

POPE 必须输出 TP、FP、TN、FN、Yes-rate，不能只输出 F1。

记录每个剪点的候选身份、RDRIFT 分数、覆盖变化及交换数量。只使用无标签的在线信号进行正式选择；正确答案只允许用于离线诊断。

## 6. 晋级与停止

首轮目标：

* GQA ≥56.37，或 POPE F1 ≥84.21；

* 另一项不得明显恶化；

* MMBench ≥60.84；

* TextVQA ≥52.91；

* 相比随机交换具有明确增益；

* 实际选择器开销可接受。

达到标准则扩展六任务 W1024，并测试 E3 W2048。

失败则停止 QCP，不启动新的模块家族，冻结现有最强 BELA 版本并集中完成论文。

## 7. 交付文件

* `QCP_IMPLEMENTATION.md`

* `QCP_TEST_REPORT.md`

* `QCP_PILOT_RESULTS.csv`

* `QCP_POPE_CONFUSION.csv`

* `QCP_TOKEN_TRACE.jsonl`

* `QCP_LATENCY.csv`

* `FINAL_MODEL_DECISION.md`

**最迟10月29日完成新候选第一次去留决定，11月6日冻结投稿算法。**

## 八、最终研究判断

当前我把四个目标的优先级定为：

|
目标

|

当前策略

|
| --- | --- |
|

Accuracy

|

主攻 W1024 的 GQA 和 POPE，保护 MMBench/TextVQA，同时确保 E3 不退化

|
|

Architecture

|

不重做双阶段；只允许一个能够解释跨阶段覆盖损失的增量机制

|
|

Scientific Story

|

研究问题相关的视觉覆盖如何在渐进删除中丢失，而不是泛泛讲动态 Token

|
|

Efficiency

|

W 不变只是第一道约束，最终必须测实际 TTFT、Fixed32、显存和选择器开销

|

我的明确主张是：BELA 接下来只冲一次 QCP，不再扩大算法树。

但也必须指出它目前最大的风险：QCP 与已有 ICER、CoRePrune、QCPruner 的方法空间非常接近，而且 B2、admission 的失败已经提醒我们，Stage1 信号不一定能有效指导 Stage2。

因此，这一轮必须用真实结果决定去留。达到目标，就把模块固定下来，在 E3 上验证并全面写作；达不到，就冻结现有 BELA，绝不能继续靠叠加模块追求一个想象中的95分。

这不是停止探索，而是把研究从“持续寻找方向”切换为“一个有明确目标、明确证伪条件、明确交付日期的最终算法冲刺”。
