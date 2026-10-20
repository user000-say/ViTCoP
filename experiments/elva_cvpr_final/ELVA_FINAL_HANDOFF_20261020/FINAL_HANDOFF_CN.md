# ELVA CVPR 2027 最终收敛执行交接（2026-10-20）

**状态：研究问题已冻结；R3 GPU运行版尚待服务器绑定。不是新模型准确率报告。**

## 0. 给实现 Agent 的一句话

以昨晚 **R3 V1–V5 实际运行的仓库与评测配置** 为主，不以 10/10 E014 历史源码覆盖它。保留已经验证的原 pen/L9/dual Stage1、原 COVER/PCOV、原 Q_ATTN、原 RDRIFT、原 host gather/KV/mask/position。现在只做可控替换：**W2048 恢复强选择器并比较生命周期日程；W1024 联合比较入口量与中间生存时长；小规模真实同预算生存交换；冻结后扩展九个 benchmark。** 没有模型级验证前，代码产物不能当新精度成绩。

## 1. 当前可核实证据与目标（不要猜补未给的日志）

基础环境按本轮用户提供信息：LLaVA-1.5-7B、CLIP 576 原 patch、LLM 32 decoder block，旧 R4 分母 GQA61.9 / POPE85.8 / MME1852 / MMBench64.1；给出的源分数为官方四任务结果的已报告数，不是我重新跑的 GPU 分数。

- **W2048 = 88.888…%** 整体视觉 token-layer 压缩率。现有最佳 V5 dual+COVER+Q_ATTN，R4=95.90；要求尽量达到 **R4≥97.0**，需约 +1.10 点。
- **W1024 = 94.444…%**。现有最佳 V4 pen+PCOV60_B12+Q_ATTN，R4=92.71；要求尽量达到 **R4≥94.0**，需约 +1.29 点。
- W1536 = 91.666…%，保留作最终方法的中档泛化检验；现在没有可信可比 R3 实测结果，不能填分。
- 历史 E020.3：W2284，COVER+SHIFT+REF≈96.50，改 RDRIFT≈96.79，再加旧 merge≈96.95；这是**不同预算与日程**，不能把增益在 W2048 直接相加。
- R3 W2048 Stage1 对照：V5−V4≈+0.53，V3−V1≈+0.52；低预算 W1024 dual+PCOV 落后 pen+PCOV≈0.29。
- **必须审计**：新 W2048 pen+COVER+REF 95.37 vs 旧 W2284 pen+COVER+REF 96.50，约1.13R4差异混合日程、工作量与实现。RDRIFT 可以是因素，但不是唯一因素。
- R3 W2048 用户提供的实际日程：172×4 + 90×8 + 32×20 = 2048；W1024仅已知入口60及 PCOV60_B12，未提供完整逐层真实 trace；附件中的 `E94_K60_THEORETICAL` 是**算术示例，不是R3实测配置**。

### 压缩口径

`W = sum_{block=1}^{32} actual_num_visual_tokens_during_that_block`，未剪枝基准 `576×32=18432`，压缩率 `100×(1−W/18432)`。完成 block `b` 后执行剪枝，**只改变 block `b+1` 的视觉 token 数**。Stage1 的入口容量不预先锁死172，Stage2最终量也不预先锁死32。每个正式实验必须逐样本保存32项真实 prefill trace，且实际 W≤目标。

### 精度指标口径

`R4 = 25×(GQA/61.9 + POPE_F1/85.8 + MME/1852 + MMBench/64.1)`。只允许读取**官方 scorer 汇总**后计算；不能从平均逐题 F1 / MME 子分伪造分数。新增 benchmark 绝对不能改变这四个分母。外部扩展使用每任务绝对成绩、相对无剪枝保留率及配对区间；如需九任务宏平均只能预注册公式并标明 secondary，不用于替换主 R4。

## 2. 为什么两条预算应分别优化

### W2048：先恢复因子，再调生存

强基线 V5 有双层选点优势，但无 RDRIFT、无 merge，且更早缩到90；因此首要修复方案不是继续调 β，而是在**完全相同 entry、COVER、日程、最终量、scorer 和卡**下只替换第一次选择器：

| 组 | Stage1 | COVER | FIRST | FINAL | W | 用途 |
|---|---|---|---|---|---:|---|
| 已有 V4 | pen | 原 COVER | Q_ATTN | Q_ATTN | 2048 | paired baseline |
| 已有 V5 | dual(原β=0.5) | 原 COVER | Q_ATTN | Q_ATTN | 2048 | paired baseline |
| H1 | pen | 原 COVER | **原 RDRIFT** | Q_ATTN | 2048 | 选择器迁移 |
| H2 | dual | 原 COVER | **原 RDRIFT** | Q_ATTN | 2048 | Stage1×FIRST互补性 |

RDRIFT 原定义（由历史用户补充）：`rank_drift`, `current_weight=0.6`, `s=0.6*now+0.4*(now-old)`，now/old 来源是当前和更早边界 prompt→vision 响应排名；**必须调用原RDRIFT代码**，不得仅靠上式新写同名替代品。存留 token 个数不变、投影器输入仍为 pen 行，不允许多层 hidden 拼接混入。

**P0 复现闸门**：复跑旧 W2284 COVER+SHIFT+REF 小型确定性面板及必要的官方全四任务点；先对齐 selected IDs 与答案（不要求跨卡 bitwise 完全相同浮点），若无法逼近历史96.50则暂停新公式开发，检查 commit/提示模板/scorer/预处理/层位/原patch顺序。

### W2048：RDRIFT胜出后做严格 W 配对日程

先在固定相同选择器与 Stage1 下做小开发面板，只有显著候选再做四任务全量。

1. `E88_R3_ANCHOR`: 172×4 + 90×8 + 32×20 =2048，已有结果对应的主锚点；
2. `E88_ENTRANCE144`: 144×4 + 104×8 + 32×20 =2048，改变入口/中间容量权衡；
3. `E88_ENTRANCE128`: 128×4 + 112×8 + 32×20 =2048，放大容量权衡；
4. `E88_WIDE_SHORT`: 172×4 +148×4 +32×24=2048，前中层宽而短；
5. `E88_NARROW_LONG`: 172×4 +61×16 +32×12=2048，中间窄而长。

1–3比入口量/中间容量，1/4/5比深度寿命。注意第2/3行同时改入口和中段，因此仅检验**预算分配权衡**，不能把变化单独归因其中一项。1/4/5保留入口、最终K、剪枝次数和第一次剪枝深度，解释力更强。先测试 2、4、5（3作为不足时补组），不要无理由把5组全铺九任务。

### W1024：极端预算必须从入口重新分配

**当前强基线是 W1024 V4=92.71（pen+PCOV60_B12）而不是 V5=92.42。** 必须恢复这组真实预处理和逐层 trace；其PCOV不等于高预算的原COVER。固定参照以后优先比较：

| 实验 | 分层实际计算 token | W | 目的 |
|---|---|---:|---|
| `E94_K80_48_16` | 80×2，48×12，16×18 | 1024 | 缩短高入口、延长中期 |
| `E94_K80_64_16` | 80×2，64×8，16×22 | 1024 | 同入口和终量，检验中间宽短 vs 窄长 |
| `E94_K128_48_16` | 128×2，48×9，16×21 | 1024 | 增大入口条件 |
| `E94_K128_64_16` | 128×2，64×6，16×24 | 1024 | 128入口的严格匹配第二对 |
| `E94_K96_40_16` | 96×4，40×8，16×20 | 1024 | 较长入口交互的备选 |

执行顺序：先 80 入口两组做 panel，胜出后按时间预算补一个 128入口或96入口候选；若入口80显著不如实际V4（入口60），优先补同入口60的真实日程配对、修复selector/PCOV，不再盲目加大入口。**每次K0变化都必须明确记录PCOV的覆盖名额与实际语义**：先固定B12来隔离容量，再仅针对胜出的K0补B比例小扫描，绝不可把`PCOV60_B12`代码行为默认视为K128上相同的覆盖比例。

低预算的 FINAL K16 不是先验最优，切勿把结果写成“16是信息下界”；它只是达到W的候选。若图中文字/计数退化，比较稍大 tail K 的**同W**配方（例如 80×2+40×9+24×21=1024），而不是直接加预算。

## 3. 未来效用：必要机制验证而非另开大型项目

实验名 **Equal-Budget Survival Swap**。在共同已运行的 decoder 前缀后，预先选定相同最终幸存集合 S，构造互不相交且同长度的候选暂态集合 Ga/Gb；两分支都额外保留 h 个 block，后都剪到同一S，真正分叉 forward、真实生成、保存loss和答案。运行严格比较同K同h、同输入、同事件数、同W。

- oracle 只用于**离线诊断**，不能把根据最终答案挑选的赢家部署到在线路径或用于测试集选模。
- 记录 `current attention`, `original RDRIFT`, ViT L9/pen 分歧、COVER边际覆盖、预测收益之间的相关性以及 cluster bootstrap区间；按 image_id 归组。
- 若低成本、剪枝时刻可得信号在独立输入上**不能预测**真实收益，则取消动态估值器研发，论文退路为严格等W的层位/容量分析＋强固定方法，不能用注意力可视化假装预测成立。
- 若能预测，才做一个轻控制器，从2–3条**相同W**策略中做条件选择，并与强固定、随机路由、打乱问题绑定、常数输出对照。同样要记录选择器运行时长。

## 4. Stage1 的实质工作：热点图 + 为什么有用

旧测试 R3 W2048 只能证明综合分双层选点更好；不能证明其普遍优于全部端点+Cover（缺 L9+COVER），不能证明 L9 专门负责OCR/空间/计数，且W1024有反例。

对最终 Stage2固定方案进行：`pen`, `L9`, `dual(0.5)`，必要的 `L9+COVER` 与 `dual β=0.25/0.75`，以及**相同K的随机patch补充**。真正投影到LLM的是 `pen` hidden行，仅token选择由其他ViT层指导。输出每个问题上的原24×24 patch保留图、Layer9-pen Jaccard、仅某层独有区域、COVER新增补充、错误→正确/正确→错误两类实例。热点图必须来自真实 IDs，不能用不同CLS响应权重图冒充保留区域。对共享图像多问题要展示差异；无题目区域标注时只能做解释性个例，不能声称客观召回率提升。

## 5. 九个任务：先小面板，后独立完整确认

固定主九项（提前写到 manifest，不按结果删任务）：

1. GQA（原R4）
2. POPE（原R4，官方F1及split）
3. MME（原R4，官方总分及perception/cognition）
4. MMBench EN（原R4，原scorer）
5. ScienceQA-IMG（SQA图像子集，固定prompt和官方ScienceQA评测）
6. TextVQA（OCR/小字）
7. VQAv2（常规问答）
8. SEED-Bench-IMG（明确image-only，不能混视频）
9. MM-Vet（组合推理及外部评判器版本须固定）

可选第十 VizWiz，以及专门诊断的 OCRBench。不能用九任务的平均替换四任务锁定的R4；九任务同时报告原始分、相对同checkpoint全token分数、每项退化和配对CI。加入ScienceQA-IMG**不是为了增加一个对自己有利的数据集**。

分三级：

- **开发小面板**：提前随机分层并按image_id分组，至少覆盖原四任务和ScienceQA-IMG/TextVQA，并锁seed与题目ID；禁止test-only泄漏到调参。
- **正式四任务全量**：只有门槛组（两档各最多2个强候选）在主R4上确认；MME/POPE调用官方整集scorer；不要拿小panel的R4代替官方结果。
- **冻结后九任务全量**：最终强固定基线和最终方法在W2048、W1024比较；中预算W1536作为迁移验证，根据时限完成核心九项。九任务同样保留无剪枝模型参照和强近邻。

不同任务偏好策略是合理的观察，但线上**不能按benchmark标签直接路由**。只有能从用户问题或模型自身已可用特征推断的问题类型，才是公平的输入条件；类别预测器另计时。

## 6. Gate规则：有结果才往下走，止损要明确

| Gate | 时间（America/Los_Angeles） | 成功标准 | 失败处理 |
|---|---|---|---|
| G0 接线/追溯 | 10/20–10/22 | 真实R3 runner；W trace、身份、scorer、历史强点对齐 | 优先修实现，不优化新公式 |
| G1 精度恢复 | 10/22–10/25 | RDRIFT两组真实全四任务或固定开发面板有明确收益 | 保存Q_ATTN强点，转日程而不是调新头 |
| G2 两预算等W日程 | 10/25–10/29 | 各档冻结至少1个进阶强固定方案，记录收益与各任务回退 | 保留当前V5/V4，必要时针对失败类别调整K0/Kmid/Ktail |
| G3 机制 | 10/27–11/02 | 同预算交换显示可预测的跨样本收益 | 放弃复杂动态控制器，仅完成冻结固定方案论文证据 |
| G4 算法冻结 | 尽量11/04–11/06 | 最终代码/config/manifest不再改动；TTFT同卡交错 | 先完成可复现强固定主结果 |
| G5 投稿 | 11/07–11/16 | 九任务关键表、消融、热点图、失败分析、论文 | 规模与方法不再临时扩展 |

**指标门槛**：W2048希望R4≥97，W1024希望R4≥94，是项目目标不是结果已达成，也不是唯一保留条件。若差目标0.2–0.5点但同预算强基线明显提升、九任务和TTFT前沿更好，可以优先冻结而不是截稿前进行不可归因的持续调参。若完整四任务R4小于当前原强锚点，则不宣称新方法最终最优。出现TextVQA/ScienceQA-IMG/MME等明显回退，必须报告并回到具体样本/区域分析，而不是换数据集。

每个新配置最多先一轮开发panel；若明显失败不扩大数据，若略有提升用相同id配对统计再确认，若差异处于误差范围停止继续细调。提交完整代码、官方scorer导出、样本IDs、git hash，避免“平均分相同却不知道选点有无变化”。

## 7. 实验优先级：最少回合推进，不堆大矩阵

### 本周第一波（硬优先级）
1. P0：绑定当前R3仓库、真实K60逐层trace、R3原始四任务文件、旧E020.3 REF、RDRIFT代码，做开关关闭时与V1/V4/V5的同题身份/回答一致性检查。
2. P0：在当前W2048 V4与V5两个Stage1分别开启原RDRIFT的FIRST，保持FINAL=Q_ATTN，关merge，做 paired开发panel、择优全四任务。
3. P0：并行准备W1024 `K80/48/16`与`K80/64/16`两条严格等W日程，先固定pen及PCOV的覆盖参数，按image配对。

### 第二波（只处理第一波胜者）
4. P1：在W2048最佳FIRST+Stage1条件下，对比入口144以及宽短/窄长 2–3种严格等W日程。
5. P1：在W1024最佳日程下才补K128或K96对照；如首轮没有收益就回到入口60检查PCOV变体，不直接扫描β。
6. P1：必要时恢复旧merge：先α=0与原历史α固定，必须补replacement control；若端到端TTFT变慢且精度增益很小立即退场。

### 第三波（机制成立才上线动态）
7. P1：共享prefix的Ga/Gb真实生存交换（先几十/少量几百个独立image+问题的开发样本，具体数依统计稳定性）；不可用只验证ID的toy测试替代模型结果。
8. P2：轻型两策略路由，对比最强固定及随机/打乱对照；满足同平均预算约束并实测路由成本。
9. P0：收集九项完整指标、按题热点图/实际选点ID、真实TTFT分布并写论文；不要等第三波结束才开始写作。

## 8. 真正要修改的宿主代码接线（必须按当前R3实现完成）

1. `CLIP/Stage1`：透出与原始patch IDs一一对应的 pen、ViT9评分；不要改变投影器输入pen hidden行；R3已有V1–V5必须作为 gold implementation，不要引入另一个β融合实现替代历史V3。
2. `decoder`：建立每请求独立`Schedule`，在每个block**开始前**记录真实视觉 token 数；在block计算**结束后**检查该block是否有保留事件，用原 selector 给**active local keep indices**，先校验数量与ID，再通过原**物理compaction**同步修改 hidden/mask/position/cache。对于不同模型/transformers版本必须沿用原实现。
3. `selector`：原 RDRIFT 只用于 FIRST，FINAL照历史 Q_ATTN；保存now/old rank来源、原始身份、排序/tie和真实日志；不可只写公式冒充相同动作。
4. `prefill`/`generation`：不得只缩短一个隐藏张量而漏掉KV；decode 1/decode 2、无图、单图、多图（不支持则fail fast）及batch=1，输出不得错位。确认无剪枝关闭后预测与原仓逐题一致。
5. `swap runner`：copy/fork真实prefix状态（注意KV所有张量及位置一致），两分支分别实际计算h层再同S剪枝，**再次生成真实答案**。先通过身份合同，再通过真实前缀复制合法性测试。
6. `scorer`：结果完全交给现有官方benchmark评测；新控制代码只读官方已汇总四项，不解析原始回答替代官方评分。
7. `TTFT`：固定GPU/dtype/backend/batch/prompt和生成参数，同卡ABBA/BAAB交错，报告median/p90/mean及vision、selector、gather、cache、prefill、首token等开销；不可混跨GPU/日期的绝对数。

### 交付给审核的每组文件

`run_manifest.json`（git hash、model/checkpoint、tokenizer、image processor、seed、配置、events、stage1、B值、backend、scorer commit、GPU）;
`token_trace.jsonl`（逐样本32长数组、W、original patch IDs、每事件active local indices/keep IDs）;
`predictions.jsonl`；`official_scores.json`；`latency_raw.jsonl`；`failure_cases.jsonl`；`patch_maps/`。

开发panel与最终确认数据严格按image_id不相交；同图多问不可拆到两边。图像多crop需给patch身份加image/crop命名空间，当前控制代码对默认576同图patch做了最小合同。

## 9. 当前源码资产与本次新增代码的真实边界

**这次实际可读取**：用户上传的历史对话、较早E014快照源码、E020.2B/3/4外围Python包与交接文件。E020.3运行态 `runtime_torch.RecyclingState.compress()` 仍只支持`first/final`事件；E014仓不是最新R3 V1–V5 runner。故绝不能将二者直接覆盖当前服务器仓。

**本次真正新增并已单测**：

- `elva/core.py`: 完整日程/预算合同、严格单调物理保留、原patchID ledger、等预算生存交换身份合同、可选rank融合诊断、旧R4算术；
- `elva/cli.py`: 从候选JSON生成工作配置、检测真实32层trace与已官方评分R4；
- `elva/plot_patch_maps.py`: 从真实patchID生成24×24保留区域图；
- `server_adapter/integration_contract.py`: **显式保留原host函数调用的接线合同**，没有做不可信的缓存/position伪实现；
- `plans/schedules.json`: 三档严格预算日程候选，旧W2284参考与W1024入口60理论组注明非实测；
- `tests/test_contracts.py`: 事件/身份/预算/反事实/R4的CPU单测。

这是一份**可运行的实验控制层实现与具体的宿主接口合同**，但不是已经完成的CUDA模型版ELVA。**实现Agent的首要编码任务**正是将真实R3 forward接到这些严格合同并跑出真结果；如做不到，请回报实际失败位置和trace，不得只交一批新的外围工具或代码目录就算完成。

## 10. 为什么这仍然有原创研究机会；何种结果会推翻

STAR-Pro（2026）、ATP-LLaVA（2025）与AdaptInfer（2025）已涉及覆盖、渐进及动态层位。不能把「多层ViT、COVER、分阶段Q_ATTN、再merge」说成独立首创。

我们的可检验差异应是：**通过同预算、同后续共同幸存集合的真实反事实，识别某类视觉证据在未来一段LLM深度的额外计算收益；利用早期可得的ViT+交互信息，提前估值并低开销调度其生存期限。** 多层ViT证据是估值输入，不必独立声称发明新视觉编码器。

能识别但在线控制不能涨分/速度：论文机制有意义，但主方法需收敛到强固定方案，不能捏造动态收益。反事实效果和预测都不成立：正面承认尚未证明未来效用，保留工程/层位结果，调整论文主张。单纯九数据集表和目标达标不等于创新已经成立。

## 11. 截稿和必须并行的写作

CVPR 2027 官方：注册 **2026-11-10 AoE**、论文 **2026-11-16 AoE**、补充材料 **2026-11-23 AoE**；截稿日期不延期。当前10/20距正文截止27天，留给代码冻结最多约15–17天。现在就同步写 Setup/Related Work/Method Draft，不用等全量结果。冻结后不再添加新的主要模块或新指标定义。

## 12. Agent完成后的最小回报模板（禁止以“测试通过”代替GPU结果）

```
CURRENT_R3_COMMIT = ...
SCORER_COMMITS = ...
MODEL_CHECKPOINT_SHA = ...
EXACT_R3_W1024_TOKEN_TRACE = [32 ints]
P0_REF_REPRODUCTION = official task scores or failure reason
W2048_RDRIFT_PAIRS = run_ids + per-task original files
W1024_K80_PAIRS = run_ids + per-task original files
ID_AND_CACHE_SMOKE = pass/fail + offending sample
BUDGET_ASSERTION = per-input min/max W, mismatch count
LATENCY = same-GPU paired raw logs
SCIENCEQA_IMG = exact subset, official eval commit, result path
UPCOMING_CHOICE = rationale based on observed differences
```

**不要把目标 97/94 写为已完成，不要把设备CPU单测写成GPU集成，不要忘记ScienceQA-IMG。**
