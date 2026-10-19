# Cursor 执行书：ELVA R3 两档预算、Stage1 收敛与 K60 覆盖平衡 — 2026-10-19

## 0. 本轮范围、已知事实与交付边界

继续在现有 ViTCoP/LLaVA 仓库和环境工作。用户报告代码已装到 `experiments/elva_r2`、提交为 `486fc68`，服务器完成了包测试和一次真 CLIP 预检；这里将这些视为服务器报告，不伪造本地复现。先核对当前 commit/diff，保留已有工作，不强制 reset 或覆盖用户改动。

本轮只保留 **W2048（88.9%）和 W1024（94.4%）**。原 W1536 中间档全部退出活跃目录，不再开测。先收敛 Stage1，Stage2 固定为原 Q_ATTN。旧 E020 学习头、临时存活标签训练、merge、动态入口不进入本轮队列。

把本包放在 `experiments/elva_r3` 等独立目录。它提供实际 CLIP→选点→投影桥接、新的 K60 覆盖选择器、原 COVER 容量/源码约束、两档配置、GPU 队列、逐例结果验收与冻结工具；需要在服务器绑定已有的原 COVER 和真实 VLM/官方 scorer。当前对话未提供服务器 COVER 函数或 `stage1_four_gpu.py` 的完整源码，因此这里没有虚构它们的逐行补丁。

## 1. 先处理已有 pilot 的问题

1. K90 下的 V2/V1、V3/V1 重叠仅说明选点发生变化，不证明 ACC 提升。保留原记录作为 CLIP 诊断，不能计入新两档主表。
2. V4/V5 若未实际调用 COVER，其结果就是错误标签；旧的这些记录不得复用为 COVER 结果。**但真实 COVER 在某张图上返回与 top-k 相同集合是合法的**，不能为了显示生效而强行换点。通过原函数实际调用、参数、源码与整组结果判断。
3. CLIP pilot 没有执行 VLM 生成和官方 scorer，不是四任务 ACC。R3 验收明确拒绝 `clip_selection_only` 和旧 schema。
4. 原 COVER 报告为 N576→K172、最多32次替换。先定位 K172 写死在调用器还是核心函数，不能仅凭报告认定算法无法参数化。不要把172输出静默截成90/60后仍命名原 COVER。

## 2. 两档日程已经逐层求和并反算

计数为**每个实际 decoder block 执行前**可裁视觉 patch 数。事件在完成对应 block 后发生，层号从1开始。

| 档位 | block 区间与视觉数量 | 完成后的剪枝事件 | W |
|---|---|---|---:|
| 88.9% 主日程 | 1–4:172；5–12:90；13–32:32 | 完成4后→90，完成12后→32 | 2048 |
| 94.4% 主日程 | 1–10:60；11–20:28；21–32:12 | 完成10后→28，完成20后→12 | 1024 |

因此88.9%可以保留172入口，直接复用历史 COVER 的原容量。它需要用后续较窄/较短计算偿还入口工作量；若172运行前10层、最终容量不低于32，最低工作量已是2424，不能称 W2048。

保留172不是已证明最优。本轮先解决原 COVER 的可信接入，并在相同入口下收敛多层选点。94.4%继续检验用户要求的60/28/12长阶段，不能在这之前隐含增加172-token LLM热身。CLIP内部576、或离线选择器内部临时172候选，不等于LLM执行172；这些额外选点开销仍计入TTFT。

后续 Stage2 同入口、同最终容量、同事件数的同W日程：

| W | counts | lengths |
|---|---|---|
| 2048 | 172→90→32 | 4/8/20 |
| 2048 | 172→61→32 | 4/16/12 |
| 2048 | 172→94→32 | 2/12/18 |
| 1024 | 60→28→12 | 10/10/12 |
| 1024 | 60→28→12 | 8/16/8 |
| 1024 | 60→32→12 | 10/8/14 |

这些是预定比较，不是最佳日程证明。相同W仍不等于相同FLOPs、KV或TTFT。历史RDRIFT 96.79属于W2284，不填进新档位结果。

## 3. 覆盖接线：高档保留原 COVER，低档使用明确分名的新对照

### 3.1 绑定标准与局部/原始身份

`elva.cover_binding.load_cover(spec)`导入真实 `module:function`。薄回调签名为：

```python
def callback(*, metric_features, scores, patch_ids, k, context):
    ...  # 在这里调用原函数；只适配参数和返回形式，不另造覆盖公式
```

此处说明的是**真实工程接线合同**，不是可冒充原函数的实现。Cursor必须在当前仓库落实函数体，原 COVER 的公式、空间分区、tie、归一化和候选域继续使用真实源码。

- COVER回调返回活动池中的**局部行索引**，不是原 patch ID。
- `select_checked` 返回的 `ids` 才是原身份；后续 decoder selector/compact 使用原身份。
- 不允许通过向量相似度反查身份。若原函数只返回向量，应在原选择位置导出已有索引，K172旧行为保持不变。
- `source_files` 同时包含薄回调和实际原函数文件的绝对路径及SHA256。不能把top-k包装器当原COVER。
- 同样要确认回调确实使用传入的metric/scores；不能在内部重读固定pen证据，导致V5名为dual却忽略L9。

### 3.2 W2048：`legacy_native`

声明 `supported_k=[172]`，`replacement_slots['172']=32`，直接调用原实现。用同一真实图像面板比较原路径与新回调的原始IDs、顺序、payload、首token logits与生成。通过后才设置 `legacy172_parity_verified=true`。

R2选点函数按活动行排序输出。如果历史原路径顺序不同，必须先明确公共排序/位置协议，不能只凭集合相同就声称完全复现。所有V组遵守相同协议，原mask、位置及后续cache语义继续由原模型维护。

### 3.3 W1024：PCOV60_B12，新实现，不能命名原 COVER

本轮默认低档 V4/V5 使用包内 **`protected_coverage`**，配置名含 `PCOV60_B12`。这是新的、可执行的独立对照，**不是原 E013 的复现，也没有新算法有效性或新颖性声明**。W2048 的 V4/V5 仍使用原 COVER；两档不能被写成“相同 COVER 算法仅改 K”。层互补结论在每档内部配对判断。

K60 下不应照搬 b32：那会让最多 53.3% 的入口位置参与覆盖替换。32×60/172≈11.16 只说明按比例缩放可取 b11，并不证明最佳平衡。主对照预先固定 **b12：保住 Top48 显著性核心，剩12个名额补覆盖**。敏感性仅 b6/b12/b18，即保护90%/80%/70%；b0是已有纯Top60。不联合扫描多个池大小、温度和空间权重。

实际算法在 `elva/low_budget_cover.py`，NumPy参考与PyTorch实现均给出：

1. 由该组的同一 Stage1 分数 s 排序，保护 Top(K−b)；dual 的 s 与 metric 沿用R2的明确beta权重。
2. **全部576个patch均可作为候选和被覆盖目标**，不先限制在Top172/180，因此低显著点仍有被选择的机会。
3. 单层metric取相应层向量；dual取 `concat(sqrt(1-beta)*unit(pen), sqrt(beta)*unit(L9))`。正余弦相似度 `a(j,i)=max(0,cos(z_j,z_i))`。
4. 目标权重固定为 `w_j = 0.5/N + 0.5*s_j/sum(s)`；全零分数时退为均匀目标权重。均匀一半给低CLS响应区域留下需求，显著性一半抑制无差别背景覆盖。这个一半是预定控制设定，不是已证明最优的参数。
5. 从保护集出发，逐次加入最大化 `sum_j w_j * max_{i in S} a(j,i)` 增量的点，直至60个。完全相同增量时按显著性、原活动行顺序稳定打破平局。
6. 最后只用选中的原始局部行取 **原pen**，不平均、不池化、不改hidden维数、不额外引入视觉token。被选新增点直接进入最终60，不再经过另一道Top60截断。

这是一种**特征覆盖**控制，不保证文字、空间或目标区域被覆盖。图像背景冗余、CLS对小字的偏差、patch粒度与无问题条件等限制仍在；不得凭F分数上升就声称OCR会提高。所有576×576相似度和b轮greedy、捕获与同步成本均计入TTFT。当前实现保留可审查的FP32矩阵计算，尚未宣称延迟优化。

日志区分 `cover_calls=1`、`native_calls=0`、`origin=r3_protected_control`。历史实现则 `native_calls=1`、`origin=historical_source_bound`。相同集合是合法输出，不强迫换点。

服务器先生成当前文件路径的指纹：

```bash
python bind_protected_cover.py --binding server_binding.json --out server_binding.json
```

它**不会**把 `implementation_parity_verified` 自动改true。先跑本包Torch参考对齐，再在真实CLIP面板验证K、Top48保护、原IDs、pen payload和投影，实际通过后记录预检并设置true。冻结正式binding后才启动官方任务；若之后改源码、数据或binding，旧结果会被拒绝复用，应明确新实验版本。

**原 COVER 的参数化 K60 仍保留为可选强对照 `COVER_PORT60_*_B11`：** 在当前源码显式传K/b，保留其它历史规则，先K172/b32严格回归再K60/b11验证。它不是PCOV的别名。没接好时记录BLOCKED，不让这个可选分支阻塞PCOV或核心冻结。已接好则在领先guide上补一个匹配比较，不做全交叉搜索。

**接线未完成时：** 已就绪的V组继续，未就绪组BLOCKED；没有自动TopK替代。该档五个核心组未齐不能冻结。绝不能把新版选择器的结果回填旧版原COVER行。

### 3.4 172-pool→60只作诊断

包内保留明确命名的池化算子，但不让它进入主 ACC 队列。如果原COVER只替换Top172尾部32，前140个不动，那么再按同分数取Top60必然等于纯Top60。`tail_pool_equivalence.py`对缩小问题的所有排列与合法尾部替换进行穷举；这是条件性的设计核查，仍需读取原函数确认条件是否成立。

`python make_catalog.py --include-pool-diagnostic`只额外生成`diagnostic_pool`配置，官方队列不接受这个phase。仅在CLIP pilot中记录原pool新增点、最终60中幸存的新增点、与Top60差异。`NativeCover.last_native_keep`、`last_requested_keep`可用于该诊断。没有实际作用的池化路径不再浪费全量ACC。

## 4. Stage1主实验与条件扩展

| 组 | 选点证据 | COVER | 投影输入 |
|---|---|---|---|
| V1 | pen评分 | 关 | 原pen |
| V2 | 完成L9的对应评分 | 关 | 原pen |
| V3 | dual，beta=.5 | 关 | 原pen |
| V4 | pen | W2048原K172；W1024新PCOV60/b12 | 原pen |
| V5 | dual，beta=.5 | 与V4完全相同 | 原pen |

两预算各五组，即首批10配置、40个四任务作业。先在固定开发面板闭环，必要扩展后冻结，再对预先固定候选运行独立确认；不是把原83目录全开。两档分别判断，不能把不同入口K之间的差异归因于ViT层。

beta是评分与几何的权重，不是“固定比例token来自某层”。V1必须复核当前原saliency的来源：如果原工程用最后层attention给pen payload选点，不能含糊改成另一层评分再称原基线。记录feature层、score层、head聚合、CLS/key归一化和norm协议；新单层候选与历史原基线必要时分列。

核心完成且有双层收益线索时，`stage1_optional`提供每预算：beta=.25/.75、纯L9+覆盖端点、只变几何、只变评分。低档另提供b6/b18（pen/dual各一）与原COVER_PORT60/b11（pen/dual各一）。默认共16个可选配置，**默认不启动，也不全跑**；用`--ids`只选领先guide的一小组。按开发集质量与完整TTFT选稳定配置，冻结后在独立图像确认。没有收益时保留强单层，不硬保留多层叙述。

如果要进一步声称“多层互补而非一般扰动”，还需匹配实际改点数量的随机补充和必要类别/区域分析。`visual_audit/stage1_visual_audit.py`有该随机补充的CPU参考；它不是已接线的正式随机选择器，本轮不以未实现/未运行控制冒充已有机制结论。

## 5. 模型 forward 与结果导出接线

沿用当前CLIP前向、原processor、投影层、生成与官方scorer。不要另下载演示模型。native COVER实例在任务初始化时构建，源码检查不放进逐token循环。

```python
from elva.cover_binding import load_cover, select_checked

family = config['method']['stage1']['cover_family']
cover = load_cover(binding['native_cover'][family]) if family != 'none' else None
selection = select_checked(
    pen, pen_scores, original_patch_ids, config,
    early=l9, early_scores=l9_scores, early_ids=l9_patch_ids,
    native_cover=cover, context=original_cover_context, backend='torch')
# 将 selection['payload'] 交给原投影层；用 selection['ids'] 维护原身份。
```

上面变量来自现有真实前向，不造heatmap或评分。原COVER所需groups/crop/coords在context中按源码传递。V1不要因为统一诊断面板而在正式TTFT中额外捕获L9；双层实际需要的捕获与评分成本则全部计入。

本包新增 `elva/clip_stage1.py` 的 **`encode_projected` 实际前向桥接**。对标准未融合q_proj/k_proj的CLIP，在原 `encode_images` 中原本“vision_tower(images)→projector(features)”的位置替换为：

```python
from elva.clip_stage1 import encode_projected
projected, selection = encode_projected(
    vision_tower=the_existing_vision_wrapper, projector=the_existing_projector,
    images=processed_images, blocks=actual_clip_encoder_blocks,
    original_patch_ids=actual_original_ids, config=config, cover=cover,
    pen_completed=23, l9_completed=9,
    pen_score_completed=ACTUAL_BOUND_PEN_SCORE_BLOCK,
    l9_score_completed=9, context=original_cover_context)
# 把 projected 接回原多模态序列组装与 decoder；不要再投影一次。
```

这条路径实际执行原ViT一次，hook原q_proj/k_proj输出重建**单个CLS query**的评分，hook对应层hidden，裁出原pen后执行原投影。V1不捕获L9；不要求全层稠密attention输出，也不改原attention backend。若原评分不是其明确定义的`mean_heads_cls_key_softmax_then_patch_l1`，须用原生捕获证据调用前一段`select_checked`，或显式实现并命名另一评分协议，不能含糊把不等价分数当原基线。

`encode_projected`只支持一图/一crop、CLS+576、eval、未融合q/k的CLIP；会验证捕获pen与原wrapper的payload一致，并保留原wrapper输出dtype。融合QKV、带mask、anyres多crop拼接或非CLIP布局必须按当前模型源码适配，不应绕过拒绝条件。真正的Torch hook/真实CLIP检查必须在服务器完成，本地无Torch的skip不是通过。

使用 `BoundRuntime` 的before/after hook接到实际decoder forward：每层执行前记账、完成对应层后按事件选/裁。每次请求新建runtime，`finish()`验证全部32层。不要继续把Stage1入口K写死成90或172。原位置信息、结构token、问题span、attention backend、逐层KV和新decode cache_position必须按原实现维护。

`bound_task_runner.py`调用绑定的 `task_entrypoint='真实模块:真实任务函数'`。该函数使用原数据与生成/scorer循环，返回：

```python
{'records': actual_records, 'official_metrics': actual_official_metrics}
```

每条record用 `native_result_io.sample_record`构造，包含image_key、question_key、真实prediction、stage1_ids、selection['stage1_audit']和完整实际trace。参考答案/问题可以一并保留。`finish_task`验证后才写正式result.json；CPU导出放在正式TTFT计时之外。

源码指纹、样本数、逐例trace、native COVER调用与容量、fallback=false及真实生成缺一，结果均拒绝复用。`fixture=true`结果不能进入真实队列。官方任务缺失时保持缺失，不补零。

## 6. GPU与分片修复

先查实时状态，不复用粘贴的GPU空闲快照：

```bash
python gpu_resources.py --gpus auto --workers 3 --reserve 1
```

进程列表有PID，或显存明显被占用，即使utilization为0也不算空闲。N/A/MIG映射无法确定时停止自动调度。遵守调度系统分配，脚本会尊重继承的CUDA_VISIBLE_DEVICES；不得把别人的dai等进程当ELVA任务清理。

默认一张GPU一个任务进程，ACC最多三卡、另留一张。可在确有四张授权空卡时设置workers=4/reserve=0做ACC，TTFT另外独占测量。只有一张空卡时workers=1/reserve=0串行推进，不等待四卡才开始。

队列使用物理UUID设置CUDA_VISIBLE_DEVICES；**子进程内使用cuda:0**。不要再把物理`--gpu 7`当成屏蔽后的cuda:7。日志同时保存物理编号和UUID。共同租约位于同用户临时目录，可用ELVA_GPU_LOCK_DIR统一覆盖；它只能协调使用该协议的队列，不能排除外部程序，因此每次任务前仍查询实时进程/显存。

旧pilot如仍运行，只能核实UID、完整命令、启动时间与ELVA任务目录后处理本任务的确切进程。不得按Python进程名或一串粘贴PID批量kill。新队列的超时回收仅作用于它自己创建的进程组。

全局pilot限额在分片前应用，避免32张被每卡各取32张或同图多问重复计图像：

```bash
python partition_manifest.py --manifest real_image_manifest.json --mode clip_image --limit 32 --shards 4 --out pilot_shards
```

manifest使用真实canonical image_key。已有CLIP脚本消费各shard文件，先验证各片不交叠、总数与全局限额一致。正式ACC用vlm_question身份，不丢同图不同问。四个PID成功启动并不代表pilot或ACC成功完成，必须检查退出状态、覆盖样本与结果内容。

## 7. 按阶段执行，不能绕过冻结

在本包根目录运行，沿用现有环境。先复制示例为真实binding，填实际source/模型/数据/scorer/函数/样本数；示例空值和false故意不能开测。runner argv可以直接用示例里的bound_task_runner，或原任务入口实现相同合同。

```bash
python verify.py
python make_catalog.py
python gpu_resources.py --gpus auto --workers 3 --reserve 1
python run_queue.py --binding server_binding.json --runner runner_argv.json --out results/core --phase stage1_core --budgets 2048 1024 --gpus auto --workers 3 --reserve 1 --hours 6
python summarize_results.py --binding server_binding.json --results results/core --out results/core_summary.json
```

开正式ACC前，`preflight`至少有：原路径与关闭路径一致、真实CLIP层位、pen payload、真实位置/KV语义，以及两预算主日程的完整VLM trace。只跑CLIP不能把这些VLM检查打勾。

首轮先保证核心完整。有信号才选择预算运行扩展，不能让可选扩展阻塞另一预算核心：

先用真实吞吐pilot估算各任务GPU小时，再决定全量范围和deadline；六小时是队列时限，不是完成40个全量任务的承诺。本队列按完整任务复用，半程生成的断点由原runner维护并绑定run_key；不要丢弃已有合法缓存，也不能把未完成任务标DONE。

```bash
python run_queue.py --binding server_binding.json --runner runner_argv.json --out results/core --phase stage1_optional --budgets 1024 --ids ACTUAL_PAIRED_B6_ID ACTUAL_PAIRED_B18_ID --gpus auto --workers 3 --reserve 1 --hours 6
```

`summarize_results.py`只有在提供同版本dense参考、metric键和完整四任务时才计算归一化R4；否则R4为null，保留官方分项。不造dense分母。

冻结命令的winner取自实际完整结果；可选beta只有自己四任务也齐全后才允许入选。reason说明分项、独立确认、成本与未解决限制，不自动把最高搜索点当作机制成立。

```bash
python freeze_stage1.py --binding server_binding.json --results results/core --budget 2048 --winner ACTUAL_COMPLETED_ID --reason '基于真实分项、配对确认与TTFT的选择理由' --out stage1_decisions.json
python make_catalog.py --decisions stage1_decisions.json
python run_queue.py --binding server_binding.json --runner runner_argv.json --out results/core --phase stage2 --budgets 2048 --decisions stage1_decisions.json --gpus auto --workers 3 --reserve 1 --hours 6
```

低档同样独立冻结，可合并到同一decision文件。Stage2先Q_ATTN/RDRIFT同日程比较，补原QED、attention+coverage；源码与有效参数相同则去重，不能仅换名字。相同方法可复用Stage1已验证锚点；结果、来源或winner被改动后冻结失效。新的官方benchmark清单继续按服务器已冻结版本，不在本包发明未核实任务。

## 8. 图示、TTFT与回传

图示用 `visual_audit/`，优先在固定小面板同步检查分数、原IDs、实际crop坐标、共有/独有与集合重叠。真实结果后再画beta–官方质量/TTFT曲线。禁止用合成图解释本模型效果，也不靠热点图挑winner。

TTFT统一从请求准备到真实首token，涵盖processor/ViT、双层捕获、COVER、投影、prefill、裁剪与同步；原单层baseline不添加无用L9成本。至少三个交错重复轮次，独占卡记录wall p50/p90、峰值显存、缓存策略及原始逐次记录。诊断导出/绘图与正式计时分开，不能只给不同方法单独排除其真实算法成本。

回传：commit/diff、真实binding、源函数摘要、两预算配置、官方分项和原始预测、全样本trace/IDs/COVER审计、pilot分片与退出状态、独占TTFT原始记录、Stage1冻结及一页裁决。明确区分：接口已生效、真实质量收益、跨预算稳定性、实际效率收益。

## 9. 本地验证边界

本包本地运行CPU数学、COVER容量/无fallback/池化身份、源码变更、GPU解析/租约/可见设备、逐例结果拒绝、完整核心冻结、结果变更失效、fixture任务执行与复用，以及R2原数值回归。历史COVER公式保持外部源码绑定；新增PCOV明确作为不同选择器单列。本地没有服务器模型或GPU精度/TTFT结果。PyTorch状态以validation中的实际skip记录为准，不能把服务器报告的48项与本地状态混写。

## 10. K60 专项裁决与收敛顺序

1. **预检**：32个不同图像的固定CLIP面板，先全局限额再分片；将OCR/计数/空间/普通对象案例按现有标注预先选入，不按结果挑例。记录每组实际IDs、显著性分数、metric、真实processed crop与变换坐标。
2. **核心开发评估**：两档V1–V5，Stage2固定，不变其它recipe。可先按现有有效任务分组构造约128图像规模开发面板，保留MME成对题、POPE平衡组、MMBench题组等官方评分结构；不是机械截前128问。固定图像/问题manifest及对应dense参考并记录样本数。若官方scorer不支持该抽样，使用合法完整子集而不是自造ACC。
3. **低档判据**：V5−V4检查多层在同覆盖器下的收益，V3−V1检查无覆盖收益，V4−V1与V5−V3检查覆盖收益。不要比较跨W的rawACC来证明层作用。按官方指标、同图配对不确定性和任务退化裁决；同图多问按图像聚类，非加性官方指标用原scorer在重采样组上重算，不能把POPE F1/MME等偷换成逐问均值。
4. **只扩一个维度**：多层有效才先在同覆盖b12下比较beta .25/.5/.75。固定领先guide/beta后，用`python make_catalog.py --coverage-refine-from configs/ACTUAL_COMPLETED_B12_CONFIG.json`生成同一guide/beta的b6/b18；然后只补这两个配置。不得同时跑层号×beta×b×日程全交叉。源码已备好时再补同guide的PORT60/b11；如要声称多层机制，还需真实接线的匹配随机控制，不能用离线随机函数代替ACC控制。
5. **独立确认**：确认集按图像与开发集不重叠；固定已选guide/beta/b和强单层基线后再跑。总体收益须超过配对不确定性，同时检查OCR/计数/空间退化与完整TTFT。如果不稳，选更简单的强单层或b0，不保留多层/覆盖成功的叙述。例图、coverage分数或单一ACC峰值不能取代独立确认。
6. **冻结**：每档独立冻结已有完整四任务结果，记录选择理由、确认文件、失败项及成本；然后才启该档Stage2。服务器已冻结的8–10 benchmark清单留作最终确认，不在开发轮全部启动。

离线新增 `coverage_report.py` 绘制覆盖名额–显著性损失、特征覆盖增益和**最终60中实际新增数量**三联图，并导出CSV。输入为真实逐图诊断JSONL，同一图像面板、guide/beta与源码；fixture需显式允许并带水印。

在正式TTFT之外调用（先把实际tensor导出为NumPy，不可放计时路径）：

```python
from elva.low_budget_cover import diagnostics
row = diagnostics(actual_metric, actual_scores, actual_keep_local, 60, slots)
row.update(schema='elva_r3_cover_audit', fixture=False,
           image_key=canonical_image_key, config_hash=actual_config_hash,
           source_hash=actual_source_hash, guide=frozen_guide,
           beta=frozen_beta, work=1024)
# 保存真实JSONL；b0用同一分数的Top60，b6/b12/b18用对应实际输出。
```

```bash
python coverage_report.py --records actual_cover_audit.jsonl --out figures/cover_tradeoff
python visual_audit/stage1_visual_audit.py images ACTUAL_CASE_JSON --baseline V1 --candidate V5 --out figures/stage1
```

具体CLI参数以`--help`为准；原始分数/向量/IDs/crop仍是溯源基础。未标注文字或目标区域时不要把“空间分散”写成“语义证据被保留”。`diagnostics`记录保护数量、真实增删local索引、Top60及保留分数质量、uniform/weighted特征覆盖。空间热点、共有/独有图与真实区域覆盖由visual_audit按原crop坐标绘制；这些图和官方质量/TTFT互相补充。

## 11. 与上一版的联系、区别及相关方法

继承：R2的原IDs对齐、只取raw pen payload、beta几何/评分定义、精确W记账、原decoder语义和已有回归模块。没有恢复旧E020训练。

实质增加：K60独立保护覆盖选择器、真实CLIP→选择→原投影前向桥接、原COVER无fallback调用与容量/源码审计、只用两档的配置和并发队列、严格真实生成/trace验收、可选组精确ID筛选、完整核心后冻结Stage1、低档覆盖/显著性诊断与绘图。历史私有模型forward与官方scorer仍需Cursor在服务器完成准确接线；本包不宣称已经修改到无法访问的服务器文件。

这不是新的复杂学习算法。覆盖与显著性组合已有相关研究，PCOV只作透明的研究控制；后续论文贡献要由多层互补和生命周期实验的独立证据支撑。可核对的原始资料：

- Hugging Face原CLIP attention源码（旧版结构参考，不代表服务器安装版本）：[原始源码](https://huggingface.co/transformers/v4.11.1/_modules/transformers/models/clip/modeling_clip.html)
- Bifocal，2026-10-13预印本，讨论多任务证据与特征覆盖：[原文](https://arxiv.org/abs/2610.18399)
- QCPruner，2026-09-17预印本，覆盖优化相关工作：[原文](https://arxiv.org/abs/2609.19990)

这些文献不能替代本工程的ACC/TTFT测试，也不应把其报告分数移入本实验。
