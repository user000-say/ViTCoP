> 历史E020.1执行书，已由 `CURSOR_E020_1_R2_CN.md` 替代。不要执行其中28配置和g1r4默认策略。

# E020.1 四卡执行书：补齐效用分支，不重新铺开72配置

目标：把E020同预算继续交互集合交换真正跑到真实权重、ACC和TTFT；并行恢复已暂停E019。当前没有任何学习模块已测结论。不要将“85/104独立作业”写成主效用模型完成。

本环境没有服务器项目源码或`E019_E020_STATUS.md`实际文件。此包是完整可移植模块及执行修正；真正teacher-forced backend仍需在现有项目里接上。先阅读本文；旧`CURSOR_HANDOFF_CN.md`是历史方案，不作为本轮默认入口。

## 1. 不等待新的审批，先完成这些并行工作

物理GPU0/2/4/6。让正在跑的5个COVER priority臂收尾，保存state；不杀别的任务。平滑停止旧priority的新派发后，分工为：

| 卡 | 第一阶段 | 权重准备后 |
|---|---|---|
|0/2|E019 B/C恢复＋必要随机seed复核|与4一起跑核心学习矩阵 |
|4|真实同前缀标签采集；结束后CPU拟合|核心学习矩阵与确认 |
|6|NEXT/SHIFT/BASE补偿锚点TTFT|学习候选TTFT与anchor漂移 |

禁止在测速卡叠加ACC。新版runner有drain_event：停止派发新sidecar任务，但让已运行任务正常结束；释放到学习队列后，余量再恢复原E019矩阵。

## 2. 先交出真正的标签和权重

需要的源级接入只有已有teacher-forced调用与prompt snapshot/首event覆盖。使用`replay_adapter.py`可从同一输入完整重跑，无需新写任意层resume。

后端`run_teacher_forced`须：

- 复用现有COVER/BASE和原始后续日程，只在首event按给定原始patch IDs覆盖选择。
- snapshot只读prompt问题/视觉位置；teacher-forced答案不进入线上特征。
- 返回真实prompt前缀hash、同一答案token身份、平均答案NLL和缓存策略。
- 所有分支eval且确定性。参考保留集按原始序列顺序；前缀相同不是只给相同的字符串标签。

backend确实接通后，先用少量样本自动验证并继续bulk采集，不逐项等用户确认。参考命令：

注意物理剪枝会缩短decoder logits，而原始labels可能仍是展开后的长序列。若直接使用内置loss出现长度不匹配，令模型只返回logits，使用`teacher_forced_nll_at_rows`按原始text token身份对齐预测行；如果确认答案是无padding的连续末尾suffix，可直接用`teacher_forced_answer_suffix_nll(logits, answer_ids)`。不能把原始labels随意裁成同长度：预测第一个答案token的行是它的因果前驱。新包提供这两个完整工具；这是一项接入注意，不代表已定位服务器未采集的根因。

```bash
python pairdata.py --adapter replay_adapter:make --config replay_config.json --out /ABS/PROJECT/e020_pairs.npz --states 768 --actions 6 --group-sizes 1 --shortlist 16 --noop-every 8
```

replay config的`max_examples=384`，`arms=["COVER","BASE"]`，`schedules=["FIRST_SHIFT","FIRST_NEXT_WMATCH"]`；同图两臂交错、日程按图交错。source对象与runtime完全相同，包含checkpoint/prompt/code/输入/缓存/feature协议hash。使用独立已标注训练/calibration图像，按canonical image身份排除最终评测图像。原投影文件、问题分组规则和之前response捕获位置都固定。

`force_recompute=True`只改变离线baseline复用：每8个state真跑一次相同保留IDs的no-op。不能用缓存命中返回0来验证正确性。接口检查会比较答案token身份与原始前缀。

约3k–4.6k动作只是容量目标，按真实吞吐控制90分钟采集预算；partial shard可继续拟合。先默认g1，与主部署g1一致；g4后续另采、另报，不能混成g1 oracle。

拟合：

```bash
python fit_models.py /ABS/PROJECT/e020_pairs.npz --out /ABS/PROJECT/models --modes current,full,no_context,shuffled --forbidden-image-ids /ABS/PROJECT/eval_image_ids.json
python diagnostics.py /ABS/PROJECT/e020_pairs.npz --models /ABS/PROJECT/models --group-size 1 --shortlist 16 --out /ABS/PROJECT/heldout_g1.json
```

4个小头同容量，current优先生成；按图像隔离65/17/18。不得用随机初始化、touch空npz或REF输出来补“模型已测”。权重一旦开始筛查就冻结，不边改模型边往同一个state目录追加结果。

## 3. 真实推理与选点生效证据

主模型算法调用仍是`dispatch.refine`，仅首剪枝event交换真实patch；每层K与层位不变。current是主候选，full是历史消融，默认g1、最多4轮。最终剪枝仍走原项目。

新版`PairScorer.loaded_sha256`在实际读取模型文件时计算。每个学习任务的`result.json`增加：

```python
result['scorer_sha256'] = hook.model.loaded_sha256
```

这里必须来自真实已加载的对象，不能只复制job里的期望hash。缺项/不匹配会使该学习任务失败，其他独立任务继续。原有REF与无训练控制不要求此字段。

ACC时将baseline与最终选点原始IDs、eligible IDs、hook是否调用和scorer hash写为trace；TTFT关闭此诊断I/O。示意：

```python
from selector_audit import make_trace, append_trace
row = make_trace(request_key=stable_request_key, config_id=job['config']['config_id'],
    source_hash=source_hash, baseline_ids=original_patch_ids[baseline].cpu().tolist(),
    selected_ids=original_patch_ids[keep].cpu().tolist(), eligible_ids=original_patch_ids.cpu().tolist(),
    hook_called=True, scorer_sha256=hook.model.loaded_sha256)
append_trace(job['selector_trace_file'], row)
```

`stable_request_key`由原始样本/问题/prompt身份构造，不能取预测答案。source_hash为`build_matrix.digest(runtime['source'])`。REF/CSCORE另在32个真实样本导出同类选点记录，核对相同集合比例与相同预测比例；相同R4不是同集合证明。

matched-random任务依赖current模型在同面板的trace。runner先完成依赖，再运行对照；项目wrapper读取`job['matched_trace_input']`，调用`selector_audit.load_targets`验证配置/source，放到`hook.matched_targets`。传入`request_key`给dispatch。它严格匹配该样本最终换了多少patch；不宣称匹配每轮提案路径。

matched-random用于ACC机制对照，确认阶段也保留NEXT上这一项。它读取另一方法的trace，不当作独立部署方案画TTFT曲线；timing矩阵自动排除它，避免在不同timing面板上读不存在的ACC trace。

rank-jitter与随机复核也传稳定request_key；spec.seed与sample_seed联合决定RNG。旧的4种随机设置不是4个seed。

## 4. 真实NEXT与强基线

从已跑job读取NEXT真实event、层号语义及逐层token trace，填入runtime.schedules.FIRST_NEXT_WMATCH。不要凭96.67反推层位，也不要让floor109覆盖真实导出的110或其他配置。该字段缺失时close_core会明确要求导出。

主参照：COVER NEXT96.67；连续性参照：COVER SHIFT96.50；BASE SHIFT95.50与BASE NEXT95.83用于同入口因子；BASE KPLUS8 96.34是额外计算补偿锚点。不同面板只做同面板对照，不用全量R4冒充screen分数。

完整四任务结果同条件原始记录可复用并真实重评分；将reused_from、样本ID与实际协议写清。不要把旧聚合表当作逐样本记录。

## 5. 核心28配置＋独立10seed复核

| 矩阵 | 规模 | 目的 |
|---|---:|---|
|2入口×SHIFT/NEXT×REF/current/full|12配置|核心同成本因子 |
|2入口×SHIFT/NEXT×no-context/shuffled|8配置|容量/上下文/监督控制 |
|2入口×SHIFT/NEXT×边界jitter|4配置|检验是否简单扰动已足够 |
|2入口×SHIFT/NEXT×matched-random|4配置|匹配真实最终换点数 |
|核心合计|28配置×4=112 screen jobs|学习权重生成后执行 |
|BASE SHIFT、COVER NEXT的g4r2各5seed|10配置×4=40 jobs|检验边界随机改善是否稳定 |

先screen每任务128–256有效unit，MME保双问题和类别，POPE保图像簇/正负。最多10配置确认（5固定锚点、NEXT current/full/匹配随机，以及开发集强候选），共≤40任务，旧全量记录可复用。最后保留real JSON/state，未完行不计算R4。

```bash
python random_replicates.py runtime.json --out random_seeds.json
python core_matrix.py runtime.json --out core_screen.json
```

E01936个B/C臂仍在待办。恢复时优先current/mean/random与代表性temporal，然后其他变体；不是宣布它们失败或删除整块。状态里B=12与原包B=20可能是主动裁剪，导出实际manifest和原因即可，不凭数量认定漏跑。

## 6. 建议默认执行入口

将实际collector命令argv填入runtime.collection.command，真实shard位置填output_shard；已存在的真实shard也可填pair_shards。模型路径/trace_directory都用绝对路径。source与面板ID沿用真实导出。

```bash
python core_progress.py runtime.json --shards /ABS/PROJECT/e020_pairs.npz --out /ABS/PROJECT/core_before.json
python close_core.py runtime.json --bridge-config bridge.json --out /ABS/PROJECT/e0201_run --gpus 0,2,4 --timing-gpu 6 --hours 6
```

缺数据且缺collector命令时立即报告`NOT_TESTED_MISSING_VALID_MODELS`，不再自动铺无训练大扫。这个是依赖事实，不是算法失败。补上后直接继续，不需用户另行审批。

若要同时恢复E019，可传已有真实sidecar矩阵与其bridge：

```bash
python close_core.py runtime.json --bridge-config bridge.json --out /ABS/PROJECT/e0201_run --gpus 0,2,4 --timing-gpu 6 --hours 6 --sidecar-matrix e019_remaining.json --sidecar-bridge original_e019_bridge:make --sidecar-bridge-config e019_bridge.json
```

允许读旧e019_matrix_v1头，job本身身份/指标/配置不变；实际bridge仍需指向原有成功的E019 runner。新目录的已完成行只有通过真实cache索引才能复用；否则复用旧runner的原输出目录和state由Cursor调度，避免把144个已知待办重新造一份不兼容队列。

6小时容量分配：前45分钟完成接线/现有队列收尾并启动并行任务；约45–135分钟采集与拟合，0/2跑sidecar、6测锚点；模型可用后约135–225分钟核心screen；剩余时间确认与TTFT。12小时可用`--hours 12 --collection-seconds 7200 --screen-seconds 10800`。按真实吞吐排，不承诺所有任务全量完成。

## 7. 本轮必须带回的CPU审计

1. 使用`audit_mme.py`输出BASE/COVER FIRST与EARLY的类别分数、wrong→right/right→wrong；解释53分差距来自哪里。数据格式沿用`paired_stats.py`的id/image_id/category/correct，二问完整。
2. 用`strategy_overlap.py`比较COVER NEXT/SHIFT、BASE SHIFT/随机或RDRIFT的真实预测交集。报告逐题互补oracle；不能把gold oracle当在线方法，也不保证同成本。
3. NEXT96.67四任务完整值、逐层数量、TTFT；CSCORE/REF选点一致性；每方法真实交换数量。
4. 训练source/hash、实际图像/标签数、image-disjoint split、no-op噪声与held-out收益。抽64个独立图像验证NLL改善和真实生成正确率的关系；NLL不是准确率。
5. 保存真正的后端接入patch和一份真实成对trace，供下一轮审查。不要只回传聚合R4和“模型文件已生成”。

## 8. 正式投稿验证继续并行准备

恢复近期可运行基线；CoRePrune/DIPrune/Bifocal记录官方代码可用性，拿不到不伪称复现。已有ViTCoP/PyramidDrop/DivPrune可复用实现先安排同模型、实际计算预算与统一测速。准备另一架构和至少两类视觉密集任务；当前R4分母仅用于当前主模型，换模型需自己的dense基准。

当前含decoder随机投影的小头不能默认跨架构原权重迁移。可以先以同样小额calibration验证方法迁移，再单独研究零样本权重迁移；不要把两者混写。

本轮学习模块若没有可靠收益，要完整回报。观察真实oracle空间、预测误差与任务错误迁移后决定改哪里，不再先增加一批新的attention公式。
