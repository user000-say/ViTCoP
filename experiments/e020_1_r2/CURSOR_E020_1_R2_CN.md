# Cursor执行书：E020.1 R2，完成一个真实算法闭环

本轮只修订E020.1，不新起算法编号。先保留已运行任务及原始记录，不杀其他实验进程。
本包包含完整的通用交换、采集、拟合、诊断、调度代码；**不包含服务器私有模型后端与官方评分器接线**，不能声称下载后已经完成模型级集成。

## 1. 先冻结定义与原始锚点

1. 导出实际NEXT事件、层号约定、逐层token数、四任务明细，不从96.67或floor公式反推。
2. 原模型、输入、prompt、评分协议保持与已有结果一致。记录POPE到底用F1还是accuracy，MME保存同图双问题及类别。
3. 官方ViTCoP S0继续走原始成功路径。将其实际配置和逐层trace填入`runtime.external_anchors`；S0的W=2034目前只来自报告，先核验，不虚构events。
4. 应用R2改动后重新导出`runtime.source`；`feature_protocol_hash`必须覆盖v2特征、投影、候选协议和捕获位置。Replay config复制同一个source对象。
5. 旧v1权重不能改名复用；当前头/消融的特征语义和部署契约已变。旧结果、旧pair shard保留归档；若要迁移标签，必须重新按真实前缀生成v2特征、补齐候选目录与校准信息，不直接改schema字符串。

`runtime.external_anchors`中的每项使用如下字段，值全部来自原runner：

```json
{
  "stage1": "OFFICIAL", "schedule_name": "S0", "selector_name": "reference",
  "selector": {"kind": "reference"},
  "application": "original_project_path", "native_method": "ViTCoP_S0",
  "budget": {"per_block_visual": [], "visual_token_layers": null},
  "original_runner_config": {}
}
```

空列表/空工作量是待填模板，验证器会拒绝。项目wrapper必须按`application`分流到原路径，不能让S0误走E020首事件hook。

## 2. 完成最小真实后端

沿用`replay_adapter.py`接口，由已有项目实现：

```python
backend.examples()  # 独立已标注calibration样本，按canonical image身份隔离
backend.run_teacher_forced(example, stage1=..., schedule_name=...,
    first_keep_patch_ids=..., forced_final_ids=..., capture_prompt_state=True)
```

返回真实`answer_nll/prefix_hash/cache_policy/answer_tokens_hash/answer_token_count/snapshot`。
snapshot字段详见`pairdata.validate_snapshot`，并增加：

```python
snapshot['feature_availability'] = {'history': captured_actual_history}
snapshot['feature_origin'] = 'prompt_only_pre_event'
```

current主线不需要为了历史另算attention。full消融只有确实捕获了对齐后的历史响应时才拟合；不能把填0当历史。若未来想比较full，在同一批图像重新采集包含真实history的特征和同目录标签，保持source/version清楚。

所有特征只来自prompt和已计算视觉状态。teacher-forced答案只用于NLL。第一个答案token由其因果前驱logit预测，物理剪枝后用`teacher_forced_nll_at_rows`保留原始文本身份对齐；仅当答案确为无padding的末尾连续suffix，才用suffix工具。

先用8–16个真实样本排除接线错误，检查真正重算no-op、相同答案token身份、同层位与同K、patch原始顺序及未变的后续选择器。此后自动继续批量，不逐项停下来等用户确认。

另抽少量样本比较prompt-only前向与teacher-forced捕获的特征，并改变答案内容验证特征不变；只填写`feature_origin`字符串不构成无泄漏证明。保留实际问题行、视觉列和mask的捕获patch供审查。

## 3. 采集完整的同一部署目录

默认：shortlist=16、candidate_limit=16、g1、r1。rank网格中的确定性16动作是部署目录，离线尽量完整标注，另有不交换动作。不是每状态只随机标6个后声称覆盖在线256动作。

先从128独立图像、两入口合计256状态起步；SHIFT/NEXT按图交错，两个入口共用图像分组。根据真实吞吐再补充，不承诺半天全部完成。

```bash
python pairdata.py --adapter replay_adapter:make --config replay_config.json \
  --out /ABS/PROJECT/e0201r2_pairs.npz --states 256 --actions 16 \
  --candidate-limit 16 --shortlist 16 --group-sizes 1 --noop-every 8
```

默认每16动作尝试可选identity-fixed诊断。后端不支持全层视觉KV处理时明确记录未支持；绝不能只修改最终IDs就宣称隔离纯交互。若支持，应在预算中计入额外两分支开销。

采集进程收到自己执行器的SIGTERM时保存已完成动作。未完成目录可以参与拟合，但不参与最大值决策阈值校准。补采写新shard，按完整状态+动作身份去重；前缀相同但未来日程不同，不得合并。

```bash
python fit_models.py /ABS/PROJECT/e0201r2_pairs.npz \
  --out /ABS/PROJECT/e0201r2_models \
  --modes current,no_context,shuffled,full \
  --forbidden-image-ids /ABS/PROJECT/eval_image_ids.json

python diagnostics.py /ABS/PROJECT/e0201r2_pairs.npz \
  --models /ABS/PROJECT/e0201r2_models --group-size 1 --candidate-limit 16 \
  --out /ABS/PROJECT/e0201r2_diagnostics.json
```

按图像65/17/18拆分。训练集拟合；验证集选择epoch和阈值；诊断留出不参与训练/阈值。最终ACC图像不得与这三部分重叠。诊断留出一旦反复用于方法决策，就承认其开发用途，不能再称从未查看的最终测试。

阈值校准默认至少12个验证图像、完整目录及真正重算no-op。这是工程起步门槛，不是会议规定或统计充分性的证明。若校准策略选择全部abstain，结果必须明确写出；这不是学习模块已有效。若验证目录不足，就补采；不靠改阈值/伪造完整标志绕过。

## 4. 推理接线与生效证据

只在首个剪枝事件调用`dispatch.refine`，使用实际加载的`hook.model`；每层数量、后续事件、position IDs、RoPE、KV语义沿用原runner。R2检查模型部署契约与阈值，拒绝把g1r1权重拿去直接跑4轮。

ACC继续记录`selector_audit.make_trace`中的真实原始patch IDs、hook调用、scorer SHA；学习结果`result.json`须写`hook.model.loaded_sha256`。matched-random在同样本、同候选目录匹配0/1个最终交换，读取已完成current任务的trace；它是机制对照，不能作为独立部署方案画TTFT。

full头之外，current/no-context/shuffled不要求历史捕获。no-context只额外屏蔽共同集合均值；shuffled只打乱训练标签，使用相同的current输入和验证流程。报告既有冻结阈值策略收益，也有零阈值原始头收益，区分信息与校准作用。

## 5. 一批执行，核心不再被侧线阻塞

| 阶段 | GPU0/2 | GPU4 | GPU6 |
|---|---|---|---|
| 接线、采集与拟合 | 剩余COVER臂、必要E019代表项/随机复核 | 主采集 | S0/EARLY/SHIFT/NEXT实测 |
| current可用后 | 与4共同跑核心ACC | 核心ACC | 学习候选总开销 |
| 主比较齐全后 | 固定确认、余量恢复E019 | 确认/必要诊断 | 同卡anchor漂移与确认 |

full缺失不阻止current。E019 B/C仍在待办，优先代表性current/mean/random/temporal；不凭SHIFT的一个控制结果宣布整块失败。

核心：2入口×2日程×REF/current/匹配随机=12配置；COVER NEXT再加full/no-context/shuffled/E019CUR最多4配置。共最多64个screen jobs。固定确认9配置，另加导出的S0及少量外部锚点。已有同条件原始输出允许身份核验后复用；聚合分数不能伪装成逐题结果。

填好原ACC/TTFT命令argv、collector命令、绝对路径、实际panels后运行：

```bash
python close_core.py runtime.json --bridge-config bridge.json \
  --out /ABS/PROJECT/e0201r2_run --gpus 0,2,4 --timing-gpu 6 --hours 6
```

若同时恢复E019，沿用原成功runner和manifest：

```bash
python close_core.py runtime.json --bridge-config bridge.json \
  --out /ABS/PROJECT/e0201r2_run --gpus 0,2,4 --timing-gpu 6 --hours 6 \
  --sidecar-matrix e019_remaining.json \
  --sidecar-bridge original_e019_bridge:make --sidecar-bridge-config e019_bridge.json
```

原始模型接线尚未完成时，先完成它，不能用又一轮免训练扫描替代。运行过程中不自动扩展72旧配置。`workflow.py`及旧28配置说明仅用于历史追溯。

## 6. 并行CPU机制与统计

```bash
python factorial_interaction.py base_early.json cover_early.json \
  base_shift.json cover_shift.json --pope-metric f1 --out interaction.json
```

该脚本输入是每文件包含四任务原始预测的JSON，不是R4表；按图像对四臂同步重采样。POPE参数按真实口径改accuracy或f1。继续用`audit_mme.py`拆类别、`paired_stats.py`做主比较；共同样本不能缺题，MME不能拆散双问题。

同时比较REF/CSCORE真实选点与预测一致率，回传NEXT任务明细、随机同一策略多seed结果。交互区间不跨0仍只说明性能交互存在，不能代替机制干预。

## 7. TTFT与外部基线

同一物理卡、同输入panel、同attention后端、同batch/cache、同预热，配对交错anchor。报告CUDA均值/中位/p90、端到端wall TTFT、选择器总开销。不得只计1664参数MLP；特征捕获、排序、同步和搬运全部计入。

`timing_protocol`是给wrapper的执行契约，不会凭字段名称自动产生真实计时；必须保留wrapper实现和原始计时记录。主包只能验证结果身份，不能隔空保证wrapper实现ABBA。

S0、强固定日程、至少最直接可运行的近期方法并行准备。先区分前置选择与交互层机制的公平对照，再画端到端成本曲线。无法取得官方代码要记录，自己的简化重实现不能冒充官方复现。

## 8. 若没有增益，按原因推进一次修复而非无限开分支

- 16候选集收益很小：少量图像离线扩大到64动作，先诊断提案覆盖；输出另存，`diagnostics.py --candidate-limit 64`另报。64目录不能直接训练/冒充16目录部署权重。
- 候选有空间、预测无收益：检查噪声、特征、训练量，只做一个界定清楚的修复回合；不直接新增多套attention公式。
- 单步成立后才考虑多轮：采集实际接受后的状态，评估多轮真实rollout；不把独立单步标签加起来。
- NLL收益与生成脱节：抽样实际生成，核对答案集合与评分；必要时改标签目标并重新隔离验证，不能用NLL替ACC。
- 成本抵消收益：测特征和搜索开销，简化部署。三预算、另一架构和视觉密集任务在主方法有可靠收益后扩展。

## 9. 本轮必须回传

1. 后端真实patch、8–16样本接线检查、实际完整目录数和图像数、no-op误差。
2. 训练/验证/诊断图像数、source/权重hash、校准状态、接受率、真实负收益尾部。
3. current相对NEXT REF、匹配随机、E019CUR的同panel四任务结果和逐样本记录。
4. MME类别归因、入口×日程交互、随机复核、选点生效证据。
5. S0/固定/学习候选真实TTFT与总开销；外部基线可用性与首轮结果。
6. 明确区分未测、部分ACC、主比较完成、正/负结果与论文可用证据。不能把自动状态中的completed当成方法有增益。
