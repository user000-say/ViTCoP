# Cursor执行书：48个真实配置，四卡连续推进

先读RESEARCH_DECISION_CN.md。沿用上一轮已经跑通ACC16/16与TTFT的仓库、模型和runner，不重搭模型、不重做一轮漫长环境审计。

## 直接执行顺序

1. 从真实ONE/EARLY/LATE配置导出runtime.json和已有runner的bridge_config.json；EARLY就是TWO，复用同一身份。
2. 用32个真实样本验证原路径与新hook：K/顺序/原始patch身份、问题组身份一致；服务器运行checks.py，PyTorch项不能skip。此时GPU3可同时复现既有计时面板。
3. 构建并运行48配置筛查，GPU0/1/2独立领取作业。当前响应分支不必等待历史分支调通；发生错误的行记录后继续其他行。
4. 自动汇总并冻结最多4个候选，进入确认；GPU3并行测候选与锚点。还有时间就跑预先声明的困难面板及旧控制器的反事实诊断。

只保留三项基础约束：样本/来源身份一致；真正的token日程可核算；评分来自实际输出。其余选择由Cursor按本书和真实吞吐自主推进，不逐条请示。

## 这次新增的模块怎样接入

主入口仍用BASE/COVER。初次剪枝和原QED/Q-ATTN规则先不改。selector_hook.EvidenceHook只接到最后一次压缩：

- 在倒数第二次剪枝处，capture已有问题组→视觉响应矩阵及原始patch IDs。保存的是fp32小矩阵，不是LLM新token。
- 在最后一次剪枝处，先运行原selector得到baseline_indices；取得当前问题组响应和同一候选集合的单token分数。若QED没有独立标量分数，s统一用当前原始Q-ATTN分数，baseline_indices仍来自原QED，并记录该定义；score_topk控制单独检验换成该分数TopK是否已经解释增益。
- refine返回同样K个当前真实patch索引。沿现有工程的gather、KV、position处理继续，不自行重写cache。
- reference配置完全走原路径，不调用新模块；B=0/λ=0的exchange用作必要identity检查。

问题组从原始问题span均匀分为最多4组，固定映射跨层复用；短问题组数不能人为补出语义。排除system/模板/答案，明确MMBench选项是否属于原问题span，所有方法一致。
已有QK/attention行能复用则复用；若原QED只保留最终聚合分数，需要保留少量问题行或额外计算这些QK，不能全局退回eager attention取完整矩阵。额外计算、scatter/gather、排序和同步全部计入端到端时间。
默认debug_validate=True检查身份。通过真实嵌套集合检查后，测速显式关闭debug_validate，保持FP32并禁用该段autocast；注意torch.where/nonzero等也可能产生内部同步，不宣称零开销。
当前hook按batch=1验收。ACC诊断可在少量捕获输入上用NumPy离线重放交换，记录接受数与覆盖变化；不要把CPU诊断传输放进在线计时。scalar_ema/current_mean是本原型的标量控制，不冒充与旧Q-ATTN完全等价。
random_swap用稳定sample_id派生seed，不允许随作业启动顺序改变。它是固定最多4次扰动控制，报告与真实优化交换数的差别。
当前模块不读取答案、未来层或dense结果，不插回早已删除的token；不混入区域均值+代表点或token merging。

## runtime.json：导出真实参数，禁止填猜测值

runtime.example.json是结构模板，含null字段不能直接运行。需要：

- model_depth、initial_visual_tokens、ONE/EARLY/LATE每次事件的after/keep。after表示已完成block数，不能直接把代码layer index照抄。
- expected_visual_work至少EARLY=2284、LATE=2564，只有真实日志确认这两个值适用同一计数范围时填写。若不一致，修复边界映射或记录原口径，而不是硬调数量凑值。
- source：checkpoint_hash、code_commit、prompt_hash、input_pipeline_hash、cache_policy、query_rule。
- metric_keys：GQA、POPE、MME、MMBench各自实际key，POPE与分母85.8使用同口径。
- panels.screen与panels.confirm四任务的实际ID文件路径及hash。timing_panel使用现有固定1024条面板。可预先声明stress_panels。

筛查优先用已有开发源。只有现有评估源时，冻结开发子集和隔离确认子集并如实记录；不要把看过并调过的同集全量包装成未见测试。有正式完整评测时补全量结果即可，不为等待新数据停工。

筛查建议每任务约128个有效单位，先以实测速率调整；POPE分布平衡，MME保持题对和类别覆盖，MMBench沿官方样本组。不是所有任务都能随机抽128行后直接按accuracy打分。
筛查R4仅是开发排序代理，不等于完整benchmark R4。确认沿冻结官方协议计分；同一确认集上报告所有入选候选，不只回传赢家。

## 薄bridge复用已有runner

在真实项目实现e019_bridge.py的build(config)工厂。遵守project_bridge_contract.py：

- command(job,outdir)返回argv字符串列表，启动已有ACC或TTFT命令。queue已隔离CUDA_VISIBLE_DEVICES，子进程使用本地cuda:0。
- read_result返回实际metric_value、metric_key、job_hash、panel_ids_hash、records_path。records_path是现存原始预测或计时文件的绝对路径。
- 可选cached_result：只有代码/配置/样本/评分匹配时复用已有结果。EARLY/TWO不重复算独立测量。旧全量输出可以过滤后重算筛查子集；只有旧汇总均值不能冒充子集结果。

source hash指向真实模型路径和旧pruner，bridge_config可含实际runner路径。不要为了适配写一个“返回表中分数”的假backend。当前包没有原仓库源码，因此这个薄bridge必须由Cursor据实际已跑入口完成。

## 一条主命令与分阶段命令

适配完成后：

```bash
OPENBLAS_NUM_THREADS=1 python checks.py
python workflow.py runtime.json --bridge e019_bridge:build --bridge-config bridge_config.json --out runs/e019 --hours 6 --screen-seconds 7200 --gpus 0,1,2 --timing-gpu 3
```

workflow先筛查，通常2小时上限；提前完成立即晋级，不等待时间耗尽。确认和GPU3计时并行，随后继续预声明stress面板。所有阶段共享6小时墙钟；预留最后15分钟，不把每阶段各6小时相加。
每个phase的state.json持续保存完成、失败、待续跑状态；失败最多默认重试1次。相同命令可续跑已完成项；修改runtime/矩阵时换新输出目录，不覆盖旧结果。

也可分开执行：

```bash
python build_matrix.py runtime.json --out screen_matrix.json
python run_matrix.py screen_matrix.json --bridge e019_bridge:build --bridge-config bridge_config.json --out runs/screen --gpus 0,1,2 --seconds 7200 --reserve 0
python summarize.py screen_matrix.json runs/screen/state.json --out screen_summary.json
python promote.py screen_matrix.json runs/screen/state.json runtime.json --out confirm_matrix.json --top 4
```

分阶段执行由Cursor自行扣除已用时间。不要另起多个queue抢同一GPU。GPU3只跑计时任务。

## 配置规模与晋级规则

通常48配置=12日程锚点与控制+24选择器变体+12联合配置；每配置四任务，共192个筛查作业。不可行日程会明确skipped，不能用重复配置补数。
筛查选最多4个互补候选：最高R4、最差任务损失最小、固定EARLY的选择改进、近同工作量日程；不要求全部含新模块。
确认默认再加BASE/EARLY、COVER/EARLY、COVER/LATE、BASE/ONE四锚点，通常32个ACC作业。已有真实匹配输出可复用。
三个near-work日程的整数差异自动记录；不在0.5%内就不能称近同工作量。即使完全同token-layer，也要另给实测延迟。

时间充裕时，除预声明困难面板外，并行推进以下高价值工作，不重新扫无界超参数：

- 原始六臂逐题对齐，检查MMBench提点来自判别改变还是答案格式/解析改变；CPU先统计有效解析率和答案分布，再按规则抽例，不因此阻塞主矩阵。
- 对BASE/COVER分别统计EARLY/LATE互补与配对NLL，算逐题oracle及成本，判断上一轮控制器是否存在真实空间。
- 若训练控制器，先赢“按Stage1分别选择最优固定日程”的对照。不能只学会COVER→LATE、BASE→EARLY就称为逐题动态。
- 记录问题组间响应差异、交换前后各组保留质量、接受交换数与选点重叠。检验代理是否与真实损伤相关，不能用代理上升自证准确率。
- 利用已有数据补OCR、计数、小目标、真正同图多问。冻结子集规则，不根据结果挑子集。
- 若有已经就绪的现代模型环境，给最终一个候选做迁移smoke/独立任务；不在本轮临时下载大模型拖住主实验。

## 验收看什么

日程改进：在不更多的视觉工作量下能否胜过EARLY；若只是TTFT更好或精度更好，画完整代价曲线，不强行同时宣称。
选择模块：固定EARLY/最终K是否有增益；current_exchange是否已足够；历史是否超过当前；组覆盖是否超过标量平均；随机交换能否解释变化。有效则逐项删掉无用复杂度。
综合结果：MMBench优势是否保住，POPE/MME损失是否减轻；R4之外必须列原始四指标与不确定性。历史和当前都失败不妨碍保留好的日程；日程失败也不阻止固定日程下验证选择模块。

## 最后回传

打包runtime、bridge与模型改动diff、全部matrix/state/summary、完整逐样本预测与官方分数、计时原始csv、逐层token与KV记录、失败日志和具体未完成队列。
必须区分192个筛查作业、32个确认作业和真实完成数，不把任务条目当实验数，不把复用EARLY/TWO当独立重复，不补造缺失指标。
计时报告cuda mean、median、p90以及固定生成长度/自然EOS策略。warmup、同步、backend和输入一致；微基准不能替代端到端，0.3–0.8ms差值需结合逐样本计时和轮次漂移解释。
保留旧cuda计时口径，另列host wall TTFT及其边界，避免把CPU准备/同步开销从新模块成本中漏掉。
不要求每个小实验前请示；不因为某个科学假设被否定就停止其他有价值分支；到墙钟截止保留结果和续跑清单。
