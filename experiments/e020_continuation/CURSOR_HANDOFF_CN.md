# E020 四卡执行书：72配置容量，默认6小时，可扩到12小时

这是可执行算法模块及批量工具包，未改写你的模型主干。当前ChatGPT环境没有服务器最新项目仓库；不能声称已经在原仓库打好补丁。沿用已经成功跑完24/24 ACC的环境、数据和评测器，不重建环境，不重写原QED/COVER，不改已完成的历史结果。

本轮核心：固定每层token数量，让真实后续任务损失监督“哪些patch值得继续交互”。COVER FIRST_SHIFT R4=96.50是主要现有基准。EARLY 96.10只是辅助参照。TTFT正在补，不能先写新方案已经更快。

## 一、开工即并行，不逐项等用户批准

物理GPU为0/2/4/6。默认0、2执行独立ACC筛查，4收集反事实配对，6专用测速；配对结束后0/2/4一起跑学习模块和确认评测。进程只见local cuda:0。

用户上一轮TTFT目前可能仍占用0/2/4/6四卡。开工先读取这批任务的实际状态，已有测时完成后再复用相应卡；不要在任何正在测速的卡上叠加ACC或采集任务。

GPU6若仍在执行上一轮用户已启动的TTFT，先让它完成，读取其原始日志/设备/测时边界。不要杀其他进程，不在同一GPU重叠测速。可先加`--skip-initial-timing`，待卡空闲再单独启动timing队列。

0–45分钟同时做四件事：

1. 从成功运行入口导出真实逐层token trace、checkpoint/code/prompt/输入语义/缓存策略以及四任务metric keys，填`runtime.json`。明确是完成block后剪枝还是另一语义。
2. 将`dispatch.refine`接到首个剪枝事件，后续事件原实现不变；reference完全调用原路径。运行`python checks.py`，服务器应执行Torch parity项。做每臂8–16样本的真实shape/patch ID/位置/KV smoke，随后关闭hot path debug。
3. 给现有ACC/TTFT脚本接收`job.json`的薄封装，填`bridge.json`三种argv。输出下述result schema。已有E019 bridge可直接适配新config字段；不要重新实现模型/官方评分器。
4. 给collector实现真正的checkpoint分支adapter：导出首个剪枝前的state，clone真实hidden/cache/mask/position/RoPE，并运行剩余模型的teacher-forced答案NLL。并行恢复历史CLS分层日志，只回传来源和结果，不阻塞主队列。

## 二、唯一需要接到模型里的算法调用

代码均在包根目录，放入PYTHONPATH。每个请求独立Hook并reset，不在不同问题间共享状态。

```python
from module_torch import PairScorer, ContinuationHook
from dispatch import refine

# 模型/投影只加载一次；真正部署时复用。projection是固定[D,8]随机矩阵，
# 单独保存，hash写入feature_protocol_hash；NumPy/torch/collector使用同一份。
model = (PairScorer(spec['model_path']).to(device).eval()
         if spec['kind'] == 'continuation' else None)
hook = ContinuationHook(model=model, projection=projection, debug=False)

# 例如在零起算block0后，缓存prompt问题组→当前视觉patch的已有响应。
# group_ids来自原始问题token身份，排除system/template/答案；两检查点一致。
hook.capture(previous_question_response, previous_original_patch_ids)

# 在首个剪枝event的原选择器已得到baseline_indices、还未gather时调用：
keep = refine(
    spec, hook, scores, question_response, baseline_indices, original_patch_ids,
    after=completed_blocks, depth=model_depth,
    next_after=config['continuation_horizon_after'],  # 最终压缩点；THREE也是13
    final_k=config['events'][-1][1],
    hidden=current_visual_hidden, encoder_rank=original_encoder_rank,
    patch_xy=original_normalized_patch_xy,
    generator=sample_seeded_cuda_generator,
)
# keep是当前视觉序列的相对下标，按原顺序排列；原项目负责安全gather。
```

`scores`采用原项目可复用的单patch标量；QED若没有独立标量，用同条件Q-ATTN作为shortlist和特征，仍保留原QED的初始集合。此选择写入source并在所有方法一致使用。

不要开启全层`output_attentions=True`替代已有快速路径。复用已有响应或仅算必要prompt Q/K切片，额外开销全部计时。视觉序列在问题之前的causal结构下，不宣称视觉token本身读取了后方问题；问题条件来自text→vision响应。

reference走原路径，不为它额外计算本模块历史响应或投影。current-only可省历史capture；full/no-context/shuffled、rank-drift和temporal覆盖控制需要真实历史。各方法的必要额外工作计入各自TTFT，不能把reference也强制变慢来缩小开销差。

## 三、配对数据：约1500–2300个真实动作标签起步

使用独立、可用的标注训练/calibration图像，按canonical image/content身份排除最终评测图像。优先覆盖不同问题类型，不只采MMBench。建议192–256个独立图像、BASE/COVER均衡，FIRST_SHIFT/EARLY状态交错，最多256–384个state，每state6动作，组大小1/4混合。不要先完整遍历某一臂让超时后只有单臂数据。

标签是baseline中间集合与等大小交换集合的完整后续答案平均NLL差。先跑一次baseline复用，然后最多6个swap分支。后续各事件维持原selector；禁止两个分支共享被原地修改的cache。

**如果现有模型没有任意层resume接口，优先使用本包`replay_adapter.py`。** 它从相同输入完整重跑，仅在首event覆盖保留的原始patch IDs，校验真实prompt前缀hash相同；离线多算prefix，不要求实现新suffix循环。backend只需复用现有teacher-forced模型调用，返回hook抓到的prompt snapshot和答案NLL。脚本提供标准causal-shift NLL函数。两个分支必须使用相同原始输入、确定性eval、原始位置和同一剪枝/缓存语义；重跑前缀开销算采集成本，不能算线上开销。

此模式让`pairdata.py --adapter replay_adapter:make --config replay_config.json ...`调用；replay config包含`backend: project_backend:make`、与runtime完全相同的`source`、`backend_config`。后端需要的两个方法及返回字段写在脚本头部，完整replay和baseline复用逻辑已经实现。无需为此重建VLM forward。

`pairdata.py`顶部给出精确adapter协议；`source_hash`必须等于`build_matrix.digest(runtime['source'])`。state中的`next_after`表示最终压缩的完成block数，不是THREE中间事件。

示意命令由Cursor填真实adapter/config路径：

```bash
python pairdata.py --adapter project_e020_adapter:make --config pair_config.json --out /ABS/PROJECT/e020_pairs.npz --states 384 --actions 6
```

把这条argv数组填入`runtime.collection.command`，out参数可用`{out}`。worker自动限制物理GPU4、保存partial shard，超时后用已完成样本继续。样本不足时明确记录原因，独立ACC轴继续跑；不能用随机权重假装已有模型。

数据按图像65/17/18分为train/validation/test；两臂、同图多问、不同层和动作共享分组。4个同容量小头：full、current-only、no-context、shuffled labels。模型选择只看validation；test报告一次。collector的答案NLL只能作为离线监督，不能进入线上特征。

每16动作尝试一次identity-fixed诊断。若真实全层视觉KV purge尚不支持，先完成policy配对和独立轴，明确记录机制诊断缺失；不得将普通clone当作已证明纯信息吸收。支持时接口必须返回`visual_kv_purged_at_horizon=True`和真实`forced_final_patch_ids`。

## 四、72个不同配置，按价值排序运行

`CONFIGURATION_CATALOG.json`列出全部组合，`build_matrix.py`用真实runtime重建，不执行目录中的示例源身份。

| 轴 | 配置数 | 内容与目的 |
|---|---:|---|
| A 固定分配 |20|2入口×6已有日程及4个FIRST邻域；已有原始结果同条件复用 |
| B 新模块 |20|2入口×10项：full/current各4个组大小/轮数组合，另no-context和shuffled各1 |
| C 迁移/成本补偿 |16|2入口×EARLY/THREE/首段K108/K124×2个full设置 |
| D 强简单控制 |16|2入口×随机g1轮2/4、随机g4轮1/2、单分数TopK、简化rank-drift、E019 current/temporal集合修复 |
| 合计 |72|四任务筛查最多288个作业；不是宣称已经跑完72组 |

A中的预算邻域为首event前后一个block的近W分配，以及FIRST的116改为108/124。后两档W分别少/多80，给新模块开销留出真实效率比较空间，不把更贵结果冒充同成本。

组g1最多替换2/4个，g4最多替换4/8个；随机控制匹配组大小和尝试上限，实际接受数量仍须逐样本报告。若模型经常只接受1轮，另从已保存trace作相同实际交换数的随机小面板复核，避免把保守程度误认作效用预测。

rank_drift_control只是可复用简单控制，**不是正式DIPrune复现**。E019两个覆盖代理也不是新的remaining-utility模型；保留它们是为了判断直接任务监督是否有必要。

## 五、6小时排程及命令

| 时间 | GPU0/2 | GPU4 | GPU6 |
|---|---|---|---|
| 0–45分钟 |接线、smoke；独立评测尽早启动|接collector与真实prefix分支|完成上一轮TTFT并核验边界 |
| 约45–105分钟 |A/D独立筛查|真实配对收集；结束后CPU拟合4个小头|现有强基线成对交错测速 |
| 约105–195分钟 |B/C学习模块筛查，四任务完整优先|同左|继续测速/漂移anchor |
| 约195–345分钟 |最多8配置×4任务确认，优先COVER FIRST基准与强新模块|同左|确认候选TTFT，每项1024样本 |
| 最后15分钟 |汇总完整结果、配对统计和错误迁移；预算够则做视觉密集stress|同左|输出设备/计时/漂移记录 |

时间是预算而非吞吐承诺。512/1024请求、全量GQA/MMBench的耗时由旧日志给出的ETA决定；超预算任务保留pending并可续跑。不要把72配置全部强制全量，挤掉核心对照和确认。

筛查建议每任务128个有效unit起，POPE保持正负和图像簇，MME保持完整双问题及类别。确认使用独立面板或官方全量；若已反复用相同面板调参，明确标记development/tuning reuse。全量原始预测可真实过滤和重评分作screen，不能把全量R4直接粘为小面板分数。

填好runtime和bridge后：

```bash
python workflow.py runtime.json --bridge-config bridge.json --out /ABS/PROJECT/e020_run_01 --gpus 0,2,4 --timing-gpu 6 --hours 6
```

需要12小时容量时：

```bash
python workflow.py runtime.json --bridge-config bridge.json --out /ABS/PROJECT/e020_run_12h --gpus 0,2,4 --timing-gpu 6 --hours 12 --independent-seconds 5400 --collection-seconds 5400 --learned-screen-seconds 10800
```

中途失败只重试该行一次，其他轴继续。相同runtime/模型hash的输出可恢复；替换权重或source后用新输出目录。未知ONE事件可以保持null，目录自动少2配置，不猜其层位。

## 六、真实结果输出协议

现有runner收到job后写`result.json`：

```json
{
  "job_hash": "从job复制",
  "metric_key": "从job复制，且实际使用同一个官方指标",
  "panel_ids_hash": "从job.panel复制且验证实际输入ID",
  "metric_value": 0.0,
  "records_path": "/ABS/真实逐样本预测和评分文件"
}
```

上面的0.0只说明JSON字段类型，不是默认结果。实际运行必须来自官方评分。GQA/POPE/MMBench用百分制、MME用官方分数，R4分母固定61.9/85.8/1852/64.1，最后一项是MMBench。POPE accuracy/F1不混用。

TTFT用相同物理GPU与固定1024样本，按job里的ABBA协议与COVER FIRST_SHIFT成对交错；mean/median/p90和host wall、选择器全部开销、峰值显存同时保存。ACC与TTFT读取相同问题span。旧结果只有同设备/同边界才可直接对比；不同卡不混平均。

`summarize.py`只汇总四任务完整行；`paired_stats.py`对真实预测做图像簇配对bootstrap，POPE重算F1，MME重算acc+acc_plus。输入格式见脚本头部。四任务明细、置信区间和相对96.50的差值都必须回传，不只回传R4。

## 七、这一轮还要留下的科研产物

1. 严格同前缀交换的 held-out NLL真实收益分布；模型/零预测器/当前重要性/随机的比较。
   `workflow.py`自动输出`heldout_mechanism_diagnostics.json`，比较同一前缀上已采样单步动作的真实收益；它不是多轮oracle或新ACC结果。
2. 被交换patch的原始身份和响应轨迹，接受交换数、耗时、正确→错误/错误→正确案例。
3. BASE/COVER×固定/新模块的因子结果，判断新模块是否只在特定入口生效。
4. identity-fixed纯交互诊断能完成多少、KV行为具体是什么，不能完成就如实列缺项。
5. 最多8个确认配置（4个固定anchor＋4个互补challenger）；有余量就补TextVQA/OCRBench/grounding中已配置的两类stress，缺数据不让核心队列停下。
6. 外部近邻CoRePrune、DIPrune、Bifocal的官方源码/许可证/可运行状态与预算映射记录；能复用现成实现就进入余量队列，不以简化控制代替正式基线。现代模型迁移作为下一轮主验证，不在这轮宣称完成。

若学习模块没有超越强固定日程，也应完整回传；这会决定下一步是改善标签/特征，还是冻结简单方法去补泛化与效率证据。不要为了得到正结果临时改R4或反复查看确认面板调阈值。
