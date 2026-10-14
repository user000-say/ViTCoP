# Cursor 执行说明：只做可裁决的交互层实验

请先读 README_CN.md。此包的实际基础是用户上传的 E014 两个主源码，不能冒充已有 E017 源码。已有历史强配置、原 evaluator、prompt、数据版本和停止规则优先保留。

## A. 安装与源码身份

设 `BUNDLE` 为解压后的本包目录，`REPO` 为完整的现有服务器仓库，`MODEL` 为真实本地模型目录；路径变量请由执行者填写，不要默认当前目录就是服务器仓库。建议在独立工作树/副本上做。

```bash
python "$BUNDLE/scripts/install_e018.py" --repo "$REPO" --require-historical-cover
python "$BUNDLE/scripts/install_e018.py" --repo "$REPO" --require-historical-cover --apply
```

第一条仅检查。若 hash 不匹配，**不要加一个强制覆盖选项，也不要把 E014整个文件覆盖到 E017**。请检查 `E014_to_E018.patch` 的5个 decoder 挂接与2个 CLIP 挂接，合并到实际文件。保留最新 Stage1 的选点、token顺序、helper和历史变量名。对应 runtime 的来源发布接口是 `publish_stage1(selected_ids, original_core_ids, cls_attention, name, exchange_budget)`。

原 E010 helper、完整 builder 和精确 E013 helper 是已有服务器依赖，本包未臆造。若历史 COVER helper真的不存在，BASE可用于接口smoke；不能把 BASE/PROXY 的结果登记成 HIST-COVER。

## B. 必过的小模型回归

本地包CPU检查（无需torch）：

```bash
python -m unittest discover -s "$BUNDLE/tests" -p test_e018_cpu.py -v
```

实际服务器安装后，从完整repo根目录执行真实 PyTorch/SDPA 回归：

```bash
cd "$REPO"
python -m unittest discover -s "$BUNDLE/tests" -p test_e018_torch.py -v
```

此测试使用**实际修改过的 LlamaSdpaAttention**，无需下载模型权重。必须全部通过：完整合法前缀统计、dense identity、旧QED单次结果、两次删除后的所有文本与原RoPE、两种cache语义的prefill/decode、共前缀同final IDs/同未来操作、精确delta W、实际策略再采集、最多两次gather、PROXY controls。

如果当前 Transformers 与这份历史source API不兼容，使用当前原项目可运行的环境并调整兼容点，不盲目升级或降级整套环境。记录实际 torch/transformers/driver；任何API改动后重复这组测试。

## C. 注册实验与核实旧强结果

在安装完、开始正式采集之前运行：

```bash
python scripts/make_e018_registry.py \
  --repo "$REPO" --model "$MODEL" \
  --data-manifest "$RAW_POOL_MANIFEST" \
  --evaluation-files "$ACTUAL_EVALUATOR" \
  --output "$RUNS/registry.json"
```

将实际输出的hash设置为 `E018_PROTOCOL_ID`，后续保持同一已冻结的整体实验协议。注册对真实权重文件做SHA可能读盘较久，只做一次必要的冻结，不加入每题计时。canonical image ID应能识别同图跨数据集/跨题重复，最好来自原图内容hash。

先核实原历史强臂与 raw scores，尤其 COVER POPE 85.32/86.48冲突。保存：入口原patchIDs与顺序、补入/删出IDs、原Stage2 survivors、逐题答案、数据/evaluator/source身份。可靠已有原始日志可复用。

## D. 准备prefix（ACC/标签可用；TTFT不可复用）

raw JSONL 每条需要 `image_id, question_id, image_path, question, answer_ids`。answer_ids必须由真实 tokenizer和已冻结答案格式得到，不用本文示例虚构 token ID。可提供现有worker真实的 `prompt, question_span_input`；span是在未展开input token里的 `[start,end)`，只能包含用户问题。

自动构造默认vicuna_v1 prompt仅用于标准LLaVA。若与正式原worker不同，应提供原prompt。自动BPE边界校验失败就用原worker的确切span，不用长度近似替代。

```bash
export E014_VARIANT=FIXED_COVER E014_B=32
source scripts/e018_env.sh 32 STATIC
python scripts/prepare_e018_samples.py --model-path "$MODEL" \
  --raw-manifest "$RAW_POOL_MANIFEST" --out-dir "$RUNS/prefix_cover"
```

准备脚本检查当前 multimodal preparation 的签名、唯一embeddings、单图扩展形状。如果当前自定义arch不能适配，在原worker的MM展开之后、decoder之前调用 `scripts/e018_tools.py` 中的 `export_prepared`，导出相同字段。不要重新写一个近似图像/问题拼接流程。

BASE与COVER各准备一套真实prefix；来源值从实际CLIP选点取。保存prefix可提高离线反事实采集效率，但计时必须重新算vision；`measure_e018.py`对此另作处理。

## E. 第一轮：固定日程先裁决

只先做K32，对齐同source、selector、Stage1和官方评测口径：

```bash
export E018_EVENTS='8:32' E018_SELECTOR=Q_ATTN
source scripts/e018_env.sh 32 STATIC
python scripts/evaluate_e018.py --model-path "$MODEL" \
  --manifest "$RUNS/prefix_cover/manifest.jsonl" --output "$RUNS/one32.jsonl"

export E018_EVENTS='4:102,12:32'
source scripts/e018_env.sh 32 STATIC
python scripts/evaluate_e018.py --model-path "$MODEL" \
  --manifest "$RUNS/prefix_cover/manifest.jsonl" --output "$RUNS/two32.jsonl"
```

输出格式提供question_id和text，交给**原冻结官方 evaluator**；不要用本包给MME/POPE重造一个计分器。该导出脚本只提供greedy+EOS；正式原worker若还有separator停止规则/答案清理，应继续沿用原worker+修改后的decoder，先逐题核对两者，而不是悄悄更换生成协议。

两臂 W 都是2284。先32～64题检查，再全量。之后视差异决定是否补K64；不要立刻启动所有预算/层位/评分公式的大网格。

补充 old E014真实路径与 E018 `8:32 + QED + layer_local` 的身份回归，比较survivors和答案；如存在tie或微小数值差，登记实际原因。小模型等价通过不替代真实模型这项检查。

## F. 第二轮：真实续算收益，不先训练复杂模块

选定动态候选日程。例如先研究`4:80,12:32`，其all-retire-now W=2108。这个中间K仍是研究配置，别登记为最优方法。

```bash
export E018_EVENTS='4:80,12:32' E018_SELECTOR=Q_ATTN
export E018_CACHE_POLICY=layer_local
export E018_CONTROLLER=''
source scripts/e018_env.sh 32 COLLECT
python scripts/collect_e018.py --model-path "$MODEL" \
  --manifest "$RUNS/prefix_cover/manifest.jsonl" \
  --output "$RUNS/labels_cover_local_shard0.jsonl" --shards 3 --shard-index 0
```

另两张实验卡运行shard-index1和2，各输出独立文件；按图像分片，别用同一个输出文件互相追加。重复不同cache_policy应另开真实采集，不混到一个fingerprint训练。scrub_all至少先做可判断机制的子集。

```bash
python scripts/analyze_e018_labels.py \
  --labels "$RUNS/labels_cover_local_shard0.jsonl" "$RUNS/labels_cover_local_shard1.jsonl" "$RUNS/labels_cover_local_shard2.jsonl" \
  --split test --output "$RUNS/headroom_local.json"
```

此输出是局部fixed-future oracle与NLL/first-token统计，不能写成官方准确率。若 U 很小、代理无预测性、较晚处理反而多次有害，应先看具体失败样本，不把单调延后作为先验。

参考答案不唯一的任务不能把一个字面GT NLL等同官方正确率。第一轮可用答案格式确定的选择题、计数与OCR子集建立可信监督，再加入真实官方得分验证。

## G. 训练、标定、简单阈值与真实策略再采集

```bash
python scripts/train_e018_controller.py \
  --labels "$RUNS/labels_cover_local_shard0.jsonl" "$RUNS/labels_cover_local_shard1.jsonl" "$RUNS/labels_cover_local_shard2.jsonl" \
  --output "$RUNS/controller_full.json"
```

用`--ablate stage1/history/receiver/mass_only`输出独立模型；这些是同候选、同监督的控制器输入消融。全量test用于最终报告，不用test选择lambda/阈值。

简单mass阈值控制：

```bash
export E018_PROXY_THRESHOLD=0.02
source scripts/e018_env.sh 32 PROXY
```

0.02只是API默认值，必须在calibration集选到目标平均实际成本后冻结；不要把默认阈值当实验发现。

部署：

```bash
export E018_CONTROLLER="$RUNS/controller_full.json"
export E018_LAMBDA=0.03
source scripts/e018_env.sh 32 ONLINE
python scripts/evaluate_e018.py --model-path "$MODEL" \
  --manifest "$RUNS/prefix_cover/manifest.jsonl" --output "$RUNS/online_full.jsonl"
```

0.03同样是命令示例，不是推荐最优值。先在calibration集选择少量lambda，让实际平均W/TTFT对应目标成本；然后冻结并评test。`4:102,12:32`不可在有delay时仍声称W2284。

若第一检查点改变第二检查点的状态分布，再采集第12层：

```bash
source scripts/e018_env.sh 32 COLLECT
python scripts/collect_e018.py --model-path "$MODEL" \
  --manifest "$RUNS/prefix_cover/manifest.jsonl" --event 12 \
  --output "$RUNS/labels_onpolicy_l12.jsonl"
```

保留刚才真实控制器路径与lambda。COLLECT会在捕获第12层前使用它，GT仍只在捕获之后进入分支。记录state_id和rollout controller SHA。初始与再采集标签的重复状态需显式去重；训练脚本拒绝重复行，不默默加倍权重。同图split保持不变。

## H. 完整模型计时

第四张卡只做计时，保持GPU独占/空闲、warmup、固定32步与自然EOS另口径。不要用历史COMPUTE_EFFICIENCY结果测E018，它会遗漏新路径。

计时raw JSONL必须提供实际`prompt,question_span_input,image_path,image_id,question_id`；图像/文本预处理在计时之外，每次测量重算CLIP和projector。

```bash
export E018_TRACE=''
python scripts/measure_e018.py --model-path "$MODEL" \
  --raw-manifest "$TIMING_RAW_MANIFEST" --output "$RUNS/ttft_online.jsonl" \
  --warmup 5 --repeats 3 --tokens 32
```

该脚本严格产出32个token，EOS不提前结束；完整TTFT包含真实vision/Stage1/projector/reader/controller/prefill与首token选择。预处理/加载权重不计入，明确标注。它不替代自然EOS端到端延迟统计。

正式结果至少包括准确率、W分布、真实TTFT/Fixed32、所有新算子成本、每层KV长度、退出位置与失败样本。采结构日志的run与timing run分开；日志开关关闭不是为了隐去线上必须计算的算子。

## I. 四张卡与停止条件

| 卡 | 当前用途 | 先完成什么 |
|---|---|---|
| 实验卡A | 固定同W与Stage1身份 | K32单次/两次、BASE/COVER、逐题一致性 |
| 实验卡B | 反事实采集 | 共享prefix、now/later同future IDs、初始标签 |
| 实验卡C | 同步采集/机制控制 | shard、scrub_all、小细节任务；之后预测器评测 |
| 计时卡 | 独占时延 | 原强基线、固定日程、PROXY、ONLINE完整成本 |

物理GPU编号由服务器实际可用卡指定，别从历史“GPU4”字符串推定。使用CUDA_VISIBLE_DEVICES隔离后，脚本默认device是cuda:0。

1. 固定同成本和oracle均无空间：停止动态控制器扩展。
2. controller不及mass：先砍复杂输入/模型，不再用数学包装。
3. 去来源/历史无差异：删除相应创新主张。
4. 只有保留旧KV有效：调整机制故事，不能继续写已经吸收。
5. full score有涨点但TTFT更差：暂不算可投稿净收益。

通过这些裁决后再扩到近期强基线、现代VLM、跨任务预算与泛化。不要同时开另一轮Stage1公式搜索。
