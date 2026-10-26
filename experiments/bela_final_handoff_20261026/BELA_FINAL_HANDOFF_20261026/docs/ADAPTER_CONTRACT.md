# 生产适配、结果接口与执行边界

## 1. Cursor先定位现役实现

用rg找到真实Stage1入口、RDRIFT/Q-Attn调用、physical slicing、position IDs、attention mask与KV生成路径。记录实际call map，不根据旧文档假设E3三处都用RDRIFT。尊重用户已完成的延迟修复。只修发现的具体问题，不重建一套评测框架。

在独立实验分支加入配置开关，默认原E3。先完成Native掩码回归；再导入CD原始ID；最后只对晋级分支接入CPU参考等价的Torch实现。不要直接修改第三方CD/STAR以适配自己的语义后仍叫“官方结果”。

## 2. 两个选择接口

on_vision：返回原始视觉ID、现役实际输入特征、Dual/L9/late guide及方向、native entry priority来源。参考实现需要C9/CLate两个[0,1]亲和矩阵；只从已经执行的ViT层取特征，构建/缓存成本计入方法。特征层、图像预处理与Patch网格对所有身份对照相同。

on_cut：输入当前alive原始ID、现役native_keep、native_scores、当前问题相关query_scores、入口anchor状态及真实K；输出最终keep ID及动作日志。full-grid分数向量在已删除位置可填0作存储占位，但任何选择或query估计不得读取那些位置。矩阵是静态来源统计，不是已删Token的深层hidden state。

所有操作遵循：唯一ID；当前subset；容量准确；相对顺序保留；特殊文本/图像边界Token不进入visual计数；不恢复先前已删ID。LLaVA当前为单图576Patch；跨模型/多裁剪时显式扩展canonical ID与实际budget，不能直接复用576假设。

COVER的entry_priority若没有天然逐Patch分数，优先使用生产保存的选择边际并记录；若需从真实guide导出，明确命名这个adapter变化，不能伪称完全复用原分数。没有得到真实输入时停止该项适配，不随机造数组让smoke通过。

## 3. CD移植的最小实现

相同图像网格/预处理下，调用现有CD选择器导出K32/K64原始ID及排序前的贪心序列，放入BELA相同输入特征和Decoder后端。该实验叫“CD identities with BELA backend”，与完整官方CD系统结果分开。

CD64_L2_STATIC在真实L12/L20保留原贪心序列前17、10，并保持物理序列相对顺序。检查K32选择是否与K64前32一致。官方代码最后torch.sort排序，不能对空间排序结果直接取前缀。若为了调试离线缓存mask，不能拿该版本省略CLIP文本塔/选择器后的耗时作为正式CD速度。

CD172_E3必须真实选择172个ID；CD64加108 filler应另起实验ID和控制，不能混用。本迭代不默认开发这类filler方案。

## 4. 样本清单与元数据

每个任务、每个pilot/full范围都保存显式ID文件，文件内容含有unit、ids、expected_questions及选样seed。POPE/GQA按image选样后展开实际问题；保留跨子集相同image_id供聚类。task定义、OCR prompt、最大生成长度、answer parser与scorer都记哈希。

predictions.jsonl最小示例（由官方评测完成后导出，不由本工具重新判分）：

~~~json
{"task":"gqa","sample_id":"q123","image_id":"img7","score":1.0,"pred":"red","target":"red","valid":true}
{"task":"pope","sample_id":"random:q12","image_id":"img7","split":"random","label":0,"pred_label":1,"valid":true,"pred":"Yes"}
{"task":"textvqa_val_ocr","sample_id":"q88","image_id":"img8","score":0.6666667,"pred":"...","valid":true}
~~~

score范围[0,1]，TextVQA是官方软分。POPE的pred_label来自现役官方解析，valid保留原始解析状态；null不能被分析脚本当作no或被丢掉重算“有效样本F1”。pooled/macro通过命令行明确指定。MME仍使用官方成对/类别复合评分，在额外文件导出每图配对信息，不用通用均值脚本算MME。

run_meta.json至少包含：arm、task、actual_W（每样本实际值一致后汇总）、seed、run_tag、spec_hash、sample_count、ids_sha256、predictions_sha256、scorer_hash、checkpoint_hash、latency_fix_commit、complete。另记code_commit、prompt_hash、precision、attention_backend、模型名、真实K轨迹、特征来源、时间边界。metrics.json含status=complete和真实官方得分。

备用runner用BELA_EXPECTED_SPEC_HASH传入arm配置哈希，BELA_HANDOFF_CONFIG传入本配置路径，BELA_EXPECTED_IDS_SCOPE传入pilot/full。适配器读取这些值并核对，不是只把它们打印进meta而不应用方法。ids_sha256和predictions_sha256对对应文件的原始bytes取SHA256。actual_W必须由trace实际计数得到，不能抄名义配置。

run_tag应绑定修复commit、checkpoint、评测设置和样本版本；不要为了复用旧目录保留已过期tag。

## 5. 实验入口

Cursor在生产仓库写一个小adapter脚本，接受如下参数：

~~~text
--arm NAME --budget 1024|2048 --task TASK --ids ABS_MANIFEST --output ABS_DIR --run-tag TAG --seed N
~~~

它调用现有lmms-eval入口、严格应用对应方法与样本、导出上述三类文件。CUDA_VISIBLE_DEVICES由外部exclusive调度或备用runner绑定一个物理GPU，adapter内部使用本地cuda:0，不能再硬编码物理0/2/4/7导致错卡。

复制config/bindings.example.json到服务器自己的运行目录，填真实python/adapter/repo/run_tag/output_root以及所有pilot/full IDs文件。所有UNBOUND移除后由Cursor直接执行，无需再询问用户。config/gates.example.json由Cursor依据真实报告填写value和evidence路径；这不是权限审批。

若现有exclusive调度已经可靠，直接复用它。备用run_queue.py采用argv与shell=False、每GPU串行、繁忙卡延期、不kill进程；当前机器只测了dry-run，没有验证nvidia-smi/CUDA执行。发生真实适配故障先解决一个smoke，不在四卡上重复相同失败。

## 6. 复用、恢复与真实耗时

已完成ACC应尽量复用，条件是相同样本、checkpoint、prompt/scorer和数学前向，且源文件可追踪。将源run_meta和未改动预测保留在运行目录，另写equivalence报告，清楚区分“导入旧ACC”与“新GPU生成”。备用runner只自动复用与当前meta/hash严格匹配的结果；不匹配时先由Cursor完成可审查的导入或用现有队列跳过该作业，不应重复全量。

runner遇到部分输出会标记需要adapter恢复，避免覆盖；Cursor实现按sample_id继续、验证manifest后收尾，再继续队列。不要删旧预测重新假装无失败。临时空目录/已验证可重现的日志可按正常开发流程处理，不需要向用户逐项确认。

最新正式计时采用相同输入、warmup、batch、precision、backend与强制生成长度；方法顺序交错，报告分布。全量trace、离线mask缓存与调试分析关闭。只比较用户修复后的版本，旧1204/1189/930不作为新准入门槛。若输出变化造成生成长度不同，Fixed32必须真正强制相同步数，不能混用自然停止时间。

## 7. 条件执行与资源

phase pilot（合并low/high）、pilot_low/high只需runtime_bound；candidate需mechanism_W；controls/confirm/ablation需confirm_W及相应winner。value为true但无evidence仍不放行。条件由Cursor完成分析后自主更新，未知不当作通过。

winner只能来自已经跑出的同预算配置。外部backbone使用真实生产入口，由Cursor单独给出样本和预算映射，不由本地参考工具假装支持。当前单图32层的预算检查不能自动推广到不同层数。
