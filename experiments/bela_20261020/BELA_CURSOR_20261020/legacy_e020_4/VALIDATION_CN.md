# 本地验证

本地运行50项unittest：48项通过，2项因无PyTorch跳过。

完成：矩形/完整当前相似度参考等价、关闭路径、非排序keep、跨crop隔离、prefix方向、零特征、浮点dtype、非法索引、两阶段身份、重复事件、预算正反算、配对时间bootstrap、官方评分回调聚类重采样、配置去重、完整源码语法编译，以及原合成双阶段demo。

未完成：PyTorch CPU/CUDA数值对齐、fp16/bf16真实特征与生成回归、原仓库接线、任何VLM精度或真实TTFT。服务器必须运行validate_current_cuda.py；缺CUDA时明确失败，不自动生成通过报告。

Python核对扩展目录101项，四任务全展开404个筛查作业；这不是已运行模型实验数量。work_verified.json包含token-layer正反算；不作为实际时延。
