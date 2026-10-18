# 本次本地验证

{
  "tests_run": 41,
  "passed": 40,
  "skipped": 1,
  "failures": 0,
  "errors": 0,
  "skip_reasons": [
    "PyTorch absent locally; real CPU/CUDA parity is required on server"
  ]
}

数学和边界检查通过，包含关闭路径逐字节对齐、局部质量代数核对、跨crop拒绝、身份追踪、随机控制、raw-QK穷举均值、因果reader限制、容量DP穷举和命令行。

合成两事件流程与完整语法编译通过。没有真实VLM准确率、没有真实CUDA/TTFT结果；PyTorch/CUDA测试因本地环境缺失未执行。server_preflight.py用于服务器真实特征数值验证，不能替代完整模型验证。
