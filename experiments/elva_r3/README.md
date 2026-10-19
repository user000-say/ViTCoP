# ELVA R3：两档预算、真实 COVER、先收敛 Stage1

先读 `CURSOR_HANDOFF_CN.md`。本目录是完整策略/接线/队列代码包，替换本轮 ELVA R2 的实验目录；仍需 Cursor 在现有服务器仓库绑定真实 COVER 和 VLM task 回调。没有伪造 `/home/...` 源码补丁、GPU 运行或新准确率。

- 只保留 W=2048（88.9%）与 W=1024（94.4%）。主日程分别为 172×4+90×8+32×20 与 60×10+28×10+12×12。
- 首批 Stage1 五组×两预算。Stage2 需要真实结果冻结，学习头和 merge 不进入本轮队列。
- `elva/clip_stage1.py` 给出实际单次CLIP→选点→原投影路径；`elva/low_budget_cover.py` 给出NumPy/Torch保护覆盖实现。
- `elva/cover_binding.py` 执行显式绑定的选择器，检查容量、源码指纹、输出与调用审计，原COVER缺失时明确阻塞。
- K60 主对照为明确分名的 PCOV60/b12：保护Top48，用12个名额补特征覆盖。原COVER参数化K60/b11作为可选强对照；172-pool→60只作诊断，可能恒等于Top60。
- 队列按当前 GPU UUID/进程/显存检查，保留采样记录与逐例 trace；不同 phase 可复用完全相同方法的已验证结果。
- `visual_audit/` 用于真实小面板热点图/共有独有图，`coverage_report.py`用于K60覆盖与显著性代价图。绘图不进入正式计时。
- `legacy_reference/` 仅保存 R2 回归测试所需的旧生成器/队列；不要用它启动旧 83 配置队列。

```bash
python verify.py
python make_catalog.py
python gpu_resources.py --gpus auto --workers 3 --reserve 1
```

Python 3.9+（仅对当前实际Python版本运行了本地测试）、NumPy；真实任务沿用现有 PyTorch/Transformers 环境，不为本包升级核心依赖。图示另需 Matplotlib/Pillow。本地测试中合成 fixture 不能进入真实结果。

本地最终记录：R3 48项中43通过、5项Torch相关skip；R2回归48项中47通过、1项Torch相关skip。无新真实VLM ACC/TTFT。完整日志见 validation。
