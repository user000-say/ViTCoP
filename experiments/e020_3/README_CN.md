# E020.3 · 阶段二算子与效率探索

先读 `CURSOR_HANDOFF_CN.md`。本轮保持“问题条件的中间证据保留”方向，新增真实的表示压缩算子，同时并行验证廉价选择器和容量空间。

主动作：在已有选择器选出K个token后，对满足空间与相似性条件的被删token做有限聚合。其他token直接删除。K、anchor IDs与位置不因聚合改变，但anchor向量会改变；不能再称为只保留原始真实patch特征。

三条主线不互相等待：

1. 同K验证有限聚合的精度作用。
2. 用更窄中间宽度或更早最终剪枝争取真实TTFT收益。
3. 独立替换末次选择器，检验便宜排序是否足够。

容量响应H组独立继续，不训练新的收益头。合并不利、当前动态预算无空间，都不会自动停止其他分支。

## 本地命令

```bash
python -m unittest discover -s tests -v
python demo.py
python make_matrix.py
```

`demo.py`只有合成张量，不产生VLM准确率。生产接入用项目已有PyTorch环境的 `runtime_torch.py` / `torch_backend.py`。不要把NumPy参考实现用于生产TTFT。

| 文件 | 功能 |
|---|---|
| recycle.py | 完整NumPy参考、相似度缓存、有限聚合与身份审计 |
| torch_backend.py | 设备端聚合、廉价末次选择器、视觉行回写 |
| runtime.py / runtime_torch.py | 两事件状态追踪、entry slot到原patch身份映射 |
| fast_selectors.py | pooled raw QK、key能量、随机选择的明确对照 |
| costs.py | 层边界、真实token trace、R4、Pareto与Amdahl诊断 |
| benchmark_cuda.py | 真实CUDA/host墙钟计时，不提供CPU伪TTFT |
| server_preflight.py | 服务器捕获真实特征后的CUDA数值检查；无GPU时明确失败 |
| capacity_response.py | 三预算完整响应的事后最优诊断，DP包含全部中预算策略 |
| matrix.json | 79个去重配置，含随机重复种子 |
| CURSOR_HANDOFF_CN.md | 四卡分轮执行、来源绑定、接线与回传要求 |
| RESEARCH_DECISION_CN.md | 独立选择、证据依据、反对理由与文献边界 |

当前工作区没有原E014模型源码，交付是完整新模块和接入协议；不声称已经修改并GPU验证了原整仓。已有选择器、mask、RoPE和KV压缩继续绑定真实实现。
