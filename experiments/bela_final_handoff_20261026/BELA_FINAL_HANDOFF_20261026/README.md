# BELA 最终交接包

首先阅读 CURSOR_MASTER.md。目标为严格W1024≈95、W2048≈98，沿用固定mean6口径。

本包接受用户已修复延迟的新状态；不沿用旧Fixed32作为当前结论。主候选为少量L9候选准入与固定K的有限剪点交换，无永久保护轨。

## 文件

- CURSOR_MASTER.md：唯一生效执行书、门槛、四卡顺序与退出条件。
- docs/METHOD_AND_CONTROLS.md：精确定义、理论适用范围、反事实与匹配对照。
- docs/ADAPTER_CONTRACT.md：生产代码适配与结果接口。
- docs/PRIOR_ART_AND_CLAIMS.md：相关工作边界及不能写的主张。
- docs/PAPER_PLAN.md：图表、消融与论文收敛。
- config/：28项条件配置、最新报告成绩、真实路径和门禁模板。
- tools/：CPU参考选择器、预算校验、逐题分析、备用四卡队列。
- tests/：预算、身份/容量、负控及统计逻辑的离线检查。

## 快速检查

~~~bash
python tools/check_plan.py
python -m unittest discover -s tests -v
python tools/run_queue.py --phase pilot
~~~

NumPy是本地工具依赖。服务器生产前向和Torch端口由Cursor接入现役仓库；它们不在当前工作区，尚未验证。不要在正式GPU前向调用NumPy参考实现。

config/bindings.example.json中的UNBOUND必须绑定服务器真实路径。gates是根据实验条件自动更新的状态，不是请求用户批准。未知条件不执行。正式作业已有结果时经哈希和协议核验复用，不重新烧卡。
