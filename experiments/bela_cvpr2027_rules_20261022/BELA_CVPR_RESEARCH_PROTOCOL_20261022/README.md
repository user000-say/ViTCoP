# BELA / ELVA CVPR 2027 — 可复用研究准则包

本包的目的不是替服务器实际修改模型，而是让 ChatGPT、Cursor、代码 Agent 在每一轮研究中遵循同一套**可证伪、可追溯**的投稿决策纪律。

主要文件：
- `AGENTS.md` — 项目长期研究准则（优先阅读）
- `.cursor/rules/bela_cvpr_research.mdc` — Cursor 项目内 alwaysApply 规则
- `.agents/skills/bela-research/SKILL.md` — 供其他支持项目技能的 Agent 参考；是否自动加载取决于实际平台
- `prompts/CHATGPT_NEW_SESSION_PROMPT.md` — 新 ChatGPT 对话手动粘贴的启动 Prompt（不能自动写入 ChatGPT 系统）
- `research/DECISION_2026-10-22.md` — 已阅读文献与下一步独立研究判断
- `research/experiment_matrix.csv` — 全部候选 Token 轨迹及状态 NOT_RUN
- `tools/schedule_audit.py` — 离线精确预算与层数校验
- `tests/test_schedule_audit.py` — 正常、逆向、异常和边界测试
- `CURSOR_HANDOFF_CN.md` — 接入当前工程的逐步执行书

使用方法：
```bash
python -m unittest discover -s tests -v
python tools/schedule_audit.py research/experiment_matrix.csv
```
所有研究代码只验证数学日程可行性，**没有进行任何 GPU ACC/TTFT 实验**。
本包直接可复制到 Cursor 工程根目录。请在真实工程 commit 上执行 P0，再决定何时修改代码。
