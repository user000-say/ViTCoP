# 当前状态

更新：2026-10-31。本轮是中文非实验稿，没有启动 GPU，没有改生产算法。

## 工作区

- 生产仓库：`/home/xiangyue/ViTCoP`
- 当前 HEAD：`0204d8e9344020e8905d46fba241a9cd1caaca9a`，分支 `env/v100-4model-qacover`
- 六任务问题条件更新的运行提交：`6b011ea5f3dbdee54a25c8df3a5e4d0736d8040d`
- 论文工程：`experiments/bela_cvpr2027_paper_20261031/`
- 规程：CCFA-Skills 固定提交 `91ca9e2801698466f934b557a75ce46db8ba46b5` 的源码。本机未安装技能客户端。已读 humanization、common、idea-optimizer、paper-writer 的 SKILL.md，并按该顺序改同一套 `paper/` 源文件。

## 本轮完成

- 中心主张写入 `research/storyline.md`：有限交换受覆盖变化约束；初始化与更新不是同一个目标。
- 中文摘要、引言、相关工作、方法、结论已改。实验节仍不输入。
- `paper/main.pdf` 由修改后的源码用 Tectonic 0.15.0 实际编译，7 页 Letter，review 模式，无未定义引用、无 Overfull。
- 评审、数字核查和未关闭问题分别在 `research/review.md`、`research/claim_audit.md`、`research/open_questions.md`。

## 未做

- 没有重跑模型，没有把 W=2048、视频或 TTFT 写进正文。
- 没有重画框架图。图注已按真实剪枝规则更正。
