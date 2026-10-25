# CVPR 2027 投稿稿件蓝图（此时是草案，不可预写成功结果）
2026-10-25 起与 GPU 实验并行。8页正式正文；引用和必需 LLM Usage Statement 可另页。请用官方 2027 LaTeX submission template，匿名。

## 推荐标题（根据是否真的产生新机制选择）
主方案：**BELA: Budgeted Evidence Lifetimes for Efficient Vision-Language Reasoning**
如果最终只是固定日程、没有任务条件决策：**Revisiting Visual Token Lifetimes under Tight Decoder Budgets**（突出分析，不能伪称 SOTA）。
备选：**When Should Visual Evidence Retire? Task-Conditioned Token Lifetimes in Vision-Language Decoders**。
这些只是研究标题，不构成成果或新颖性的保证。

## 摘要草案（英文；结果全部禁止虚构）
"Training-free visual token pruning reduces inference costs in large vision-language models, yet token budgets are often allocated using fixed stagewise schedules or single-token relevance estimates. We revisit the relation between token identity, survival depth, and answer quality under exact layer-average budgets. Controlled experiments reveal a task-dependent tradeoff between maintaining evidence through intermediate fusion layers and preserving a sufficiently expressive survivor set for later reasoning. [TBD: ONLY IF EXPERIMENTS SUPPORT IT — We introduce an efficient evidence-retirement mechanism that couples early visual candidate coverage with late question-conditioned evidence need, allocating visual-token lifetimes without additional model training.] Experiments on [VERIFIED BACKBONES] and [VERIFIED BENCHMARKS] show [INSERT VALIDATED ABSOLUTE SCORES AND REAL WALL-CLOCK COSTS], with diagnostic analyses of relationship reasoning, hallucination decisions, and OCR. Our findings clarify when visual evidence can be removed under stringent compute constraints."

## 8页内容分配（可根据官方模板版面调整）
1. Introduction (0.9页): Task-specific evidence lifetime failure + evidence figure, 3 bullet contributions only if verified.
2. Related Work + Setup (0.8页): FastV/PyramidDrop/VisiPruner/ATP-LLaVA/STAR-Pro/CoRePrune/QCPruner/MiCo/SIEVE; explain W and fair budget.
3. Evidence study (1.2页): A0,A2,A3,B1,B2 plus phase-specific interventions E1/late W1024 and prediction pairing.
4. Method (1.7页): Stage1 dual COVER, Stage2 real scorer/conditional retirement (if verified), exact W accounting, overhead.
5. Experiments (1.4页): W1024/W2048/W4096, official task scores/strong baselines, wall-clock latency, subsets.
6. Ablation and mechanism (1.2页): 2×2 Stage1×Stage2; timing vs selector; POPE confusion/GQA categories/OCR.
7. Limitations/Discussion/Conclusion (0.8页): single-backbone limits, no claim of universal absorption, no full FLOPs equivalence.

## Mandatory figures and tables
Fig1: under matched W, observed A2 vs A0 tradeoffs and B2 diagnostic (label B2 higher budget).
Fig2: ORIGINAL BELA Stage1 and the final legal Stage2 callsite after audit (avoid showing unimplemented modules).
Fig3: paired question examples, original images, Patch masks at each event, correct/incorrect, not cherry-picked only.
Fig4: accuracy vs measured true end-to-end prefill/TTFT, separate W1024/2048/4096; axes fully labeled.
Table1: LLaVA-1.5-7B full absolute scores GQA/POPE F1/MME Total+Perception/MMB/SciQA/TextVQA/AI2D; strong paper baselines as PAPER-REPORTED separate.
Table2: exact per-layer schedule, entry K, final K, W, visual-Q proxy, selector overhead, TTFT, Fixed32, peak memory.
Table3: ablations Stage1 single-vs-dual COVER; Stage2 current/late/identity; row with random controls.
Appendix: confidence intervals, POPE FP/FN, experiment IDs+code SHA, speedup measurement details, failure samples, extra budget curves.

## The 3-claim evidence ledger
C1 "intermediate evidence survival improves ..." requires same-budget controlled A2/E1 etc and paired task analysis.
C2 "question-conditioned retirement decides when..." requires a policy that uses only legally available runtime signals and clearly beats best fixed + random/cheap policy.
C3 "efficient" requires encoder+projector+LLM+selector timed together and fair strong baseline.
If C2 fails, remove C2 as contribution; rename thesis as systematic empirical study if remaining significance warrants submission.

## Prior papers (public sources)
STAR-Pro: https://arxiv.org/html/2609.05916 (public since Sept 05).
CoRePrune: https://arxiv.org/html/2609.26484 (Sept 22).
QCPruner: https://arxiv.org/html/2609.19990 (Sept 17).
MiCo: https://arxiv.org/html/2609.34330 (Sept 28).
SIEVE: https://arxiv.org/abs/2608.10489.
VisiPruner: https://aclanthology.org/2025.emnlp-main.955/.
CVPR2026 Information Horizon: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.html.
ATP-LLaVA: CVPR 2025; PyramidDrop: CVPR 2025.
STAR-Pro official repository: https://github.com/EasonAI-5589/starpro.
MiCo source: https://github.com/MAX-7237/MiCo.

## 论文交付格式与风险
- 2027 官方指南：https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines。
- 作者名单 / OpenReview profiles 截止注册日不可再增减；新规则要求 reciprocal reviewer quota。
- 必须有 LLM Usage Statement，不能隐瞒研究、代码与英文润色中使用的模型。草稿："LLMs were used to assist with literature organization, experimental-planning text, code scaffolding and language editing. All technical claims, code, and experimental results were independently checked and remain the responsibility of the authors." 请作者按实际用途修订。
- 文中不能用纯主观的 "SOTA" 形容未同协议复现的方法，不能将 MME Total 与 MME Perception 混比。
- 11/10 AoE 注册、11/16 AoE 正文、11/23 AoE Supplementary；没有延期。
