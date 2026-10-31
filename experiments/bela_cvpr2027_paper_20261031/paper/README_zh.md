# BELA 中文 CVPR 初稿 v0.3

本版为非实验部分中文修改稿，使用当前对话上传的 CVPR 2027 LaTeX 模板实际编译。包含摘要、引言、相关工作、方法、结论和 LLM Usage Statement；实验章节没有撰写或启用。

## 本版内容

摘要从视觉序列的推理开销与剪枝必要性展开。引言依次交代 VLM 能力、视觉序列及注意力二次计算项、冗余研究、相关方法和具体研究问题。

中心是视觉代表基础上的受覆盖约束条件更新。方法明确两次覆盖的职责，新增“移除损失—加入补偿”的覆盖变化分解，并给出容量、集合修改范围与累计变化界。这些是所给集合算子的性质，不当作下游准确率的理论保证或未经实验建立的机制发现。

写法对照覆盖 15 篇代表论文；正文实际引用 54 项文献。新增了直接相关的 MoB，参考文献采用 PyramidDrop 的 CVPR 正式题名 Conical Visual Concentration for Efficient Large Vision-Language Models。没有使用 nocite 填充书目。

## 文件结构

- main.tex：中文双栏匿名审稿版入口。
- main.pdf：本版实际编译的 PDF。
- sec/：全部正文分节；4_experiments.tex 仅为未启用的注释文件。
- references.bib、main.bbl：BibTeX 数据库与实际生成的参考文献。
- cvpr.sty、preamble.tex、ieeenat_fullname.bst：从本轮上传模板原样复制。
- figures/：框架图 PNG 与可编辑 SVG。
- build.sh：本地编译脚本。
- notes/writing_comparison_zh.md：15 篇论文的写法对照和本版修改说明。
- notes/author_checks_zh.md：实现定义、结果与机制主张的核对说明。
- notes/reference_metadata.json：54 项书目的来源信息，不含论文全文或完整摘要。
- notes/build_verification_zh.md：本版编译与排版核验记录。

## 编译

安装完整 TeX Live 后，进入本目录运行：

```bash
bash build.sh
```

也可使用：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Overleaf 的编译器请设为 XeLaTeX。需要 ctex、xeCJK、Fandol、algorithm、algorithmic 及 CVPR 模板依赖的常用宏包。

ctex 用于支持内部中文稿；main.tex 显式恢复 CVPR 的 Times、Helvetica、Courier 西文字体，并同步中文标题字号和字重。没有修改模板样式文件、版心、栏宽、栏距或规定字号。本次 PDF 由本地 XeTeX 与 BibTeX 离线编译。

## 定稿顺序

1. 核对 notes/author_checks_zh.md 中的 C1—C4。Native COVER 和 RDRIFT 的生产定义仍为接口级描述，不能据此宣称论文已满足完整复现要求。
2. 对照实际运行配置，确认 QA 参考算子、双深度度量和 Decoder 各剪点的接线。
3. 加入实验正文并启用 main.tex 中的对应 input。当前结论自动编号为第 4 节，加入实验后会自动变为第 5 节。
4. 中文内容确认后再翻译英文，重新按 CVPR 8 页正文要求安排篇幅和图表。当前中文稿用于内容修改，不是英文正式提交稿。

摘要和引言中的实验数字来自已提供的讨论记录；本版没有重新运行 GPU 评测，也没有加入未知的延迟、FLOPs 或跨模型结果。
