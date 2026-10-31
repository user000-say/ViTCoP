# 交付核查

日期：2026-10-31。编译在 `/home/xiangyue/ViTCoP/experiments/bela_cvpr2027_paper_20261031` 执行。

## 编译

本机没有 `xelatex`。`paper/build.sh` 在找不到 XeLaTeX 时调用 `/tmp/tectonic/tectonic` 0.15.0，命令为 `tectonic -X compile --keep-logs --keep-intermediates main.tex`。模板文件没有改。

`cvpr.sty`、`preamble.tex`、`ieeenat_fullname.bst` 的 SHA256 与交接账本一致：

- cvpr.sty `1cd7f434b13d41c475df4a1901a5d82d9ca8b80b8cf6205dc4602a5c164286b4`
- preamble.tex `f666a85903f49929395951e7afb620e96d4e208334623f23135298b4d7f06228`
- ieeenat_fullname.bst `e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc`

产物 `paper/main.pdf`：7 页，612×792 pt Letter，PDF 1.5，约 687 KB。日志中 Citation undefined、Reference undefined、Overfull 均为 0。Tectonic 在第 6 遍后停止继续因为 bbl 时间戳来回变化；参考文献编号已经写入 PDF，抽页可见 [1] 起的 IEEE 条目。

`lineno.sty` 与 `algorithm.sty` 有上游非 UTF-8 字节警告，来自宏包本身，不是正文。`TU/ptm/m/n` 的字体形状警告来自 ctex 与 CVPR Times 设置并存，模板文件未改。渲染页上的中文和拉丁字母均可读，没有整段缺字。

## 视觉检查

用 `pdftoppm -r 120/130` 看了全部 7 页，并在算法改短后重看第 4–5 页。

- 第 1 页：匿名、review 横幅、双栏、摘要和引言。`O(n^2 d)` 上标正常。
- 第 2 页：表 1 在栏内，六行分数与末行 93.56% / 95.34% 对齐。相关工作接在表后。
- 第 3 页：式 (2)–(9) 在栏内，代码索引 `hidden_states[9]`、`attentions[8]` 可读。
- 第 4 页：通栏图在页顶，图注说明层号从 1 计、秩漂移只用于第一次剪枝。式 (10)–(13) 未冲出栏。
- 第 5 页：算法 1 的 18 行都落在左栏内，第一行以“CLS 和显著性”结束。结论、LLM Usage Statement 在参考文献之前。
- 第 6–7 页：参考文献，没有未解析的问号引用。

主文加图表约到第 5 页上半，参考文献占用其后。没有为了页数改字号、边距或行距。实验节没有启用。

## 主张

中心句与代码一致：初始化是空间原型，更新是受覆盖变化约束的有限交换，第一次剪枝是秩漂移，第二次是问题注意力。数字追溯见 `research/claim_audit.md`。同一代理完成评审和修改，不代表独立审稿。
