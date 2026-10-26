# Source-grounded related work / engineering facts (2026-10-25)
The citations below are primary papers and official public repositories; none is the user's current server.

| Method | Original research finding | Code inspection / configuration detail | Why BELA must not simply borrow it |
|---|---|---|---|
| STAR-Pro | Feature coverage before fusion; attended token identities evolve between decoder layers. Pivoted QR candidate and progressive attention. | LLaVA official overlay pins upstream revision `c121f0432da27facab705978f83c4ada465e46fd`; STAGE1_SCORER default `qr` in `llava_arch.py`, SELECTOR has many alternative ablation branches; LLaMA Stage2 has `TEXT_AGG_MODE` default `average_all`, `CUSTOM_PRUNING_SCHEDULE`; deployed T32 `64×12,17×8,10×12`, T64 `128×12,37×8,18×12`. Exact QR/input normalization matters. | BELA "coverage + progressive" already overlaps heavily. Benchmark copy of its S1/S2 is a DIAGNOSTIC only, not a contribution. |
| CoRePrune | Deletion depends on layer depth and simultaneous deletion set, rather than individual attention rank; vision encoder progressive pruning and question-conditioned reverse greedy after first 8 LLM blocks. | Original paper's CLIP schedule {8,10,12,14,16,18,20}; seven-task Stage1 ablation for LLaVA K64 Early 92.0, Progressive 97.2, Late 96.2. Public code of these core functions was **not independently verified here**. | CoRePrune already proves conditional removability; don't call a bounded rescue rule a theoretical new discovery. |
| QCPruner | Query-conditioned coverage both in target and representative; utility of concept anchors improves candidate relevance. | Paper LLaVA K32 final, after 8 dense 576-token decoder blocks, nominal decoder W5376 (not W1024). LLaVA K64 W6144. Published code not independently verified. | Not same total compute. It warns that "diverse visual candidates" may miss actual answer objects; don't clone weighted facility location. |
| MiCo | A common information-coverage proxy bridges visual entrance and task-conditioned selection. | Public repo https://github.com/MAX-7237/MiCo ; source core logic not fully audited line by line this turn. Ablations: Stage1-only n=1 is strong and larger pool n>1 adds gains on its own model. | Trivially reusing Stage1 similarity inside Stage2 is already prior art. |
| HAP | Attention-head prompt alignment differs by head; head averaging hides query evidence. | Official public repo https://github.com/baokou-fw2/HAP ; its PAQ is LLM text→vision head alignment. | Cannot apply HAP LLM head weights to CLIP pre-LLM CLS heads; require correct architecture and different signals. |
| HiPrune | Multiple depths offer object-centric vs global candidates. | Official repo https://github.com/Danielement321/HiPrune | A 3-layer or 2-layer ViT candidate selector alone cannot be claimed novel. |
| AgilePruner | Image-dependent visual redundancy / attention versus diversity tradeoff. | Repo https://github.com/cvsp-lab/AgilePruner ; rank/effective-rank complexity costs. | Do not justify dynamic K simply with image entropy without prediction validation. |
| SIEVE / Information Horizon | Text-space absorption and task-dependent visual need depth, including OCR exceptions. | Publication analyses rather than production patch in this audit. | Need own counterfactual/paired evidence; a lower K does not prove harmful tokens. |

## Read primary sources
STAR-Pro https://arxiv.org/html/2609.05916
STAR code https://github.com/EasonAI-5589/starpro
STAR source stage1 https://raw.githubusercontent.com/EasonAI-5589/starpro/main/llava/model/llava_arch.py
STAR source stage2 https://raw.githubusercontent.com/EasonAI-5589/starpro/main/llava/model/language_model/modelling_llama_star.py
CoRePrune https://arxiv.org/html/2609.26484
QCPruner https://arxiv.org/html/2609.19990
MiCo https://arxiv.org/html/2609.34330
HAP https://arxiv.org/abs/2608.23921
CVPR2027 https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines

## Specific warning on public STAR code
`llava_arch.py` is a LARGE multi-method ablation codebase and contains legacy `thcp` and other branches. A valid reproduction must identify `METHOD=star_pro`, `STAGE1_SCORER=qr`, target T and actual emitted original patch IDs, and ensure `TEXT_AGG_MODE` corresponds to deployed paper settings. Do not blindly grep only the first `thcp` branch then describe it as QR.
Its custom schedule option is supported but doesn't make the 2×2 trivially composable with the BELA codebase; verify original IDs, attention mask, position and KV.
