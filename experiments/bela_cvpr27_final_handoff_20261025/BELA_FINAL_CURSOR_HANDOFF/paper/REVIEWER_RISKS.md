# 模拟 CVPR Reviewer 独立审查

**Reject risk R1: "This is STAR-Pro with a changed schedule."**
Evidence required: policy selects retire depth/identity from genuinely input-conditioned *available* signal, not a fixed E1; show best fixed schedule, random schedule distribution and predicted task need vs actual future benefit. If this evidence fails, remove novelty claim.

**Reject risk R2: "CoRePrune already studies conditional removability."**
Explain no perturbation replay; use cached COVER structure and current query interactions, and demonstrate real selector cost. Still acknowledge conceptual overlap explicitly.

**Reject risk R3: "Stage1 and Stage2 are arbitrarily composed."**
2x2 crossover same W and same K: Layer9/Penultimate vs dual COVER; Stage2 baseline vs proposed. If improvement independent, do not claim synergy.

**Reject risk R4: "Improved POPE but weak GQA/OCR."**
Focus on absolute scores, POPE TP/FP/TN/FN and GQA relation/attribute/count categories, TextVQA failures. False positives can grow even when F1 improves.

**Reject risk R5: "Efficiency proxy is misleading."**
Measure REAL full end-to-end image-to-first-token latency, decoder-only prefill, Fixed32, selection overhead, peak memory on same GPU; report pre-pruned ViT token work and stage counts. A2's W equals A0's but Q differs.

**Reject risk R6: "Benchmark and Dense protocol drift."**
Audit original prediction IDs and benchmark revisions. Report MME Total/1852 and MME Perception with actual dense reference separately. STAR-Pro/QCPruner external figures are not fair comparisons until matched.

**Reject risk R7: "A single small method increment is not CVPR-worthy."**
Must have clear mechanism, strong W1024, independent validation across task types/backbone, no label-guided test tuning. If it remains a schedule tweak, state limits and consider weaker contribution framing.
