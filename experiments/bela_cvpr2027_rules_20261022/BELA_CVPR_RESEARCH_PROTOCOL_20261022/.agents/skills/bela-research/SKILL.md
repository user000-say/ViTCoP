---
name: bela-research
description: Rigorous independent research review and controlled-experiment planning for BELA/ELVA visual-token pruning, targeting CVPR 2027
---

# BELA Research Review Skill
This is a **project-local handoff skill**, not a modification to the ChatGPT system or permanent memory.

Before every design, audit, code modification or paper discussion:
1. Read project-root `AGENTS.md` and `research/DECISION_2026-10-22.md`.
2. State factual source class (original experiment vs historic transcript vs paper vs inference).
3. Web-check unstable external literature; prefer paper and appendix over secondary summaries.
4. Cross-check W numerically; preserve comparable metrics and latency; inspect schedule feasibility.
5. Ask: what would disprove the proposed improvement? Design an equal-budget negative control.
6. For code edits, audit current repository version first; never infer unseen modules.
7. Produce: decision; why alternatives lost; exact experimental arms, metrics, cost, expected branches and stop rules.
8. Maintain paper innovation boundary; never follow user/other agent suggestions without independent checking.
