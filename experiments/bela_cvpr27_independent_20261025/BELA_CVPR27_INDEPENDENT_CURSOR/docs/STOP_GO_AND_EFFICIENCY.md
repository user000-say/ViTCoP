# Mandatory independent decision record
Each claim must be supported by actual script/source run ID, not the strongest point from an exploratory test.

## Why no Stage1-only pivot based on user preferences
In P1 E1 outperforms A2 on some ACC but its Fixed32 increased by ~10%; E3 sits near Pareto, and W1024 late-first-cut still lags STAR-Pro on GQA/POPE. Those facts do not identify a single root cause. Only matched schedule 2×2 resolves the principal component defect.

## Thresholds (prioritized, NOT magical p-values)
- Full official GQA+POPE F1 should improve together vs best same-budget baseline with no systematic ScienceQA/TextVQA/MME/MMBench regression.
- Effects below dev-set error / paired CI or <0.1–0.2pp without broad robustness are not sufficient to add a module.
- New method real FULL TTFT and fixed output time within ~3% of best simple same-budget configuration unless quality gain is substantial and clearly shown on Pareto.
- W1024 should not be considered solved by a schedule that only raises mean6 while severely damaging MME or POPE.
- At minimum include a negative control: random selection or shuffled provenance at identical K/W/time.
- An observed 2×2 "interaction" can arise from incompatible hooks; physical ID/KV invariants must pass first.

## Stop/Go matrix
| Evidence | Action | Don't do |
|---|---|---|
| QR stage1 wins while Stage2 held | test independent Stage1 causal feature, same K/W | present QR as BELA original innovation |
| STAR stage2 wins while Stage1 held | audit text receiver/head normalization; one cheap alternative | append HAP+CoRe+STAR modules |
| Both win separately | select best single improvement first, then prove real joint synergy | assume combining always boosts quality |
| Neither wins | audit evaluator/checkpoint; inspect Stage1-only loss floor | unlimited K sweep |
| Any new module similar to random/shuffled | NO_GO and remove from paper | rename its weight as novelty |
| E1 wins ACC but much slower | leave as ablation, not headline | hide Fixed32 time |
| Source contract fails | stop invalid cross-over, use official independent full-system comparison | silently "fix" model internals |
