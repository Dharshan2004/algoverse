# JAILJUDGE original-response baseline

All four models judged the same 470 development records. Each row below gives counts for one repeat. The second repeat reproduced every verdict.

| Model | Label mapping | Harmful judged safe / 270 | Safe judged unsafe / 200 |
|---|---|---:|---:|
| Llama Guard | Native | 30 | 56 |
| ShieldGemma | Native | 162 | 5 |
| Ministral | A = safe | 4 | 117 |
| Ministral | A = unsafe | 2 | 154 |
| Gemma | A = safe | 105 | 6 |
| Gemma | A = unsafe | 102 | 6 |

Swapping the answer letters changed the interpreted verdict on 43/470 records for Ministral and 5/470 for Gemma in each repeat. The guards used native labels. All 5,640 verdicts were valid.

Archive integrity, dataset hash, record IDs, human labels, and complete record × mapping × repeat coverage were checked. Counts recomputed from the saved verdicts matched all four summaries. DFR@5 and DFR@3 require paired cue-condition judgments beyond this baseline.
