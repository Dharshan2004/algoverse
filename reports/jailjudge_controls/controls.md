# JAILJUDGE control results

Verified 17,640 unique control judgments across four models. Every expected source ID, human label, condition, mapping and repeat matched the baseline dataset and the planned scope. All input manifests matched the dataset SHA-256 `beb131113f4c0cfa7ffb142f074469abe08dc66689b5a55fb70c1f3e6475c012`.

B is the original response and N adds the neutral prefix. Both cover all 470 records. The five cue columns count human-safe records judged unsafe, out of 200 per cue. Counts are per repeat. All four models had zero invalid judgments and zero repeat disagreements in both baseline and controls.

| Model | Mapping | Missed harmful B → N /270 | False positives B → N /200 | Changed B → N /470 | R | R2 | paper_R1 | paper_R2 | paper_Rgreet |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama_guard | native | 30 → 30 | 56 → 49 | 15 | 31 | 32 | 31 | 25 | 31 |
| shieldgemma | native | 162 → 159 | 5 → 4 | 20 | 0 | 0 | 0 | 0 | 0 |
| ministral | A_safe | 4 → 5 | 117 → 109 | 21 | 110 | 105 | 115 | 112 | 104 |
| ministral | A_unsafe | 2 → 2 | 154 → 151 | 9 | 155 | 153 | 160 | 157 | 156 |
| gemma | A_safe | 105 → 105 | 6 → 3 | 9 | 3 | 2 | 3 | 3 | 2 |
| gemma | A_unsafe | 102 → 105 | 6 → 2 | 11 | 3 | 3 | 3 | 3 | 2 |

The five cues were tested only on human-safe responses. Full-set DFR@5 and the best trio’s DFR@3 cannot be estimated from these controls.
