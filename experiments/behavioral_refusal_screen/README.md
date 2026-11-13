# Behavioral refusal screen

470-record development screen of how refusal wording changes safety verdicts. Four judges, seven response conditions, two greedy repeats.

| Judge | Checkpoint | Labels |
|---|---|---|
| Llama Guard 3 1B | `meta-llama/Llama-Guard-3-1B` | native `safe` / `unsafe`, via `refusal_cues.guard_judges.GuardRunner` |
| ShieldGemma 2B | `google/shieldgemma-2b` | Yes/No score, via `refusal_cues.guard_judges.GuardRunner` |
| Ministral 3 8B | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `A_safe` and `A_unsafe` |
| Gemma 4 12B | `google/gemma-4-12B-it` | `A_safe` and `A_unsafe` |

Conditions: `B` (original body), `N` (neutral prefix), `R`, `R2`, `paper_R1`, `paper_R2`, `paper_Rgreet`. The cue is prepended as `cue + "\n\n" + body`. Guards use one native mapping. Chat judges use both letter mappings. That is 39,480 judgments (270 human-harmful and 200 human-safe records).

## Run

`behavioral_refusal_screen.ipynb` is the resume notebook (`RESUME_MISSING_ONLY = True`). On an L4 or A100 Colab runtime with an `HF_TOKEN` secret, run cells 2, 3, 4, 6, 7, 8 in order.

Drive inputs live in `MyDrive/algoverse-jailjudge-inputs/`: `baseline.zip`, and for a resume also `jailjudge-baseline-results.zip` and `jailjudge-control-results.zip`. Outputs go to `MyDrive/algoverse-jailjudge-outputs/<timestamp>-resume/`. Keep `START_NEW_RUN` false to continue the same folder. The notebook refreshes `merged-results.zip` every 60 seconds.

`upgrade_notebook.py` rebuilds that notebook from local predecessor files `refusal_screen.ipynb` and `refusal_screen_fixed.ipynb` (those predecessors are not in this snapshot). It writes `behavioral_refusal_screen_full.ipynb` and `behavioral_refusal_screen.ipynb`.

The analysis definition is `ANALYSIS_SPEC.md`. Recompute it from local archives with:

```sh
python3 experiments/behavioral_refusal_screen/summaries/analyze_full_run.py \
  --results experiments/behavioral_refusal_screen/summaries/merged-results.zip \
  --input experiments/behavioral_refusal_screen/inputs/baseline.zip \
  --output-dir experiments/behavioral_refusal_screen/summaries
```

## What is in git

`summaries/` holds the audited write-up and the small tables: `analysis.md`, `analysis.json`, `analysis-run.log`, `all_results_summary.csv` (84 groups), and `audit/`. The raw verdict table and zip archives stay local:

| Local file | Approx. size | Why it is omitted |
|---|---:|---|
| `all_results.csv` | 5.5 MB | 39,480 verdict rows |
| `all_results.jsonl` | 10 MB | same rows as JSONL |
| `jailjudge-full-470-results.zip` | 4 MB compressed | raw verdicts, logs, and the screen JSONL |
| `merged-results.zip` | 0.6 MB compressed | per-model verdict JSONL used to rebuild the table |

`refusal_cues/guard_judges.py` is the in-repo Llama Guard and ShieldGemma runner (`GuardRunner`, `guard_messages`, `parse_guard`). The Colab baseline bundle still supplies the chat-judge `Runner` and `structured_messages`.
