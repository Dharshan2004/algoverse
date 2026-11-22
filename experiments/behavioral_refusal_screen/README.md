# Behavioral refusal screen

Open `behavioral_refusal_screen.ipynb` in Colab on an L4 or A100. Enable the Hugging Face token for gated checkpoints. Place the adapted `jailjudge_screen.jsonl` in `MyDrive/algoverse-jailjudge-inputs/` and run all cells. The notebook checks the dataset digest and saves completed records to Drive. Rerunning resumes those records. The combined CSV and paired metrics appear under `MyDrive/algoverse-jailjudge-outputs/repo-screen/`.

The full run covers B (original), N (neutral) and five refusal cues, with two greedy repetitions. Gemma uses the decode-only JSON verdict under both label mappings, with the same prompt as unconstrained decoding and no answer prefill. Llama Guard and ShieldGemma use their native decisions. Qwen3Guard-Gen-8B emits one native label per generation, scored separately with Controversial counted as unsafe and as safe. Keep repetitions and mappings separate.

From the checkout, the two commands below score the original and prefixed responses for one checkpoint. Repeat for `llama_guard`, `shieldgemma` and `qwen3guard`.

```sh
python -m scripts.run_jailjudge_baseline gemma "$DATA" runs/baseline
python -m scripts.run_jailjudge_baseline gemma "$DATA" runs/controls --controls
```

## Data preparation and five-cue screen

`configs/refusal_screen.json` reconstructs the five-cue configuration. The original dataset revision pins are unavailable. Pass a known dataset commit via `--revision` when downloading. The preparer records the resolved revision and file hashes. The full JailJudge runner independently verifies the expected adapted dataset digest.

```sh
python -m scripts.prepare_refusal_screen --dataset jailjudge --revision "$DATASET_REVISION" --out "$DATA"
python -m scripts.run_refusal_screen --model gemma --dataset jailjudge --input "$DATA" --out runs/screen
python -m scripts.report_refusal_screen runs/screen
```

Use `--limit 20 --dry-run` on the screen command for a CPU plan. This screen uses B and the five refusal cues. The full notebook also includes N. Historical summaries in this folder retain their original cohort.

If Drive already contains the original input archive, the notebook can read just the dataset member from it. Code always comes from the checkout. Set `ALGOVERSE_REF` before the first setup cell to clone a PR branch.
