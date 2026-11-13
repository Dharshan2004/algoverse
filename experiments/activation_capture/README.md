# 4-judge activation capture

Residual-stream capture for the same 470 development records and seven conditions as the behavioral refusal screen. One greedy pass per prompt: 19,740 prompts (Llama Guard and ShieldGemma once each, Ministral and Gemma under both letter mappings).

| Judge | How it is loaded | Decision position |
|---|---|---|
| Llama Guard 3 1B | `refusal_cues.guard_judges.GuardRunner` | first non-blank `safe` / `unsafe` token (`parse_guard`) |
| ShieldGemma 2B | `refusal_cues.guard_judges.GuardRunner` | last prompt token, Yes/No score |
| Ministral 3 8B | `refusal_cues.inference.Runner` from the behavioral baseline bundle, NF4/bf16 | last forced JSON token before the A/B letter |
| Gemma 4 12B | same chat `Runner` | same JSON decision position |

Layers follow `refusal_cues.protocol.depths`: Llama Guard `[3, 5, 7, 8, 10, 11]`, ShieldGemma `[6, 9, 11, 14, 16, 19]`, Ministral `[8, 11, 15, 18, 22, 25]`, Gemma `[11, 16, 21, 26, 31, 35]`. Each prompt stores `decision`, `final_token`, `response_mean`, `cue_mean`, and `body_mean`.

## Code

| Piece | Path |
|---|---|
| Capture, shard IO, index, patching example | `src/activation_tools.py` |
| Notebook builder | `build_notebook.py` |
| Colab notebook | `activation_capture.ipynb` |
| Llama Guard / ShieldGemma runner | `refusal_cues/guard_judges.py` |

Rebuild the notebook after editing `src/activation_tools.py`:

```sh
python3 experiments/activation_capture/build_notebook.py
```

## Run

On an L4 (A100 also works) with the `HF_TOKEN` secret, run cells 1 → 2 → 3 → 4 → 6 → 7 → 8. Cell 10 is an optional patching example (`RUN_PATCH_EXAMPLE = False`).

Drive inputs: `MyDrive/algoverse-jailjudge-inputs/baseline.zip`, plus `all_results.csv` or `jailjudge-full-470-results.zip` from the behavioral screen. Outputs: `MyDrive/algoverse-jailjudge-activations/<timestamp>/`. After a disconnect, keep `START_NEW_RUN` false and rerun 1 → 2 → 3 → 4 → 6. Completed shards are skipped. Set `TAKEOVER_LOCK` only when the previous runtime is gone (the lock goes stale after 15 minutes).

Default settings in cell 1: models `shieldgemma`, `llama_guard`, `gemma`, `ministral`; shard size 47; storage dtype float16 with a bf16 fallback on overflow.

Offline tests:

```sh
python3 -m pytest experiments/activation_capture/tests/test_activation_tools.py -q
```

`tests/capture_cpu_full.py` and `tests/notebook_cells_full.py` also download public stand-in weights and need the local baseline archive plus `all_results.csv`.
