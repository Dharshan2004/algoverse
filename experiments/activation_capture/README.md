# Activation capture

Run `activation_capture.ipynb` on a Colab L4 or A100 with gated checkpoint access through the HF_TOKEN secret. Put `jailjudge_screen.jsonl` and the behavioral run's `combined/all_results.csv` (named `all_results.csv`) in `MyDrive/algoverse-jailjudge-inputs/`. Run all cells. Keep the same run directory to resume completed 47-record shards.

The plan covers 16,450 prompts across 470 records, seven conditions and five prompt mappings. Gemma uses two letter mappings and the same JSON decoding path up to the verdict token. Llama Guard, ShieldGemma and Qwen3Guard-Gen-8B each use one native prompt per condition. Qwen’s two scoring views share the same activation tensors.

The notebook writes six-layer residuals, token spans, shard markers, and a verdict-agreement index to Drive. The capture and index implementation lives in `refusal_cues/activation_tools.py`.

If Drive already contains the original input archive, the notebook can read just the dataset member from it. Code always comes from the checkout. Set `ALGOVERSE_REF` before the first setup cell to clone a PR branch.
