# Qwen3Guard behavioral and activation experiment

Open `qwen3guard_experiment.ipynb` in Colab on an L4 or A100. Put `jailjudge_screen.jsonl` in `MyDrive/algoverse-jailjudge-inputs/` and run all cells. Supply the other judges' `all_results.csv` in that folder for a combined comparison. Reconnect to the same Drive output directory to resume.

The 8B checkpoint uses its native response-moderation template. Its first output line determines Safe, Unsafe or Controversial. Invalid first lines stay invalid. Two scoring views map Controversial to unsafe or safe. Across 470 development records and seven conditions, two repeats give 6,580 generations and 13,160 scored rows. Capture stores one set of activations per prompt and indexes both views.

`refusal_cues/qwen3guard_judge.py` owns inference and resumable verdict chunks. `activation_tools.py` captures residuals and `judge_metrics.py` computes paired metrics.

If Drive already contains the original input archive, the notebook can read just the dataset member from it. Code always comes from the checkout. Set `ALGOVERSE_REF` before the first setup cell to clone a PR branch.
