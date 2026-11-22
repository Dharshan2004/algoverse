# Refusal wording and safety judgments

This research repository tests whether prefixing a saved response with refusal wording changes a safety judge's verdict. The primary behavioral run pairs the original response with a neutral prefix and five refusal cues across 470 JailJudge development records. The activation experiment captures the same prompts at six decoder depths.

| Judge | Decision |
|---|---|
| Gemma (`google/gemma-4-12B-it`) | Greedy constrained JSON under both A/B label mappings |
| Llama Guard (`meta-llama/Llama-Guard-3-1B`) | Native safe/unsafe response and optional category IDs |
| ShieldGemma (`google/shieldgemma-2b`) | Native Yes/No scores against the policy |
| Qwen3Guard (`Qwen/Qwen3Guard-Gen-8B`) | Native Safe/Unsafe/Controversial label, scored under two views |

Gemma's only chat verdict format is `json_ab_constrained_v1`. It uses the same JSON-requesting system and user messages as unconstrained generation would receive. There is no assistant verdict prefill. A decode-only constraint permits the two complete objects `{"verdict":"A"}` and `{"verdict":"B"}`, followed by EOS. Malformed verdicts are invalid. The three guards retain their native interfaces.

## Module map

| Location | Responsibility |
|---|---|
| `refusal_cues/protocol.py`, `structured_output.py`, `inference.py` | Evidence, label mappings, JSON prompt, token constraint and Gemma generation |
| `refusal_cues/judges.py`, `guard_judges.py`, `qwen3guard_judge.py` | Checkpoint revisions, dispatch and native guard parsing |
| `refusal_cues/adapt_hf_dataset.py`, `prepare_refusal_screen.py` | Dataset adaptation and provenance |
| `refusal_cues/refusal_screen.py`, `screen_report.py` | Five-cue screen, paired comparisons and exploratory trio selection |
| `refusal_cues/jailjudge_baseline.py`, `judge_metrics.py`, `activation_tools.py` | Full development run, paired metrics and activation shards |
| `scripts/`, `configs/`, `experiments/`, `tests/` | CLI entry points, settings, experiment notebooks and offline tests |

Run commands from the checkout. For CPU tests install `python -m pip install -e . pytest torch numpy pandas`, then run `python -m pytest -q`. Colab notebooks install their GPU dependencies. Dataset preparation requires an explicit `--revision` while the original pin file is unavailable. Existing historical reports describe their original cohorts.

[Experiment instructions](experiments/README.md)
