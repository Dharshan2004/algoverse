# Refusal-cue experiments

These notebooks sit on top of the `refusal_cues` package. The package owns the protocol, the pilot runner, metrics, and the Llama Guard / ShieldGemma interface. The experiments own the 470-record screens and the activation captures.

| Experiment | What it runs |
|---|---|
| [behavioral_refusal_screen](behavioral_refusal_screen/README.md) | 470 records × 7 conditions × 4 judges × repeats and chat mappings |
| [activation_capture](activation_capture/README.md) | Residual streams for those four judges (19,740 prompts) |
| [qwen3guard_judge](qwen3guard_judge/README.md) | Qwen3Guard-Gen-8B behavioral screen and residual capture |

Judge code:

| Judge | Inference | Activations |
|---|---|---|
| Llama Guard 3 1B, ShieldGemma 2B | `refusal_cues/guard_judges.py` (`GuardRunner`, `guard_messages`, `parse_guard`) | `activation_capture/src/activation_tools.py` |
| Gemma 4 12B, Ministral 3 8B | `refusal_cues/inference.py` `Runner`, plus `structured_messages` from the Colab baseline bundle | same capture module |
| Qwen3Guard-Gen-8B | `qwen3guard_judge/src/qwen3guard_judge.py` `Qwen3GuardRunner` | `qwen3guard_judge/src/activation_tools.py` |
