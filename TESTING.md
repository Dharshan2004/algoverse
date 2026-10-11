# T4 compatibility test — 11 October 2026

Both models completed the default synthetic B/R smoke on a Colab Tesla T4 (14.56 GiB), using NF4 double quantization and FP16 compute. This is the user-approved compatibility configuration, not the research protocol's BF16 configuration.

| Check | Ministral 3 8B | Gemma 4 12B |
|---|---:|---:|
| Completed judge rows | 40 | 40 |
| Additional identity calls | 2 | 2 |
| Identity equality | Pass | Pass |
| Saved-artifact audit | Pass | Pass |
| INVALID outputs | 0/40 | 4/40 |
| Peak allocated GPU memory | 6.253 GiB | 7.634 GiB |
| Complete input token range | 240–952 | 255–868 |
| Six finite saved state vectors per row | Pass | Pass |

Six protocol unit tests also passed inside Colab. The offline four-variant plan generated 80 inputs without model calls. The completed GPU tests used 84 total inference calls; failed loading attempts did not reach inference.

## Scope of evidence

The audit verifies row completeness and uniqueness, unchanged response bodies, strict semantic mapping, finite scores, probability-mass and semantic-margin arithmetic, model revision consistency, and the six saved CPU activation vectors per row. The runner checked identity output/token/log-probability agreement and exact saved-state equality. These checks do not establish benchmark accuracy, a valid refusal direction, or causal effects.

The synthetic fixtures do not approach the 4096-token boundary. BF16 execution, real dataset audits, near-boundary cases, nonidentity intervention validation, direction estimation, behavioral gates, and final evaluation remain outstanding. No final-test data was used.

## Reproduction

- Ministral checkpoint: `f6fae9795746f63c9be8344932f01275f3c63734`; tested runner commit `b25f33deeed9593302d672ad59d44dfff73b8880`.
- Gemma checkpoint: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; tested runner commit `8d981eb1c7768a7e3225aed1d6892d8de4abe53e`.
- Python 3.13; torch 2.11.0+cu130; transformers 5.19.0; accelerate 1.15.0; bitsandbytes 0.50.2; huggingface-hub 1.33.0; mistral-common 1.12.0.
- Each model ran in a separate process. Ministral's process exited and GPU use returned to 107 MiB before Gemma started.
- Artifacts include environment lock, immutable checkpoint IDs, exact tested runner source, JSONL outputs, all state tensors, audit JSON and logs. Outputs and weights are not committed to this public repository.

## Fixes found during testing

1. Corrected the Ministral tokenizer adapter: use `MistralCommonBackend` itself for encoding and decoding, not its internal `MistralTokenizer`.
2. Stream and retain Colab subprocess logs so loading failures expose the actual exception.
3. Require native BF16 capability for protocol mode; emulation does not make T4 a native BF16 device.
4. Record the native template backend instead of hashing `None` when Mistral has no Jinja template. The original Ministral run configuration contains the old placeholder hash; the bundle includes an explicit provenance correction without altering its original results.
5. Added `audit_smoke.py` and made the notebook audit a prerequisite to summarizing saved results.

## Gemma formatting finding

The four invalid rows were baseline B for synthetic-01 and synthetic-03 under both mappings. Raw outputs were `thought\n<channel|>A` or `thought\n<channel|>B`, reached the four-token cap, and had first-token A/B mass between approximately 0.000002 and 0.000005.

The native processor was checked directly with `enable_thinking=False` and `True`. False appends the checkpoint's native empty thought-channel block; True leaves thinking enabled. Thus the requested native thinking-off switch was applied. No thought text was stripped to manufacture a valid verdict, no label was forced, and the token budget was not expanded. Gemma's native formatting needs further screening before research use; an artifact-audit pass is not a formatting-gate pass.

Ministral remains the research site's first candidate. These synthetic outcomes do not qualify either model for final scientific evaluation.
