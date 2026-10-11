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

## Gemma formatting follow-up — 11 October 2026

A user-requested attempt to resolve the four INVALIDs did **not** fix them. A no-inference check compared native rendered text with the decoded actual input IDs for both thinking settings: they matched exactly. Thus the thinking-off option reaches the tokenized model input; no missing-toggle or render/tokenize discrepancy was found.

One opt-in diagnostic, `--format-reminder`, repeats the existing output mapping and exact-A/B instruction after the evidence JSON. It changes no saved response body, native template, output parser, decoding constraint or four-token limit. It is synthetic-only, recorded as `prompt_format=reminder_v1` / `protocol_prompt_match=false`, and is **not** adopted as the research prompt.

| Follow-up check | Result |
|---|---:|
| Completed rows / additional identity calls | 40 / 2 |
| Identity / saved-artifact audit | Pass / Pass |
| INVALID under A_safe | 2/20 (10%) |
| INVALID under A_unsafe | 2/20 (10%) |
| Verdict counts | SAFE 22, UNSAFE 14, INVALID 4 |
| Peak allocated GPU memory | 7.648 GiB |
| Input token range | 287–900 |
| Verdicts changed relative to native run | 2/40 |

The same four conditions remained invalid. They emitted `-thought\n<channel|>` or `- The user is`, rather than an exact code. The reminder is an unsuccessful diagnostic, not a validated repair.

Tested inference/protocol source: `b1f4e96f6375e83e50303cd3110df9ab26473032`; same pinned Gemma checkpoint and original T4 software environment. Results SHA256: `5d890a01594f94316fc9c4c75fe571112f25c9af15a9ebe9cc8cffdac33ce6ac`. Total GPU calls across the original smoke and follow-up: **126**, within the 160-call allowance.

Seven unit tests pass locally, including unchanged evidence/mappings and rejection of thought-channel outputs. All 80 default native prompt constructions were compared with the previous implementation and are unchanged. The audit now reconstructs each prompt hash and reports formatting separately from artifact integrity, so an integrity pass cannot be mistaken for a format pass.

Next: validate the unchanged native protocol on an available native-BF16 GPU (L4/A100). This is a precision diagnostic, not a promise of a fix. Do not strip generated prose, force A/B, expand the token cap, or adopt a new prompt to report zero invalids. Ministral remains the first candidate; both models still require the site's actual development gates before scientific use.
