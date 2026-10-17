# Latest validation — 17 October 2026

The [50-record exploratory pilot](reports/pilot-2026-10-17/README.md) completed on both models with BF16 on an NVIDIA L4. Each has 400 judgments plus two passing identity calls. Both local audits passed condition completeness, source/body preservation, prompt hashes, checkpoint consistency, verdict parsing, score arithmetic, and six finite activation vectors in every saved state file. Ministral has 0/400 invalids; Gemma has 10/400. Runtime success and artifact integrity do not qualify the dataset or establish a causal result.

All 12 protocol/pilot unit tests and all 14 existing team metrics tests pass. The new launcher was separately dry-run validated for 400 planned conditions per model; the actual GPU run used the bundled existing inference script. The clean notebook was parsed and checked for absent outputs. Raw artifacts remain ignored; aggregate results are committed separately from them.

Earlier dated sections below retain their historical scope; their statements that BF16 was pending describe those earlier runs.

---

# T4 compatibility test — 11 October 2026

**Latest:** the native Gemma parser is corrected and freshly verified at **0/40 INVALID**, with four raw empty-channel wrappers retained. See [the verified correction](#verified-gemma-parser-correction--11-october-2026). Earlier results below document the legacy parser and unsuccessful prompt diagnostic.

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

## Verified Gemma parser correction — 11 October 2026

**Fixed: the unchanged native Gemma T4 run now has 40/40 valid parsed verdicts.** The earlier interpretation was too strict: `thought\n<channel|>` is an empty native channel wrapper, with `thought` serving as the role label, not reasoning content. The research site's parsing rule permits removal of known native control syntax.

The implementation now removes exactly one recognized empty Gemma wrapper, then still requires an exact A or B. It never removes nonempty reasoning, accepts an explanation, searches prose for a letter, forces a code, changes the input template or extends the four-token cap. Raw output and token IDs remain unchanged. New rows retain `raw_parsed_text`, `output_normalization` and parser version metadata.

References: [Google model-card thinking configuration](https://huggingface.co/google/gemma-4-12B-it#2-thinking-mode-configuration) and [vLLM's Gemma parsing utilities](https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/gemma4_utils.py), which document bare channel-role labels when thinking is disabled. Our rule is narrower than a general reasoning parser: only an empty channel qualifies.

| Corrected native run | Measured result |
|---|---:|
| Completed rows / additional identity calls | 40 / 2 |
| INVALID after native control normalization | **0/40** |
| INVALID by mapping | 0/20 A_safe; 0/20 A_unsafe |
| Empty wrappers normalized | 4 |
| Raw outputs not already exact codes | 4 |
| Length-cap flags retained | 4 |
| Semantic verdicts | SAFE 28; UNSAFE 12 |
| Identity / artifact / synthetic format checks | Pass / Pass / Pass |
| Six finite CPU state vectors per row | Pass |
| Peak allocated GPU memory | 7.633773 GiB |
| Complete input token range | 255–868 |
| Minimum original-position A/B mass | 0.0000019711345752524745 |

The four normalized rows are the previously affected synthetic-01 and synthetic-03 original-body conditions under both mappings. **A/B probabilities remain the original first-generated-token probabilities**, not probabilities recomputed after a channel wrapper. Low code mass remains visible and this parsing fix does not establish confident first-token classification or research qualification.

Fresh validation used nine passing unit tests and the complete 42-call native smoke. The tests reject nonempty reasoning, malformed/repeated wrappers, explanatory text, punctuation, missing answers, and the same wrapper on non-Gemma models. An independent eight-condition comparison confirmed identical input hashes and generated token IDs, with first-token scores agreeing within 1e-5 before and after the parser change.

A separate 16-call boundary diagnostic reproduced four legacy-parser INVALIDs in eight native conditions; appending a newline produced eight INVALIDs in eight conditions. The newline adjustment was discarded. The failed output-reminder adjustment is also not used. The default native prompt is preserved.

- Corrected source revision: `d9a7d62a5629836a7643400a00a689ec8546e87c`.
- Gemma checkpoint: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- Corrected results SHA256: `310fcf6a4527b8d21e504bde63d638eae47c813afca3517743f0c8c62d0e7342`.
- NF4 double quantization / FP16 on Tesla T4; same pinned inference-package versions as above.
- Call accounting: original and reminder runs 126; this user-requested repair investigation 16 + 42 = 58; **184 cumulative GPU calls**. The added repair work is outside the original 160-call initial smoke allowance. No development screen or final scientific evaluation was run.
- Corrected Colab bundle: `gemma-corrected-smoke.zip`, containing raw rows, state vectors, configuration, audit, comparison, source archive, environment and logs. Download it from the notebook before its temporary runtime expires.

BF16 execution, actual development-data gates and the remaining scientific controls are still pending. See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for the pipeline explanation.

## Documentation and integration checks — 12 October 2026

At source revision `d71be31d99f1396409889e4da670e58bc62be339`, the documentation update passed all nine offline protocol tests and generated 80 planned inputs from the ten synthetic fixtures with all four variants and both mappings. The notebook JSON, ordinary Python cells and local documentation links were also checked. No GPU inference was rerun.

The README now documents startup, CLI defaults, input/output contracts, recorded evidence and pending research gates. The Colab checkout and repository links use `main` for the integrated version. SETUP.md records owner-authorized integration and the still-pending mentor review and validation work. GitHub reported no attached CI checks before integration; offline checks above are not a CI or scientific qualification claim.
