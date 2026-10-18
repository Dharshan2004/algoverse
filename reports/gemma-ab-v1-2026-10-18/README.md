# Gemma constrained-output comparison — 18 October 2026

**Completed experiment, exploratory classification evidence.** Output protocol `ab_constrained_v1` was run on a Colab NVIDIA L4 with BF16/NF4, based on PR #3 commit `03b32cb2ac6ae6aabf67a6b2da7632c3de852834` plus versioned implementation `20450d8`. Original pilot files and their ten INVALIDs were preserved.

The experiment ran the original and constrained protocols on all 50 development records, four variants and both mappings, plus eight clean synthetic controls under the same conditions: **928 judgments and eight identity calls**. All four arm audits passed, including every saved activation file.

## Formatting and classification are different outcomes

Validity means a parseable code. Accuracy here means agreement with supplied labels, which remain unqualified for the dataset; controls have authored synthetic labels. INVALID counts as incorrect in the all-judgment denominator. Valid-only accuracy has a changing denominator and must not be used alone to claim improvement.

| Cohort / protocol | Valid output | Correct / all | Accuracy / all | Accuracy / valid only |
|---|---:|---:|---:|---:|
| Original pilot | 390/400 (97.50%) | 339/400 | 84.75% | 86.92% |
| pilot_native_v1 | 390/400 (97.50%) | 339/400 | 84.75% | 86.92% |
| pilot_ab_constrained_v1 | 400/400 (100.00%) | 340/400 | 85.00% | 85.00% |
| controls_native_v1 | 64/64 (100.00%) | 63/64 | 98.44% | 98.44% |
| controls_ab_constrained_v1 | 64/64 (100.00%) | 60/64 | 93.75% | 93.75% |

**Interpretation:** this protocol fixes formatting in the tested run, but does not establish a classification-quality improvement. The net dataset gain is one correct judgment, while clean-control accuracy falls by three judgments. Keep this as a separately reported experimental arm; do not replace the original pilot or call its ten INVALIDs solved semantically.

Three previously correct dataset judgments became wrong. On controls, the added errors are false alarms on a privacy refusal (R and N, A=UNSAFE) and fair-hiring advice (R2, A=UNSAFE). The native error on a response encouraging unauthorized medical-record disclosure (R, A=UNSAFE) remains. All four constrained control errors occur under A=UNSAFE; inspect mapping sensitivity before considering a new protocol version.

## What happened to the original ten INVALIDs

Of the ten originally invalid judgments, **4 became source-label-correct and 6 became source-label-wrong**. The constrained protocol also changed **3 previously valid verdicts**. A valid answer therefore cannot be equated with a correct answer.

Original → constrained transitions: `{"INVALID->SAFE": 4, "INVALID->UNSAFE": 6, "SAFE->SAFE": 240, "SAFE->UNSAFE": 1, "UNSAFE->SAFE": 2, "UNSAFE->UNSAFE": 147}`.

Original → native rerun transitions: `{"INVALID->INVALID": 10, "SAFE->SAFE": 241, "UNSAFE->UNSAFE": 149}`.

## Dataset classification by condition

Rates below average both mappings on the same 25 source-unsafe and 25 source-safe records. Lower false-safe/false-alarm rates are better; higher recall/specificity are better. These are 50 records, not 400 independent examples.

| Variant | Native false-safe | Constrained false-safe | Native unsafe recall | Constrained unsafe recall | Native safe specificity | Constrained safe specificity | Native safe false alarm | Constrained safe false alarm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 22.0% | 22.0% | 78.0% | 78.0% | 88.0% | 92.0% | 0.0% | 8.0% |
| R | 28.0% | 28.0% | 72.0% | 72.0% | 98.0% | 100.0% | 0.0% | 0.0% |
| N | 24.0% | 24.0% | 76.0% | 76.0% | 96.0% | 96.0% | 0.0% | 4.0% |
| R2 | 28.0% | 32.0% | 72.0% | 68.0% | 98.0% | 98.0% | 0.0% | 2.0% |

### Mapping sensitivity

| Variant | Native FSR: A=SAFE | Native FSR: A=UNSAFE | Constrained FSR: A=SAFE | Constrained FSR: A=UNSAFE |
|---|---:|---:|---:|---:|
| B | 20.0% | 24.0% | 20.0% | 24.0% |
| R | 28.0% | 28.0% | 28.0% | 28.0% |
| N | 24.0% | 24.0% | 24.0% | 24.0% |
| R2 | 28.0% | 28.0% | 36.0% | 28.0% |

R-minus-B false-safe change: native +6.0 points; constrained +6.0 points. Robust B→R UNSAFE-to-SAFE flips under both mappings: native 2, constrained 2.

## Audit and scope

Both protocols use identical prompts, tokenized input hashes and checkpoint revision. The constrained protocol masks all generated token choices except A/B and stops after one complete answer token. The native protocol retains its four-token budget. The saved probability scores are the original full-vocabulary scores, not normalized A/B probabilities.

- pilot: identical saved prefill activations in 400/400 pairs; maximum original A/B log-probability difference 0.
- controls: identical saved prefill activations in 64/64 pairs; maximum original A/B log-probability difference 0.

Every constrained raw output was checked to be a single actual generated A/B token equal to the raw-score A/B argmax. All identity checks passed. Token budget completion is explicitly recorded as `fixed_code_complete`; it is not mislabeled as a failed native truncation.

Clean controls are eight complete, authored policy examples (four safe/four unsafe), not independently annotated benchmark gold. Simple control success cannot establish general accuracy. The development dataset still has source-label, text-boundary and family-grouping review items. No significance, generalization or causal-mechanism claim is made. No original verdict was patched and no failed cases were selectively rerun.

See [summary.json](summary.json) for all four arm audits, checkpoint/environment metadata, output hashes, complete per-mapping counts, low A/B code-mass counts and comparisons with the preserved original pilot. Source/overlay and artifact preservation checks are in [provenance.json](provenance.json).

Execution: [sanitized executed notebook](../../notebooks/history/2026-10-18/gemma_ab_v1.executed.sanitized.ipynb). Exact executed notebook, raw rows and tensors remain under ignored `runs/gemma-ab-v1-2026-10-18/`. The clean [launch notebook](../../notebooks/gemma_ab_v1.ipynb) and [reproduction instructions](../../AB_REPRODUCTION.md) are tracked. No remote push is included.
