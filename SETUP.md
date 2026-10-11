# Setup handoff — 11 October 2026

## What you were assigned

October 8 Circleback notes: Priyadharshan sets up inference scripts for two models, covers proposal scenarios, uses hypothetical examples to check the pipeline, prefers Google Colab, and pushes scripts to GitHub in about five days (approximately October 13). Next meeting: October 15; no Sunday meeting.

Earlier October 4 repo/invitation task was already acted on: your [Slack update](https://823algoversefall26.slack.com/archives/C0C00BWEQTZ/p1791118400705729) says invites were sent. The repo existed but was empty when this work began. No duplicate invites were sent.

## Source links

- [Hosted research protocol](https://refusal-wording-safety-judgments.dharshanlab.chatgpt.site/)
- [Pratham's original HTML attachment message](https://823algoversefall26.slack.com/archives/C0C00BWEQTZ/p1791184015466599) — final_research_proposal.html
- [Your hosted-site message](https://823algoversefall26.slack.com/archives/C0C00BWEQTZ/p1791191378034719)
- [October 8 task-note follow-up](https://823algoversefall26.slack.com/archives/C0C00BWEQTZ/p1791432365856199)
- [Gemma checkpoint](https://huggingface.co/google/gemma-4-12B-it)
- [Ministral checkpoint](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16)
- [Official Ministral Transformers integration](https://huggingface.co/docs/transformers/model_doc/ministral3)
- [Official Gemma Unified Transformers integration](https://huggingface.co/docs/transformers/model_doc/gemma4_unified)

## Decisions and blockers

1. Meeting notes say "two Gemma models"; current hosted protocol names Ministral 3 8B and Gemma 4 12B. The setup follows the exact hosted checkpoints; confirm this discrepancy with Pratham before spending GPU time. The original HTML attachment has been located, not downloaded/compared.
2. GPU tests have not run in this local environment. Native model loading, template behavior, output tokens, hooks, NF4 memory and finite scores must pass on Colab. A dependency/API failure stops the run; it is not silently replaced with a different model or precision.
3. Synthetic fixtures establish no safety accuracy or causal result. Include actual short/long dataset cases near the 4096-token boundary in the GPU execution audit when eligible source data arrives.
4. Before development screening, verify each complete input against both tokenizers and the longest approved prompt/variant, source policy compatibility, cluster splits and provenance. This runner checks only the model currently loaded.
5. Check all saved smoke outputs manually; do not infer success from script completion. Preserve environment locks, model SHAs and the actual measured memory for review.
6. Final evaluation remains blocked until the site's audit, generation-validation, power and freeze gates are met.

## Ownership from October 8

| Person | Workstream |
|---|---|
| Priyadharshan | Colab inference scripts and hypothetical pipeline tests |
| Mehrage | Interpretability probes / phase-two pipeline |
| Hemanya and Laasya | Evaluation metrics |
| Sabika | Initial 20–50 dataset samples and prompt categories |
| Pratham | Written instructions, benchmark reference and review |

Circleback calls the dataset "JailBench"; the hosted protocol specifically uses JAILJUDGE ID human-labeled records. Preserve the source identity rather than downloading a similarly named benchmark by guesswork.

## Review checklist

- Run local protocol tests and inspect 80 planned inputs.
- Execute the default Colab B/R smoke on one model, then a fresh runtime for the other.
- Verify raw A/B outputs, INVALID handling, code IDs, six state files per run (stored as one dictionary), identity equality and memory.
- Save the generated run directory privately and send only a concise result summary / permitted artifacts for review.
- Coordinate the normalized input schema with Sabika and the `Runner.run(..., edit=...)` interface with Mehrage.

Training vectors must be model-specific and come from independent approved partitions. A donor must match the response family, model revision, output mapping, rubric, layer and token position. The low-level operator cannot establish those provenance facts by itself; callers must validate them before edited research runs.
