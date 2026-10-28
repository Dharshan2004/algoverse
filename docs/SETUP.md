# Setup handoff — 11 October 2026

## What was assigned (October 8)

Circleback notes from October 8 scoped: two-model inference scripts, proposal scenarios, hypothetical pipeline checks, Google Colab as the primary runtime, and a GitHub handoff within about five days (approximately October 13). Next meeting: October 15; no Sunday meeting.

Earlier October 4 repo/invitation task was already acted on: [Slack update](https://823algoversefall26.slack.com/archives/C0C00BWEQTZ/p1791118400705729) confirms invites were sent. The repo existed but was empty when this work began. No duplicate invites were sent.

## Source links

- [Hosted research protocol](https://refusal-wording-safety-judgments.dharshanlab.chatgpt.site/)
- [Original research proposal attachment (Slack)](https://823algoversefall26.slack.com/archives/C0C00BWEQTZ/p1791184015466599) — final_research_proposal.html
- [Hosted-site message](https://823algoversefall26.slack.com/archives/C0C00BWEQTZ/p1791191378034719)
- [October 8 task-note follow-up](https://823algoversefall26.slack.com/archives/C0C00BWEQTZ/p1791432365856199)
- [Gemma checkpoint](https://huggingface.co/google/gemma-4-12B-it)
- [Ministral checkpoint](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16)
- [Official Ministral Transformers integration](https://huggingface.co/docs/transformers/model_doc/ministral3)
- [Official Gemma Unified Transformers integration](https://huggingface.co/docs/transformers/model_doc/gemma4_unified)

## Decisions and blockers

1. Resolved 11 October: follow the hosted research site as the authoritative protocol. Use `mistralai/Ministral-3-8B-Instruct-2512-BF16` and `google/gemma-4-12B-it`. The earlier meeting-note phrase "two Gemma models" does not require further confirmation. Ministral is the first primary-model candidate; screen both models and apply all protocol qualification gates.
2. Both user-approved Colab T4/FP16 synthetic runs completed: 42 calls/model, identity and saved-artifact audits passed. Ministral had 0/40 INVALID outputs; Gemma initially had 4/40 legacy-parser INVALIDs. A fresh 42-call run after correcting empty native channel parsing passed with 0/40 INVALID; raw wrappers and first-token scores remain recorded. Nine regression tests passed. See [TESTING.md](TESTING.md). BF16 protocol execution and actual development qualification remain pending; a synthetic format pass does not satisfy them.
3. Synthetic fixtures establish no safety accuracy or causal result. Include actual short and long complete development cases in GPU execution audits when eligible source data arrives. **Do not exclude records solely because templated input exceeds 4,096 tokens** (team decision, October 2026); measure and report token lengths instead.
4. Before development screening, tokenize each complete input on both models for length reporting and eligibility review, plus source policy compatibility, cluster splits and provenance. This runner checks only the model currently loaded and does not apply a hard 4,096-token intake gate.
5. Check all saved smoke outputs manually; do not infer success from script completion. Preserve environment locks, model SHAs and the actual measured memory for review.
6. Final evaluation remains blocked until the site's audit, generation-validation, power and freeze gates are met.

## Workstreams (October 8 assignment)

| Workstream | Scope |
|---|---|
| Inference runner | Colab scripts and hypothetical pipeline tests |
| Interpretability | Probes / phase-two pipeline |
| Evaluation metrics | Team metrics module and controls |
| Dataset intake | Initial 20–50 samples and prompt categories |
| Research protocol | Written instructions, benchmark reference and review |

Circleback calls the dataset "JailBench"; the hosted protocol specifically uses JAILJUDGE ID human-labeled records. Preserve the source identity rather than downloading a similarly named benchmark by guesswork.

## Review checklist

- Run local protocol tests and inspect 80 planned inputs.
- Execute the default Colab B/R smoke on one model, then a fresh runtime for the other.
- Verify raw A/B outputs, INVALID handling, code IDs, six state files per run (stored as one dictionary), identity equality and memory.
- Save the generated run directory privately and send only a concise result summary / permitted artifacts for review.
- Coordinate the normalized input schema (dataset intake) with the `Runner.run(..., edit=...)` interface (directions package).

Training vectors must be model-specific and come from independent approved partitions. A donor must match the response family, model revision, output mapping, rubric, layer and token position. The low-level operator cannot establish those provenance facts by itself; callers must validate them before edited research runs.

## Repository integration decision — 12 October 2026

The repository owner authorized merging this initial inference setup after checking the October 8 Circleback assignment. That assignment requested two-model inference scripts, proposal scenarios, hypothetical examples and a GitHub handoff for research review. Integration is scoped to that foundation; it does not mark the research validation gates complete.

No submitted GitHub review was present when this handoff was prepared. Mentor research review remains pending; owner authorization to integrate is not mentor approval. Existing GPU evidence is documented in TESTING.md; no new GPU run is claimed for this documentation update. The README and Colab launcher target `main` for use after integration.

### Follow-up checklist

- [ ] Research protocol: review the inference setup and its assumptions.
- [ ] Inference runner: validate native BF16 execution and long complete development inputs.
- [ ] Dataset intake + inference: agree on input schema, source rubric, human-label provenance, cluster splits and cross-model length reporting (no 4,096-token exclusion gate).
- [ ] Inference + directions: validate nonidentity operators and model-specific direction/donor provenance before intervention experiments.
- [ ] Metrics: integrate evaluation metrics and required controls.
- [ ] Team: complete protocol qualification, power and freeze checks before final evaluation.

These are open handoff items, not newly completed experiments or confirmations from individual contributors.
