# A/B-constrained output protocol v1

Base: PR #3, immutable commit `03b32cb2ac6ae6aabf67a6b2da7632c3de852834`. Additive experiment version: `ab_constrained_v1`. The original pilot and its ten INVALIDs are immutable comparison evidence.

The constrained arm uses the same Gemma checkpoint (`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`), native prompt/template, BF16 and NF4 double quantization, response bodies and A/B mappings. A logits processor masks every next-token candidate except verified single-token A and B. Greedy generation stops after exactly one token, so EOS is not required. Ties follow the generation argmax (lower vocabulary index). No prose extraction or post-hoc relabeling occurs. Raw full-vocabulary first-position scores and unnormalized A/B mass remain saved. One token is a complete fixed-code answer, separately recorded from an unexpected native generation-cap stop.

Four arms: 50 development records × 4 variants × 2 mappings under native_v1 (4-token budget) and ab_constrained_v1 (1-token budget); 8 clean synthetic controls × 4 variants × 2 mappings under both protocols. Total: 928 judgments plus two identity calls per arm = 936 calls. Arm order is fixed: pilot native, pilot constrained, controls native, controls constrained; no sampling or tuning on results.

Controls are authored policy examples, four safe and four unsafe, each with a different request family and complete standalone text. They are simple format/classification checks, not independently annotated benchmark gold. Dataset source labels remain unresolved. Prompt construction never includes labels or control rationale.

Audits validate every condition, hashes, input/body preservation, parser and probability arithmetic, six finite state vectors, identity equality, raw constrained token identity and agreement with raw A/B argmax. Paired comparisons verify matching prompt/token hashes and model revisions. Prefill states and original logits are compared across protocols. Audits do not assert dataset scientific qualification.

Report format validity separately from source-label accuracy (INVALID counts as incorrect), valid-only accuracy with its changing denominator, unsafe false-safe rate/recall, safe specificity/false alarms, per-mapping behavior, original invalid transitions, changes among previously valid answers and native rerun drift. No confidence intervals or causal claims on unreviewed families.
