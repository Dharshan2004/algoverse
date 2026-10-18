# Exploratory pilot results

Model: `google/gemma-4-12B-it`

**Not final evaluation.** Labels, text completeness and request-family grouping remain under review. No causal intervention was run. Rates below average the two mappings within each saved record; they are not independent calls or confirmed independent families. No confidence intervals are claimed.

| Variant | False-safe % | Unsafe recall % | Safe specificity % | Safe false alarms % | Invalid % |
|---|---:|---:|---:|---:|---:|
| B | 22.0 | 78.0 | 88.0 | 0.0 | 6.0 |
| R | 28.0 | 72.0 | 98.0 | 0.0 | 1.0 |
| N | 24.0 | 76.0 | 96.0 | 0.0 | 2.0 |
| R2 | 28.0 | 72.0 | 98.0 | 0.0 | 1.0 |

Refusal cue effect: +6.0 percentage points.
Unsafe-to-safe flips under both mappings: 2.

See summary.json for per-mapping counts, invalid-output bounds, checkpoint/software provenance, artifact hashes and state-audit status. Raw text, identifiers, outputs and tensors are not included in this aggregate report.

Follow-up: [investigation of all ten INVALID outputs](../../gemma-invalid-2026-10-18/README.md). Original results are unchanged.
