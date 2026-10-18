# Exploratory pilot results

Model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`

**Not final evaluation.** Labels, text completeness and request-family grouping remain under review. No causal intervention was run. Rates below average the two mappings within each saved record; they are not independent calls or confirmed independent families. No confidence intervals are claimed.

| Variant | False-safe % | Unsafe recall % | Safe specificity % | Safe false alarms % | Invalid % |
|---|---:|---:|---:|---:|---:|
| B | 6.0 | 94.0 | 70.0 | 30.0 | 0.0 |
| R | 10.0 | 90.0 | 72.0 | 28.0 | 0.0 |
| N | 6.0 | 94.0 | 72.0 | 28.0 | 0.0 |
| R2 | 12.0 | 88.0 | 78.0 | 22.0 | 0.0 |

Refusal cue effect: +4.0 percentage points.
Unsafe-to-safe flips under both mappings: 0.

See summary.json for per-mapping counts, invalid-output bounds, checkpoint/software provenance, artifact hashes and state-audit status. Raw text, identifiers, outputs and tensors are not included in this aggregate report.
