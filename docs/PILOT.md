# Running and reporting the exploratory pilot

The pilot executes real safety judgments on saved development records. It asks whether refusal wording changes verdicts and whether the inference/metrics pipeline works on actual data. It does not establish a causal mechanism, certify source labels, or replace final evaluation.

The 17 October preview has 50 records (25 source-unsafe / 25 source-safe). B, R, N and R2 under both mappings produce 400 judgments per model, plus two identity calls. Both models total 804 calls. Incomplete source text, proposed shared-request families, and source-label reconciliation remain review items; preserve the original input and report these limitations with results.

## Execute

Use a native BF16 CUDA GPU, such as the L4 used for this pilot. The existing T4/FP16 synthetic compatibility mode does not accept development data. Install the recorded dependencies and keep raw data under ignored `data/` or `runs/` directories.

```sh
python pilot.py --input data/candidate_inputs.jsonl --out runs/pilot-2026-10-17
```

The launcher pins both checkpoint revisions, invokes the existing `inference.py` in separate processes, streams logs and records input/runner hashes. It refuses to overwrite a run directory. A failed model leaves partial artifacts and marks the manifest as failed; a successful inference run is still audit-pending. There is no automatic resume or silent retry. `--models ministral` limits a run to one model.

```sh
python pilot.py --input data/candidate_inputs.jsonl --out runs/planned-pilot --dry-run
```

This second command prepares inputs only; it never runs a model. Do not report its outputs as pilot results.

The clean `notebooks/pilot.ipynb` provides Colab setup, private input upload, real execution and download. Its remote checkout is pinned to a recorded integration commit. It contains no embedded dataset or outputs. The original execution notebook used a local source bundle uploaded to Colab; its exact source copies are retained in the private run archive.

## Audit and summarize

```sh
python pilot_report.py runs/pilot-2026-10-17/ministral \
  --input data/candidate_inputs.jsonl --check-states \
  --out reports/pilot-2026-10-17/ministral
python pilot_report.py runs/pilot-2026-10-17/gemma \
  --input data/candidate_inputs.jsonl --check-states \
  --out reports/pilot-2026-10-17/gemma
```

The audit rejects missing/duplicate conditions, changed source fields or variants, mismatched mappings/prompts/checkpoints, invalid score arithmetic and missing state files. `--check-states` additionally loads each tensor with `weights_only=True` and checks six finite vectors of consistent dimensions. Without that option the report explicitly says tensor contents were not checked. The original `audit_smoke.py` is specific to ten synthetic records and must not be used for this pilot.

The adapter supplies normalized `parsed_text` to the team's `metrics.py` parser; raw native channel/EOS text remains in the private output. Rates retain INVALID in the denominator. Summaries expose per-mapping counts, mapping-averaged false-safe rates, recall/specificity, false alarms, invalids, cue effects and robust flips. An invalid verdict is neither correct nor a rescued/created SAFE flip. The false-safe sensitivity upper bound adds unsafe-invalid percentage to the observed false-safe percentage.

These are descriptive **record-level** rates. Mapping calls are paired, not independent examples. Cluster assignments remain provisional, so the pilot report makes no bootstrap precision or final power claim. No repair metric is reported because this behavioral pilot contains no nonidentity intervention.

## Results intended for Git

Keep reviewed aggregate `summary.json` and `README.md` under `reports/<run>/<model>/`. They contain counts, configuration, hashes and limitations, not source IDs, saved responses or model output text. The private archive contains raw input, runner source, qualification flags, logs, environment, results and activations. Do not commit the execution notebook with its embedded input or output cells.

Before a later push, inspect the staged file list and report for raw text or tokens. Raw artifacts are ignored; `.gitignore` is a backstop rather than a substitute for inspecting what is staged. Dataset access/redistribution permission and scientific approval are separate from a successful runtime.
