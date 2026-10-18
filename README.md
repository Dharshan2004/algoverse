# Algoverse: refusal wording and safety judgments

Inference tooling for a study of how refusal wording changes an LLM judge's safety decisions when the underlying response stays the same. The runner judges saved prompt–response pairs, records output probabilities, and captures internal activations for the team's interpretability work.

**Current status:** [50-record exploratory pilot completed](reports/pilot-2026-10-17/README.md) on both models using native BF16 on an L4: 800 judgments plus four identity checks, with full saved-artifact audits passed. Source-data qualification and scientific development gates remain open. These results do not establish benchmark accuracy, a causal refusal mechanism, or readiness for final evaluation.

[Open in Google Colab](https://colab.research.google.com/github/Dharshan2004/algoverse/blob/main/notebooks/step_zero.ipynb) · [Research protocol](https://refusal-wording-safety-judgments.dharshanlab.chatgpt.site/) · [Pipeline walkthrough](INFERENCE_GUIDE.md) · [Test evidence](TESTING.md) · [Team handoff](SETUP.md)

[Animated code and results walkthrough](media/pilot-explainer/README.md) · [Notebook and run history](notebooks/README.md)

## What this repository provides

- Two model adapters: Ministral 3 8B and Gemma 4 12B, using native chat templates and 4-bit NF4 weights.
- Four response conditions: original body (`B`), refusal prefix (`R`), neutral prefix (`N`), and alternate refusal prefix (`R2`). The original body is preserved byte-for-byte.
- Both output mappings: A = SAFE / B = UNSAFE, and A = UNSAFE / B = SAFE.
- Raw generated text and token IDs, strict verdict parsing, full-vocabulary first-token A/B log probabilities, and a semantic score margin.
- Six decoder activation vectors at the last input token, with identity checks and an independent saved-artifact audit.
- A Python intervention interface for subsequent interpretability work. Nonidentity interventions still require validation.

The judge evaluates an existing response; it is not asked to fulfil the saved request. No model training happens in this runner.

## Quick start

For Sabika’s source-data pipeline, see [Dataset preparation](DATASET_PREPARATION.md).

For actual development-data runs, use the [exploratory pilot workflow](PILOT.md) and [open its clean notebook in Colab](https://colab.research.google.com/github/Dharshan2004/algoverse/blob/codex/pilot-dataset-integration/notebooks/pilot.ipynb). It runs all four variants on both models and provides a separate artifact audit and aggregate report. Dry runs only construct inputs; they do not produce model judgments.

### Google Colab

1. Open the notebook above and select a GPU runtime.
2. Run the checkout and dependency cells. If model access requires authentication, use Colab Secrets with the name `HF_TOKEN` and complete the provider's access agreement.
3. Run the unit tests and offline input plan.
4. Run `MODEL = "ministral"`, inspect the audit and outputs, and download the artifact archive.
5. Start a fresh runtime for `MODEL = "gemma"` and repeat.

The notebook defaults to **FP16 on a T4 for synthetic compatibility testing**. The command-line runner defaults to **BF16**, which requires a GPU with native BF16 support. Set the notebook's `COMPUTE_DTYPE` to `"bfloat16"` for that configuration; its successful execution would still need the research qualification checks below.

Each default model run uses ten synthetic records × two variants × two mappings = **40 judging calls**, plus **two identity-check calls**. Use a new output directory for every run. The runner refuses to overwrite an existing directory.

### Local checks without a GPU

These commands need Python 3 and the analysis dependencies below. They do not download models or make inference calls.

```sh
git clone https://github.com/Dharshan2004/algoverse.git
cd algoverse
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python inference.py --dry-run --variants B R N R2 --out runs/planned
```

The dry run validates the ten fixtures and writes **80 planned inputs**. It checks prompt construction, not GPU execution or tokenizer length limits.

### GPU execution

Use the Colab notebook for dependency installation and environment capture. It installs the inference packages and saves an environment lock; the recorded test versions and checkpoint revisions are in [TESTING.md](TESTING.md). Use the saved lock and checkpoint revision when reproducing a run, rather than assuming a fresh dependency installation is identical.

For a synthetic T4 compatibility run:

```sh
python inference.py --model ministral --compute-dtype float16 --out runs/ministral-t4
python audit_smoke.py runs/ministral-t4
```

For a BF16 smoke run on a compatible CUDA GPU:

```sh
python inference.py --model gemma --compute-dtype bfloat16 --out runs/gemma-bf16
python audit_smoke.py runs/gemma-bf16
```

`audit_smoke.py` audits the default ten-record, B/R, two-mapping synthetic run. Custom datasets and expanded variant runs need corresponding review; they are not covered by that default audit.

## Configuration

| Option | Default | Purpose |
|---|---|---|
| `--model` | `gemma` | Choose `gemma` or `ministral`. |
| `--compute-dtype` | `bfloat16` | Compute precision; `float16` only accepts synthetic fixtures. |
| `--revision` | `main` | Hugging Face model revision, resolved to an immutable SHA at startup. |
| `--input` | `examples.jsonl` | JSONL input records. |
| `--out` | `runs/smoke` | A new directory for this run's artifacts. |
| `--variants` | `B R` | Any selection of `B R N R2`. |
| `--baseline` | `native` | `native`, `content`, or `strict` judging instruction. |
| `--dry-run` | Off | Write planned prompts without loading a model. |

`--format-reminder` is a synthetic-only diagnostic from an unsuccessful formatting experiment. It is not the adopted research prompt. Expanding variants or baselines increases inference calls: all four variants on both models require 164 calls including identity checks, exceeding the original 160-call initial smoke allowance. Plan additional runs explicitly.

## Input and output contract

Each input line is a JSON object with these required fields:

| Field | Meaning |
|---|---|
| `source_id` | Unique record ID. |
| `cluster_id` | ID connecting related examples. |
| `original_prompt` | Saved user request. |
| `original_response` | Saved assistant response, preserved across variants. |
| `study_partition` | `synthetic` or `development`; final-test records are rejected. |
| `mapped_unsafe` | Boolean label metadata, never inserted into the judge prompt. |
| `label_provenance` | Must be `human` for development records. |

See [examples.jsonl](examples.jsonl) for the schema. Its labels are illustrative fixtures, not human-validated benchmark labels. Development data additionally needs the source-policy, provenance, split, and cross-model length checks described in [SETUP.md](SETUP.md).

| Artifact | Contents |
|---|---|
| `planned_inputs.jsonl` | Constructed prompts, only for dry runs. |
| `run_config.json` | Resolved model revision, environment, precision, prompt and run metadata. |
| `identity_check.json` | Native versus identity-intervention agreement. |
| `results.jsonl` | One row per condition with raw output, verdict, scores, hashes, timing and memory. |
| `states_00000.pt`, etc. | One dictionary of six CPU activation vectors per judging row. |
| `artifact_audit.json` | Results of running the separate synthetic smoke audit. |

Keep raw datasets, generated outputs, model weights, tokens and environment secrets out of Git. The notebook packages artifacts for private download; download them before the temporary runtime expires.

## What has been validated

The recorded 11 October tests used Tesla T4, NF4 weights and FP16 compute:

| Recorded result | Ministral 3 8B | Gemma 4 12B, corrected parser |
|---|---:|---:|
| Judging rows plus identity calls | 40 + 2 | 40 + 2 |
| Normalized INVALID verdicts | 0/40 | 0/40 |
| Identity and artifact checks | Pass | Pass |
| Peak allocated GPU memory | 6.253 GiB | 7.634 GiB |

Gemma's correction removes only a recognized empty native channel wrapper before exact A/B parsing. Four raw wrappers and length-cap flags remain recorded. Their low first-token A/B probability mass remains visible: valid parsing does not establish confident classification. Nine protocol regression tests passed in the recorded corrected run.

[TESTING.md](TESTING.md) preserves the chronological evidence, failed diagnostics, exact revisions and result hashes. Those GPU tests are historical evidence, not a fresh GPU rerun accompanying every documentation change.

## Remaining work before research evaluation

- Validate the BF16 configuration and actual development examples, including inputs near the 4,096-token limit.
- Reconcile the rubric with source annotation rules; audit human labels, provenance and cluster splits.
- Check complete-input eligibility with both tokenizers and the longest approved prompt/variant.
- Validate nonidentity edits, model-specific refusal directions and behavioral controls.
- Integrate evaluation metrics, remaining controls and statistical analysis; satisfy the protocol's power and freeze gates before final evaluation.

The runner provides an inference foundation. Dataset acquisition, probe training, Shieldstral scoring, literal-label sequence scoring, the complete control matrix and final-run orchestration are outside this initial implementation.

## Repository map and collaboration

| File | Responsibility |
|---|---|
| [protocol.py](protocol.py) | Input validation, variants, judging messages and parsing. |
| [inference.py](inference.py) | Model loading, execution, activation capture and output writing. |
| [audit_smoke.py](audit_smoke.py) | Independent audit of the default synthetic smoke artifacts. |
| [notebooks/step_zero.ipynb](notebooks/step_zero.ipynb) | Colab setup, execution and artifact download. |
| [tests/test_protocol.py](tests/test_protocol.py) | Offline protocol regression checks. |
| [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) | Detailed pipeline and score interpretation. |
| [SETUP.md](SETUP.md) | Assignment, ownership, integration checklist and pending work. |
| [TESTING.md](TESTING.md) | Recorded test results and reproduction details. |

Use separate branches and pull requests with a short description, validation evidence and remaining limitations. Priyadharshan owns inference setup; Sabika owns dataset preparation; Mehrage owns interpretability; Hemanya and Laasya own evaluation metrics. Pratham provides research guidance and review. See the handoff for the October 8 assignment and the distinction between repository integration and scientific approval.
