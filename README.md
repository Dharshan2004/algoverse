# Algoverse: refusal wording and safety judgments

Inference tooling for a study of how refusal wording changes an LLM judge's safety decisions when the underlying response stays the same. The runner judges saved prompt–response pairs, records output probabilities, and captures internal activations for the team's interpretability work.

**Current status:** [50-record exploratory pilot completed](reports/pilot-2026-10-17/README.md) on both models using native BF16 on an L4: 800 judgments plus four identity checks, with full saved-artifact audits passed. Source-data qualification and scientific development gates remain open. These results do not establish benchmark accuracy, a causal refusal mechanism, or readiness for final evaluation.

**18 October follow-up:** [Gemma A/B-constrained comparison](reports/gemma-ab-v1-2026-10-18/README.md) completed 928 judgments plus eight identity checks. Output validity improved to 100%, but dataset label agreement changed only 84.75% → 85.00%, and clean-control accuracy fell 98.44% → 93.75%. The original pilot and ten INVALIDs remain intact. See the [frozen protocol](docs/AB_PROTOCOL.md) and [reproduction guide](docs/AB_REPRODUCTION.md).

[Open in Google Colab](https://colab.research.google.com/github/Dharshan2004/algoverse/blob/main/notebooks/step_zero.ipynb) · [Research protocol](https://refusal-wording-safety-judgments.dharshanlab.chatgpt.site/) · [Pipeline walkthrough](docs/INFERENCE_GUIDE.md) · [Test evidence](docs/TESTING.md) · [Team handoff](docs/SETUP.md)

[Animated code and results walkthrough](media/pilot-explainer/README.md) · [Notebook and run history](notebooks/README.md)

**Case-review update:** [Both robust Gemma flips need source/policy adjudication](reports/gemma-case-review-2026-10-18/README.md). One is an unfinished request continuation and the other fictional narrative. Setting them aside in a post-hoc sensitivity check reverses the net refusal effect; scaling remains on hold pending original-source reconciliation and human review.

## Repository layout

```
algoverse/
  README.md
  pyproject.toml              # pip install -e .  →  import refusal_cues
  requirements.txt            # GPU inference stack (unpinned; pin for reproduction)
  requirements-analysis.txt   # numpy/pandas for offline tests and data prep
  configs/                    # YAML stubs mirroring CLI defaults (pilot vs smoke)
  data/
    manifests/examples.jsonl  # small synthetic fixtures (tracked)
    controls/ab_v1.jsonl      # authored clean controls for A/B protocol
    raw/ processed/           # local HF exports (gitignored; .gitkeep only)
  refusal_cues/               # importable package (protocol, inference, metrics, …)
    directions/               # probe / direction loaders (placeholder)
    diagnostics/              # offline replay on saved runs
  scripts/                    # thin CLIs → refusal_cues
  notebooks/                  # Colab workflows
  docs/                       # SETUP, PILOT, INFERENCE_GUIDE, TESTING, …
  reports/                    # historical audited summaries (tracked)
  results/ artifacts/         # new bulky outputs (gitignored)
  tests/                      # unittest + pytest (metrics)
```

**Functional areas**

| Area | Primary paths |
|---|---|
| Inference runner, Colab workflows, artifact audits | `refusal_cues/inference.py`, `scripts/run_inference.py`, `notebooks/` |
| Dataset intake (CSV preview + HF adapter) | `scripts/prepare_preview.py`, `scripts/adapt_hf_dataset.py`, `docs/DATASET_PREPARATION.md` |
| Refusal directions / probes | `refusal_cues/directions/` |
| Evaluation metrics | `refusal_cues/metrics.py`, `tests/test_metrics.py` |
| Research protocol (hosted site) | [Protocol](https://refusal-wording-safety-judgments.dharshanlab.chatgpt.site/), `docs/SETUP.md` |

## Install

**Analysis-only (no GPU, no model download):**

```sh
git clone https://github.com/Dharshan2004/algoverse.git
cd algoverse
python3 -m pip install -e ".[analysis]"
python3 -m unittest discover -s tests -v
python3 -m pytest tests/test_metrics.py -q
```

**GPU inference:** `python3 -m pip install -r requirements.txt` and `python3 -m pip install -e ".[inference]"` (or the Colab cells in `notebooks/step_zero.ipynb`). Both declare `mistral-common`, which Ministral needs for `MistralCommonBackend`. `pip install -e ".[analysis]"` does not install it.

## Quick commands

| Task | Command |
|---|---|
| Dry-run input plan (80 calls on fixtures) | `python3 scripts/run_inference.py --dry-run --variants B R N R2 --out runs/planned` |
| Exploratory pilot launcher | `python3 scripts/run_pilot.py --input data/processed/candidate_inputs.jsonl --out runs/pilot-dev` |
| Pilot aggregate report | `python3 scripts/run_pilot_report.py runs/pilot-dev/gemma --input … --check-states --out runs/pilot-dev/reports/gemma` |
| Synthetic artifact audit | `python3 scripts/audit_smoke.py runs/smoke` |
| Team metrics checks | `python3 -m pytest tests/test_metrics.py` |
| JAILJUDGE HF → runner JSONL | `python3 scripts/adapt_hf_dataset.py --split development` → `data/processed/development_inputs.jsonl` |

Default fixture path: `data/manifests/examples.jsonl`. YAML defaults live under `configs/` (`pilot.yaml`, `inference_smoke.yaml`).

## Migration notes (October 2026 layout)

Former root modules now live under **`refusal_cues/`**. Prefer **`scripts/`** for CLI entrypoints (same flags as before).

| Before | After |
|---|---|
| `from protocol import …` | `from refusal_cues.protocol import …` |
| `python inference.py …` | `python scripts/run_inference.py …` or `python -m refusal_cues.inference …` |
| `python pilot.py …` | `python scripts/run_pilot.py …` |
| `examples.jsonl` | `data/manifests/examples.jsonl` |
| `controls/ab_v1.jsonl` | `data/controls/ab_v1.jsonl` |
| Root `*.md` guides | `docs/*.md` |

Historical notebooks pinned to older commits still expect root-level files; use **`main`** (or re-pin after merge) with `pip install -e .`. Frozen A/B reproduction (`docs/AB_REPRODUCTION.md`) continues to use commit-pinned overlays.

## What this repository provides

- Two model adapters: Ministral 3 8B and Gemma 4 12B, using native chat templates and 4-bit NF4 weights.
- Four response conditions: original body (`B`), refusal prefix (`R`), neutral prefix (`N`), and alternate refusal prefix (`R2`). The original body is preserved byte-for-byte.
- Both output mappings: A = SAFE / B = UNSAFE, and A = UNSAFE / B = SAFE.
- Raw generated text and token IDs, strict verdict parsing, full-vocabulary first-token A/B log probabilities, and a semantic score margin.
- Six decoder activation vectors at the last input token, with identity checks and an independent saved-artifact audit.
- A Python intervention interface for subsequent interpretability work. Nonidentity interventions still require validation.

The judge evaluates an existing response; it is not asked to fulfil the saved request. No model training happens in this runner.

## Google Colab

1. Open [step_zero.ipynb](notebooks/step_zero.ipynb) or [pilot.ipynb](notebooks/pilot.ipynb) and select a GPU runtime.
2. Clone the repo and install GPU deps with `pip install -r requirements.txt` and `pip install -e ".[inference]"` (includes `mistral-common` for Ministral). Notebook cells in `notebooks/step_zero.ipynb` install the same package.
3. Run unit tests and a dry-run or full inference cell.
4. Download artifact archives before the runtime expires.

The notebook defaults to **FP16 on a T4 for synthetic compatibility testing**. The command-line runner defaults to **BF16**, which requires a GPU with native BF16 support.

## Configuration

| Option | Default | Purpose |
|---|---|---|
| `--model` | `gemma` | Choose `gemma` or `ministral`. |
| `--compute-dtype` | `bfloat16` | Compute precision; `float16` only accepts synthetic fixtures. |
| `--revision` | `main` | Hugging Face model revision, resolved to an immutable SHA at startup. |
| `--input` | `data/manifests/examples.jsonl` | JSONL input records. |
| `--out` | `runs/smoke` | A new directory for this run's artifacts. |
| `--variants` | `B R` | Any selection of `B R N R2`. |
| `--baseline` | `native` | `native`, `content`, or `strict` judging instruction. |
| `--output-protocol` | `native_v1` | `native_v1` is unconstrained (4 tokens). `ab_first_token_v1` forces the first token to A or B, then allows trailing text (256-token ceiling). `ab_constrained_v1` is the frozen one-token answer-only arm. |
| `--dry-run` | Off | Write planned prompts without loading a model. |

See [docs/INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md) and [docs/PILOT.md](docs/PILOT.md) for full workflows. For dataset intake (CSV preview and HF release), see [docs/DATASET_PREPARATION.md](docs/DATASET_PREPARATION.md).

## Input contract

Each input line is a JSON object with required fields `source_id`, `cluster_id`, `original_prompt`, `original_response`, `study_partition`, `mapped_unsafe`, and `label_provenance`. See [data/manifests/examples.jsonl](data/manifests/examples.jsonl). Development data additionally needs the checks in [docs/SETUP.md](docs/SETUP.md).

Run outputs (`results.jsonl`, `states_*.pt`, etc.) belong under **`runs/`** locally or **`results/`** for new bulky exports; both are gitignored. Historical audited summaries remain in **`reports/`**.

## Remaining work before research evaluation

- Validate the BF16 configuration and actual development examples, including long complete inputs (no arbitrary 4,096-token intake exclusion; report lengths and handle model context limits at runtime).
- Reconcile the rubric with source annotation rules; audit human labels, provenance and cluster splits.
- Check complete-input eligibility with both tokenizers and the longest approved prompt/variant.
- Validate nonidentity edits, model-specific refusal directions and behavioral controls.
- Integrate evaluation metrics, remaining controls and statistical analysis; satisfy the protocol's power and freeze gates before final evaluation.

Use separate branches and pull requests with a short description, validation evidence and remaining limitations.
