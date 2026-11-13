# Algoverse: refusal wording and safety judgments

Inference tooling for a study of how refusal wording changes an LLM judge's safety decisions when the underlying response stays the same. The runner judges saved prompt–response pairs, records output probabilities, and captures internal activations for the team's interpretability work.

**Current status:** [50-record exploratory pilot completed](reports/pilot-2026-10-17/README.md) on both models using native BF16 on an L4: 800 judgments plus four identity checks, with full saved-artifact audits passed. Source-data qualification and scientific development gates remain open. These results do not establish benchmark accuracy, a causal refusal mechanism, or readiness for final evaluation.

**18 October follow-up:** [Gemma A/B-constrained comparison](reports/gemma-ab-v1-2026-10-18/README.md) completed 928 judgments plus eight identity checks. Output validity improved to 100%, but dataset label agreement changed only 84.75% → 85.00%, and clean-control accuracy fell 98.44% → 93.75%. The original pilot and ten INVALIDs remain intact. See the [frozen protocol](docs/AB_PROTOCOL.md) and [reproduction guide](docs/AB_REPRODUCTION.md).

[Open in Google Colab](https://colab.research.google.com/github/Dharshan2004/algoverse/blob/main/notebooks/step_zero.ipynb) · [Research protocol](https://refusal-wording-safety-judgments.dharshanlab.chatgpt.site/) · [Pipeline walkthrough](docs/INFERENCE_GUIDE.md) · [Test evidence](docs/TESTING.md) · [Team handoff](docs/SETUP.md)

[Animated code and results walkthrough](media/pilot-explainer/README.md) · [Notebook and run history](notebooks/README.md)

**Case-review update:** [Both robust Gemma flips need source/policy adjudication](reports/gemma-case-review-2026-10-18/README.md). One is an unfinished request continuation and the other fictional narrative. Setting them aside in a post-hoc sensitivity check reverses the net refusal effect; scaling remains on hold pending original-source reconciliation and human review.

**470-record experiments:** the later screens and activation captures live under [`experiments/`](experiments/README.md). Llama Guard and ShieldGemma inference is [`refusal_cues/guard_judges.py`](refusal_cues/guard_judges.py) (`GuardRunner`, `guard_messages`, `parse_guard`).

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
  refusal_cues/               # importable package (protocol, inference, guard judges, metrics, …)
    guard_judges.py           # GuardRunner, guard_messages, parse_guard (Llama Guard, ShieldGemma)
    directions/               # probe/direction hooks (Mehrage); README stub
    diagnostics/              # offline replay on saved runs
  experiments/                # 470-record screens and activation captures (see below)
    behavioral_refusal_screen/
    activation_capture/
    qwen3guard_judge/
  scripts/                    # thin CLIs → refusal_cues (stable teammate entrypoints)
  notebooks/                  # Colab workflows for the pilot and A/B comparison
  docs/                       # SETUP, PILOT, INFERENCE_GUIDE, TESTING, …
  reports/                    # historical audited summaries (tracked)
  results/ artifacts/         # new bulky outputs (gitignored)
  tests/                      # unittest + pytest (metrics, guard judges)
```

`refusal_cues/` is the installable package: protocol construction, the Gemma 4 12B and Ministral 3 8B pilot runner (`inference.Runner`), metrics, A/B reports, and the guard interface. `experiments/` calls that package for those pieces. Pilot notebooks stay in `notebooks/`. The 470-record Colab notebooks stay in `experiments/`.

## Refusal-cue experiments

Details and run order: [experiments/README.md](experiments/README.md).

### Behavioral refusal screen

[`experiments/behavioral_refusal_screen/`](experiments/behavioral_refusal_screen/README.md). 470 development records (270 human-harmful, 200 human-safe), seven conditions (`B`, `N`, `R`, `R2`, `paper_R1`, `paper_R2`, `paper_Rgreet`), two greedy repeats.

| Judge | Checkpoint | Output |
|---|---|---|
| Llama Guard 3 1B | `meta-llama/Llama-Guard-3-1B` | native `safe` / `unsafe` |
| ShieldGemma 2B | `google/shieldgemma-2b` | Yes/No score |
| Ministral 3 8B | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `A_safe` and `A_unsafe` |
| Gemma 4 12B | `google/gemma-4-12B-it` | `A_safe` and `A_unsafe` |

Notebook: `behavioral_refusal_screen.ipynb` (resume schedule, `RESUME_MISSING_ONLY = True`). On an L4 or A100, run cells 2 → 3 → 4 → 6 → 7 → 8. Drive inputs are `baseline.zip` plus, for a resume, the baseline and control result zips. `upgrade_notebook.py` rebuilds the notebook when the predecessor notebooks are present locally. `summaries/analyze_full_run.py` rebuilds `analysis.md` from those local archives. Git tracks the code, the notebook, and the small summaries (`analysis.md`, `analysis.json`, `all_results_summary.csv`, `audit/`). The 39,480-row verdict table and the result zips stay local.

### 4-judge activation capture

[`experiments/activation_capture/`](experiments/activation_capture/README.md). Same four judges, same 470 records and seven conditions, one greedy pass: 19,740 prompts. Six decoder depths from `refusal_cues.protocol.depths`. Vectors: `decision`, `final_token`, `response_mean`, `cue_mean`, `body_mean`.

Rebuild and run:

```sh
python3 experiments/activation_capture/build_notebook.py
```

Open `activation_capture.ipynb` on an L4 and run cells 1 → 2 → 3 → 4 → 6 → 7 → 8. Cell 10 is an optional residual-patch example.

### Qwen3Guard-Gen-8B screen and capture

[`experiments/qwen3guard_judge/`](experiments/qwen3guard_judge/README.md). `Qwen/Qwen3Guard-Gen-8B` (bf16, native `Safety:` labels) on the same 470 records and seven conditions, two greedy repeats (6,580 generations) and a residual capture (3,290 prompts). Controversial is stored as two views, `controversial_unsafe` and `controversial_safe`. Decision position is the label token after `Safety:` (generation step 2). Layers are `refusal_cues.protocol.depths(36)` = `[8, 12, 16, 19, 23, 26]`.

```sh
python3 experiments/qwen3guard_judge/build_notebook.py
```

Open `qwen3guard_experiment.ipynb` on an L4 and run cells 1 → 2a → 2b → 2c → 3 → 4 → 6 → 7 → 9 → 10 → 11.

### Where judge inference and activation code live

| Judge | Inference | Activations |
|---|---|---|
| Llama Guard 3 1B | `refusal_cues/guard_judges.py` — `GuardRunner`, `guard_messages` (typed text blocks), `parse_guard` | `experiments/activation_capture/src/activation_tools.py` (`build_turns`, `capture_one`, `run_capture`, `verdict_with_residual_patch`) |
| ShieldGemma 2B | `refusal_cues/guard_judges.py` — `GuardRunner`, `guard_messages` (string contents), Yes/No score in `GuardRunner.run` | same capture module |
| Gemma 4 12B | `refusal_cues/inference.py` `Runner` for the pilot letter protocol. The 470-record screen and capture use that runner together with `structured_messages` from the Colab `baseline.zip` | same capture module. The notebook's `load_judge` constructs `Runner` for Gemma |
| Ministral 3 8B | same chat `Runner` plus the baseline bundle's structured JSON turns | same capture module |
| Qwen3Guard-Gen-8B | `experiments/qwen3guard_judge/src/qwen3guard_judge.py` — `Qwen3GuardRunner`, strict `Safety:` parser, two Controversial views | `experiments/qwen3guard_judge/src/activation_tools.py`, including `qwen3guard_verdict_with_residual_patch` |

`GuardRunner` matches the chat runner's surface (`.model`, `.tokenizer`, `.revision`, `.meta`, `.encode(turns)`, `.run(turns, budget)`). Import it with:

```python
from refusal_cues.guard_judges import GuardRunner, guard_messages, parse_guard
```

Metrics shared with the 470-record screen are in `experiments/qwen3guard_judge/src/judge_metrics.py` (DFR@5, DFR@3, false negatives and false positives, the development gate). The screen's own recomputation script is `experiments/behavioral_refusal_screen/summaries/analyze_full_run.py`.

Offline tests for the package and the experiments:

```sh
python3 -m pip install -e ".[analysis]" pytest
python3 -m pytest -q
```

**Role ownership**

| Area | Owner |
|---|---|
| Inference runner, Colab, audits | Priyadharshan |
| Dataset intake, preview pipeline | Sabika |
| Refusal directions / probes | Mehrage (Sayed) — `refusal_cues/directions/` |
| Evaluation metrics | Hemanya, Laasya — `refusal_cues/metrics.py` |
| Research guidance | Pratham |

## Install

**Analysis-only (no GPU, no model download):**

```sh
git clone https://github.com/Dharshan2004/algoverse.git
cd algoverse
python3 -m pip install -e ".[analysis]"
python3 -m unittest discover -s tests -v
python3 -m pytest tests/test_metrics.py -q
```

**GPU inference:** install `requirements.txt` (or Colab cells in `notebooks/step_zero.ipynb`), then `pip install -e .` so notebooks and scripts resolve `refusal_cues`.

## Quick commands

| Task | Command |
|---|---|
| Dry-run input plan (80 calls on fixtures) | `python3 scripts/run_inference.py --dry-run --variants B R N R2 --out runs/planned` |
| Exploratory pilot launcher | `python3 scripts/run_pilot.py --input data/processed/candidate_inputs.jsonl --out runs/pilot-dev` |
| Pilot aggregate report | `python3 scripts/run_pilot_report.py runs/pilot-dev/gemma --input … --check-states --out runs/pilot-dev/reports/gemma` |
| Synthetic artifact audit | `python3 scripts/audit_smoke.py runs/smoke` |
| Team metrics checks | `python3 -m pytest tests/test_metrics.py` |
| Package and experiment tests | `python3 -m pytest -q` |

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
2. Clone the repo, run `pip install -e ".[analysis]"` plus inference dependencies (see notebook cells).
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
| `--dry-run` | Off | Write planned prompts without loading a model. |

See [docs/INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md) and [docs/PILOT.md](docs/PILOT.md) for full workflows. For Sabika's source-data pipeline, see [docs/DATASET_PREPARATION.md](docs/DATASET_PREPARATION.md).

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
