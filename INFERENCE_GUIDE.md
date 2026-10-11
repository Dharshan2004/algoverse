# How the inference scripts work

These scripts make a language model act as a **judge of an already saved answer**. The judge is not asked to carry out the saved user's request, and no model training occurs.

Read [TESTING.md](TESTING.md) for measured test results and remaining limitations. The ten bundled examples are synthetic plumbing fixtures, not a human-labeled research benchmark.

## Start with the four main files

| File | Job |
|---|---|
| [protocol.py](protocol.py) | Builds response variants and judge messages; defines A/B meanings and strict parsing. |
| [inference.py](inference.py) | Loads the model, runs inference, captures activations, and writes evidence. |
| [audit_smoke.py](audit_smoke.py) | Independently checks the completed default 40-row synthetic run. |
| [notebooks/step_zero.ipynb](notebooks/step_zero.ipynb) | Colab launcher: installs dependencies, runs checks, launches the script, and packages outputs. |

[examples.jsonl](examples.jsonl) supplies one JSON record per line. [tests/test_protocol.py](tests/test_protocol.py) checks prompt construction and parsing without loading a model.

## 1. Read and validate the input

Each record supplies an ID, cluster ID, saved request, saved response, study partition and label metadata. The IDs keep related comparisons together.

The runner rejects duplicate IDs and final-test records. Development records require explicit human label provenance. The T4/FP16 mode accepts synthetic fixtures only.

The stored label is carried into the result for later evaluation. It is **never inserted into the judge's prompt**.

## 2. Make controlled variants

For each saved response body, `variant()` constructs:

| Variant name | Text before the original body |
|---|---|
| B | Nothing; original body |
| R | “I cannot help with that request.” |
| N | “I have read the request above.” |
| R2 | “I am unable to assist with this request.” |

A nonempty prefix is followed by exactly two newline characters. The original body is otherwise preserved, including its UTF-8 bytes and whitespace. Hashes and byte offsets let the audit check that claim.

**Variant B and output code B are different things.** The variant name means “original response”; the output code gets its safety meaning from the mapping below.

The default smoke uses only B and R. Ten records × two variants × two output mappings = **40 judging rows**.

## 3. Build the judge's prompt

`messages()` produces a system message containing the safety policy and output rule, plus a user message containing the saved interaction as JSON.

The model judges every condition twice:

| Mapping | A means | B means |
|---|---|---|
| A_safe | SAFE | UNSAFE |
| A_unsafe | UNSAFE | SAFE |

This separates a change in safety meaning from a preference for a particular letter. If a change makes the model emit A under both mappings, it has not consistently moved toward UNSAFE.

The `native`, `content` and `strict` baselines differ in the judging instruction. The opt-in `--format-reminder` repeats the output instruction after the evidence; it is a synthetic-only diagnostic, not the default research prompt.

## 4. Load one pinned model on the GPU

`Runner.__init__()` resolves the requested Hugging Face revision to an immutable commit. It loads the checkpoint's native Transformers architecture and processor/tokenizer, then records their identities and software versions.

Weights use **4-bit NF4 with double quantization** to reduce GPU memory. Compute precision is a separate setting: BF16 is the research configuration; FP16 is the tested T4 compatibility configuration.

`eval()`, disabled gradients and inference mode prevent training. The script verifies that A and B each encode to one token, and identifies the language decoder blocks used for activation capture.

## 5. Convert messages into model input

`Runner.encode()` applies the model's native chat template. That template adds the model-specific system/user/assistant boundary tokens.

For Gemma, the runner explicitly passes `enable_thinking=False`. It checks the **complete templated input**, not just the saved response, against the 4,096-token maximum. An overlong input raises an error; it is not silently truncated.

## 6. Run inference and capture six states

`Runner.run()` first processes the whole input. This is the **prefill** stage. It then generates up to four new tokens using greedy decoding: choose the highest-scoring permitted next token, with no random sampling.

During prefill, hooks capture the hidden-state vector at the **last input-template token** from six selected decoder depths. For a 48-layer decoder, the zero-based layer indices are 11, 16, 21, 26, 31 and 35.

Each vector is a snapshot of internal model numbers at the boundary immediately before the answer. It is not a safety label. Vectors are moved to CPU and saved; the script does not retain every token from every layer.

Hooks fire once during prefill and skip later decoding steps. They are removed in a `finally` block, including when a run fails.

## 7. Keep verdicts and scores distinct

The script preserves the raw generated text and token IDs. It removes only trailing known EOS/PAD markers, then `parse()` accepts an exact A or B after outer whitespace. Prose, an empty answer, or unexpected thought-channel text stays **INVALID**.

Separately, it records the first generated position's full-vocabulary log probabilities:

- `logp_A` and `logp_B`: scores for the two code tokens.
- `ab_mass = exp(logp_A) + exp(logp_B)`: how much next-token probability is assigned to either valid code.
- `semantic_margin = logp(unsafe code) - logp(safe code)`: positive favors the unsafe meaning; negative favors safe.

For example, if A means SAFE, `logp_A=-2` and `logp_B=-0.2`, the margin is +1.8, favoring UNSAFE. The sign convention stays the same when the code meanings reverse.

A model can have finite A/B scores while generating an INVALID response. The score does not automatically replace the observed verdict. Very low A/B mass warns that neither code is a likely first token.

## 8. Check identity, then save every row

Before the 40-row smoke, the runner makes two additional calls: ordinary inference and an identity intervention. Identity changes nothing. The outputs, code scores and six saved states must agree; otherwise execution stops.

This is why one default model run has **42 calls**.

| Output file | What it contains |
|---|---|
| `run_config.json` | Model revision, precision, template/backend, software, policy, arguments and purpose |
| `identity_check.json` | Whether the two-call identity check passed |
| `results.jsonl` | One row per condition: raw output, verdict, scores, hashes, IDs, timing and memory |
| `states_00000.pt` etc. | Six CPU activation vectors for the corresponding row |
| `artifact_audit.json` | Independent integrity and formatting checks |

The output directory must be new, preventing an accidental mixture of runs.

## What the intervention interface does

An optional edit acts at one layer and the final prefill token. Given hidden state h and unit direction r:

- Remove: subtract the component along r.
- Add: add a specified amount along r.
- Replace: replace that component with the donor state's component.
- Identity: leave h unchanged.

These are mathematical operators. The script does **not** learn a refusal direction or establish that a supplied direction controls refusal. That needs the separate research validation workstream.

## How to run and interpret it

Offline construction checks:

```sh
python -m unittest discover -s tests -v
python inference.py --dry-run --out runs/planned-new
```

Default native Gemma T4 compatibility run:

```sh
python inference.py --model gemma --compute-dtype float16 --out runs/gemma-native-new
python audit_smoke.py runs/gemma-native-new
```

Use a new output directory every time. Pin `--revision` and restore the recorded dependency versions for reproducibility.

An **artifact audit pass** means the stored run is structurally consistent. A **format pass** concerns exact code output. Neither establishes research accuracy or a causal refusal mechanism. The actual development dataset and the research site's remaining gates must still be evaluated.
