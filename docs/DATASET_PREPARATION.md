# Dataset preparation → pilot

PR #2 supplies the four CSV preview preparation scripts. They are incorporated with their original commits, followed by integration safeguards: preserve CSV text literally, reject ambiguous labels, validate converted records before writing, and use the shared protocol for response variants.

## Prepare a private candidate sample

From the repository root, install the analysis dependencies and place your authorized original source file at `JAILJUDGE_ID.json` (ignored by Git):

```sh
python -m pip install -r requirements-analysis.txt
python scripts/prepare_preview.py
python scripts/qualify_preview.py
```

The preview selects up to 25 examples per boolean source label, with a fixed seed. Check the printed counts: fewer than 50 records is possible when a class is undersupplied. Nonboolean source labels and empty text are excluded. The cluster audit is a preliminary lexical screen, not a scientific qualification or a verified count of independent requests. Token-length checks (for example `notebooks/history/.../token_check.py`) **report** templated lengths per model; they do **not** drop complete cases solely for exceeding 4,096 input tokens.

Verify the original dataset's label provenance before conversion. Only when confirmed human-labelled:

```sh
python scripts/convert_preview.py --confirm-human-labels
python scripts/create_variants.py
```

The flag records an explicit operator assertion, not automated label verification or scientific approval. Without it, conversion stops rather than automatically claiming human provenance. Converted records are validated before writing; the preview remains candidate development data.

All generated files remain under ignored `data/preview/`. CSV reads preserve text such as `NA`, `null`, leading zeros, quoted commas and whitespace. Labels must be exactly `True` or `False`; truthy strings are never interpreted as booleans.

## JAILJUDGE HF release (development screen adapter)

Public dataset: [`Sabka4/refusal-signal-safety-judgments`](https://huggingface.co/datasets/Sabka4/refusal-signal-safety-judgments) (`development.jsonl`: 470 records, ~270 unsafe / ~200 source-safe). Token length is **diagnostic only** (no 4,096-token intake gate).

From the repo root (requires `huggingface_hub`, included in analysis deps):

```sh
python3 -m pip install -e ".[analysis]"
python3 scripts/adapt_hf_dataset.py --split development
# → data/processed/development_inputs.jsonl (gitignored)
```

Use a local HF-shaped JSONL instead of downloading:

```sh
python3 scripts/adapt_hf_dataset.py --input /path/to/development.jsonl --split development --out data/processed/development_inputs.jsonl
```

**Colab development screen (B/R only, both models)** — after `pip install -e ".[analysis]"` plus GPU deps from `requirements.txt`:

```sh
python3 scripts/adapt_hf_dataset.py --split development
python3 scripts/run_inference.py --model gemma --variants B R --input data/processed/development_inputs.jsonl --out runs/dev-screen-gemma
python3 scripts/run_inference.py --model ministral --variants B R --input data/processed/development_inputs.jsonl --out runs/dev-screen-ministral
```

**Gemma re-run when native 4-token judging returns INVALID** (the model started prose or a channel header before any A/B). This forces the first generated token to A or B, then allows trailing text. The verdict is that first letter. BF16 is the default and is repeated here so the Colab command is complete. Native judging stays available by omitting `--output-protocol`:

```sh
python3 scripts/run_inference.py \
  --model gemma \
  --variants B R \
  --compute-dtype bfloat16 \
  --input data/processed/development_inputs.jsonl \
  --output-protocol ab_first_token_v1 \
  --out runs/dev-gemma-br-first-token
```

`--output-protocol native_v1` is the unconstrained 4-token path. `--output-protocol ab_constrained_v1` is the older one-token answer-only arm and does not allow text after the letter.

Or run `bash scripts/run_development_screen.sh` (same steps; needs CUDA). This is **not** the full four-variant pilot (1920+ calls per model); plan GPU time accordingly.

`--split final_test` writes adapted JSONL but prints a warning: **`validate_records` / inference refuse `final_test`** until protocol freeze — do not bypass.

## Run inference

Upload `data/preview/candidate_inputs.jsonl` through [the clean pilot notebook](notebooks/pilot.ipynb), or use:

```sh
python scripts/run_pilot.py --input data/preview/candidate_inputs.jsonl --out runs/new-pilot
```

The runner generates B/R/N/R2 itself; `variants.jsonl` is an optional inspection artifact, not its input. The actual 17 October pilot used a previously shared candidate JSONL, not a regenerated sample from these integrated scripts. Historical outputs are unchanged. See [PILOT.md](PILOT.md) for GPU setup and full artifact auditing.

## Integration validation

The integration test runs all four preparation scripts on synthetic edge cases, checks exact prompt/response preservation, confirms label handling, requires explicit provenance confirmation, and checks that variants match `protocol.py`. All 17 protocol/pilot/preparation tests and all 14 team metrics tests pass locally. No fresh GPU inference or original-source scientific qualification was performed for this integration.
