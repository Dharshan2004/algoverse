# Dataset preparation → pilot

Sabika's PR #2 supplies the four preparation scripts. They are incorporated with their original commits, followed by integration safeguards: preserve CSV text literally, reject ambiguous labels, validate converted records before writing, and use the shared protocol for response variants.

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

## Run inference

Upload `data/preview/candidate_inputs.jsonl` through [the clean pilot notebook](notebooks/pilot.ipynb), or use:

```sh
python scripts/run_pilot.py --input data/preview/candidate_inputs.jsonl --out runs/new-pilot
```

The runner generates B/R/N/R2 itself; `variants.jsonl` is an optional inspection artifact, not its input. The actual 17 October pilot used Sabika's previously shared JSONL, not a regenerated sample from these integrated scripts. Historical outputs are unchanged. See [PILOT.md](PILOT.md) for GPU setup and full artifact auditing.

## Integration validation

The integration test runs all four preparation scripts on synthetic edge cases, checks exact prompt/response preservation, confirms label handling, requires explicit provenance confirmation, and checks that variants match `protocol.py`. All 17 protocol/pilot/preparation tests and all 14 team metrics tests pass locally. No fresh GPU inference or original-source scientific qualification was performed for this integration.
