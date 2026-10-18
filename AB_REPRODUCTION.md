# Reproduce the frozen Colab upload

```sh
python prepare_ab_upload.py --input runs/pilot-2026-10-17/candidate_inputs.jsonl \
  --out runs/ab-v1-upload.zip
```

Open `notebooks/gemma_ab_v1.ipynb` in Colab through Chrome, select an L4, run top-to-bottom, and upload that ZIP when prompted. The notebook clones GitHub at the exact requested base and verifies every additive source-file hash. The builder retrieves the frozen overlay from immutable Git commit `20450d8`, verifies its expected hashes and the input hash, and refuses existing output archives. Current audit improvements do not silently change the historical experiment. The dataset is uploaded separately from the notebook and is not embedded in committed cells. This notebook is the frozen executed experiment source; changes require a new version and manifest.

The implementation overlay is committed locally at `20450d80a88942388342cc7440f7b58d9ea36a54`. This commit need not be pushed to run the notebook: the upload supplies the reviewed overlay. The downloaded archive records exact executed source copies and their hashes.

After execution, download the result ZIP and executed `.ipynb` through Chrome. Extract into a new ignored run directory and preserve the exact notebook as `executed-colab.ipynb`. Audit every saved tensor and compare with the preserved original:

```sh
python ab_report.py runs/gemma-ab-v1-2026-10-18 \
  --original runs/pilot-2026-10-17/gemma \
  --out reports/gemma-ab-v1-2026-10-18
python diagnostics/archive_ab_evidence.py
python diagnostics/write_ab_comparison.py reports/gemma-ab-v1-2026-10-18/summary.json
```

The report destination must not already exist. The dated archive helper requires the original pre-run fingerprint manifest in `runs/ab-v1-preparation/original_fingerprints.json`; it verifies preservation and notebook execution before sanitizing. It does not execute or fabricate notebook outputs. Use new dated paths and a new protocol version for any changed experiment. The private data, exact downloaded notebook, raw judgments and tensors are intentionally excluded from Git; the committed report and sanitized snapshot contain aggregate execution evidence.
