# Notebook and run history

This index distinguishes code templates, evidence of execution, and sanitized historical snapshots. A notebook file alone is not proof that its cells ran.

| Date / stage | Notebook or source | Execution evidence | Coverage and limitations |
|---|---|---|---|
| 11 October: T4 compatibility and Gemma parser work | [step_zero.ipynb](step_zero.ipynb); source revisions preserved in Git | [TESTING.md](../TESTING.md) records runs, checkpoint revisions and output hashes | Reusable source is present. Exact executed Colab notebook exports for every older diagnostic were **not recovered**; this is not a complete execution-notebook archive. |
| 17 October: sample qualification | [Sanitized qualification notebook](history/2026-10-17/qualification.sanitized.ipynb) | Original local notebook has execution counts 1 and 2; private generated check artifacts retained | No model inference. Counts are metadata, not independent proof of each cell result. Outputs cleared in Git; original helper scripts included for provenance. |
| 17 October: actual BF16 pilot | [Sanitized executed pilot notebook](history/2026-10-17/pilot.executed.sanitized.ipynb) | Downloaded completed Colab notebook with execution counts 1 and 2; [audited results](../reports/pilot-2026-10-17/README.md) confirm 400 judgments + 2 identity calls per model | Original notebook preserved privately. Embedded data/source bundle, outputs and runtime metadata removed from public snapshot. Setup replaced with loading a private archive; edited copy has not been rerun. |
| 17 October: reusable pilot launcher | [pilot.ipynb](pilot.ipynb) | Notebook syntax checked; launcher dry-run validated | **Not the notebook that launched the completed pilot.** New GPU execution through this template has not been performed. |
| 18 October: Gemma A/B-constrained comparison | [Frozen launcher](gemma_ab_v1.ipynb); [sanitized executed snapshot](history/2026-10-18/gemma_ab_v1.executed.sanitized.ipynb) | Chrome/Colab L4 execution counts 1–4; [audited comparison](../reports/gemma-ab-v1-2026-10-18/README.md); [notebook provenance](history/2026-10-18/provenance.json) | 928 judgments plus eight identity calls: 50 records × four variants × both mappings × two protocols, and eight synthetic controls likewise. Code cells unchanged; original aggregate stdout retained. Formatting success is separate from label agreement. |

## Evidence preserved privately

The exact downloaded pilot notebook is at `runs/pilot-2026-10-17/executed-colab.ipynb`; raw results, source bundle, model configuration, logs and 800 tensor files are in the same ignored directory. The complete run archive is `runs/pilot-2026-10-17.zip`. The qualification original and private artifacts remain in the parent workspace's `output/sample-qualification-2026-10-17/`.

[provenance.json](history/2026-10-17/provenance.json) records original notebook hashes, observed execution counts, embedded payload hashes and sanitization changes. Neither edited snapshot claims to be byte-identical to its original. Inference and protocol copies in the actual run were verified identical to the tracked code; their hashes are in the result report.

The qualification helper scripts are historical source copies, not portable launchers. They refer to the original workspace/input layout. The archived original pilot summarizer is the lightweight summary used in Colab; the current `pilot_report.py` subsequently performed the full local tensor audits and produced the committed reports.

## Earlier source history

Git retains the earlier step-zero notebook versions, including `73e1615` (T4 configuration), `1ea4592` (streamed logs), `931c582` (audit before summaries), `d71be31` (corrected-parser documentation), and `69fcb4a` (main-branch workflow). These are source revisions, not recovered executed exports. See `git log --all -- notebooks/step_zero.ipynb`.

## Future runs

1. Save an executed notebook export privately before ending the runtime.
2. Keep raw data, model outputs and credentials out of committed notebooks.
3. Add a dated sanitized snapshot and a provenance record describing every edit.
4. Link the immutable runner version, input/output hashes, audit, and aggregate report here.
5. Mark dry runs, incomplete attempts and templates explicitly; never reconstruct missing execution output as though it were original.

The exact 18 October executed notebook and 928 tensor files are retained in ignored `runs/gemma-ab-v1-2026-10-18/`, together with results, source copies, environment, logs and remote audit. Its downloaded ZIP is preserved separately. All 817 original pilot files were verified unchanged.
