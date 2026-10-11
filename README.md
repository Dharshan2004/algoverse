# Algoverse: refusal wording and safety judgments

**TLDR:** Colab step-zero inference setup for Priyadharshan's October 8 assignment. Builds exact response variants, runs native judging under both output mappings, saves raw verdicts/log probabilities and six boundary-token activations. Includes a reusable prefill-only intervention interface for the interpretability workstream. T4/FP16 synthetic smoke execution is verified; BF16 protocol validation remains pending.

- [Open the Colab notebook](https://colab.research.google.com/github/Dharshan2004/algoverse/blob/dharshan/inference-setup/notebooks/step_zero.ipynb)
- [Research protocol](https://refusal-wording-safety-judgments.dharshanlab.chatgpt.site/)
- [Setup, ownership and remaining checks](SETUP.md)
- [T4 test results and limitations](TESTING.md)
- [How the inference scripts work](INFERENCE_GUIDE.md)
- [Inference runner](inference.py) · [Protocol helpers](protocol.py) · [Synthetic examples](examples.jsonl)

## T4 compatibility test

Priyadharshan approved T4 testing on 11 October. The notebook defaults to `--compute-dtype float16` for synthetic fixtures only. The CLI still defaults to BF16. FP16 outputs are labeled `T4_FP16_compatibility_only` and cannot be used as validation of the research protocol's BF16 configuration. Both models completed 42-call T4 runs and passed artifact/identity checks. Ministral produced 0/40 INVALID outputs; Gemma initially produced 4/40 legacy-parser INVALIDs; its empty native channel wrappers are now handled correctly. A fresh unchanged-prompt 42-call Gemma run passed with 0/40 INVALID and nine unit tests passed. Raw wrappers, length-cap flags and original low A/B mass remain recorded. Peak allocated memory was 6.253/7.634 GiB respectively. See [TESTING.md](TESTING.md) for exact revisions, limitations and fixes.

```sh
python inference.py --model ministral --compute-dtype float16 --out runs/ministral-t4
# Fresh runtime for the second model:
python inference.py --model gemma --compute-dtype float16 --out runs/gemma-t4
```

### Gemma formatting follow-up

The issue was in native control-syntax parsing: Gemma's empty channel role header was counted as answer prose. The corrected parser removes only a recognized empty wrapper before checking exact A/B, preserving raw text and scores. The default research prompt and four-token cap are unchanged. The reminder and newline experiments were unsuccessful and are not part of the correction. See [verified evidence](TESTING.md#verified-gemma-parser-correction--11-october-2026) and [the script walkthrough](INFERENCE_GUIDE.md). BF16 and scientific qualification remain pending; cumulative smoke/debug calls are 184.

## Protocol configuration

Use the notebook with a **BF16-capable CUDA GPU**, preferably L4 24 GB or A100. Run one model per fresh runtime. A T4 does not match the protocol's BF16 configuration. No paid runtime is provisioned by this repository.

The notebook installs current official dependencies for an initial compatibility smoke test, records installed versions and resolves each model to an immutable Hugging Face commit. Save that environment lock and use it for later runs; a moving dependency install is not a frozen final experiment.

Offline, without downloading weights:

```sh
python -m unittest discover -s tests -v
python inference.py --dry-run --variants B R N R2 --out runs/planned
```

On Colab after dependency installation:

```sh
python inference.py --model gemma --variants B R N R2 --out runs/gemma-smoke
# Restart runtime before loading the other checkpoint.
python inference.py --model ministral --variants B R N R2 --out runs/ministral-smoke
```

Ten illustrative records × four variants × two mappings = 80 calls/model, plus two identity checks. This full variant plumbing test across both models would use 164 calls, exceeding the protocol's 160-call smoke allowance by four. **The notebook defaults to B/R only (42/model; 84 total)**. Additional variants/baselines require an explicit budget review before running. Offline dry runs do not consume inference calls.

## Scope

Implemented: B/R/N/R2 construction with byte offsets and hashes, both A/B mappings, native/content-focused/strict prompts, strict output parsing, NF4/BF16 loading, greedy decoding, immutable checkpoint resolution, complete-input length rejection, CPU boundary states, identity reproducibility checks, run metadata, and raw JSONL outputs. `Runner.run` also supports generation with `max_new_tokens=256` and optional identity/removal/addition/coordinate-replacement hooks. A hook fires only once during prefill. It does not select or validate a refusal direction.

Not implemented here: dataset acquisition/grouping, human label audit, direction estimation and behavioral validation, Shieldstral scoring, final locked-run orchestration, full control matrix, literal-label sequence scoring, or statistical analysis. Those remain separate workstreams/integration tasks. This is an inference foundation, not a completed experiment. `messages(..., literal=True)` constructs a literal prompt, but primary runner scoring is A/B only.

## Outputs

`run_config.json`, `identity_check.json`, `results.jsonl`, and `states_*.pt` under a new run directory. Every row retains raw output, mapped verdict, A/B full-vocabulary log probabilities and combined mass, semantic margin, model revision, template input hash, prompt hash, original-body hash, timing, memory and source IDs. Invalid output stays INVALID. Run outputs are ignored by git and should be kept in private working storage.

The ten examples are synthetic placeholders, not human-annotated benchmark data. Their provisional labels test the schema only. Real development input must be normalized by Sabika's pipeline and include `label_provenance: "human"`; this runner deliberately rejects final-test partitions. The primary rubric is the site's operational paraphrase and still needs reconciliation with released source annotation rules before a scientific screen.

## Collaboration

Use separate branches and pull requests. Include a TLDR, validation evidence and blockers in each handoff. Do not commit tokens, gated source text, model weights or output archives. Team invites were reported sent on October 4; invitation acceptance has not been verified here.
