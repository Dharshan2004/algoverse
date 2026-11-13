# Qwen3Guard judge (replaces Ministral)

This is a self-contained Colab experiment. It adds `Qwen/Qwen3Guard-Gen-8B` (bf16, native labels) to the refusal-cue study and produces two things:
- **Behavioral results:** the same 470 dev records, 7 conditions and 2 greedy repeats as the 4-model run, in the same `all_results.csv` schema and with the same metrics.
- **Residual-stream activations:** the same layout as the 4-model capture.

| File | What it is |
|---|---|
| `qwen3guard_experiment.ipynb` | The Colab notebook, built by `build_notebook.py`. Run it on an L4. |
| `src/qwen3guard_judge.py` | `Qwen3GuardRunner`, prompt, strict parser, views, and the resumable behavioral loop + export. Guard turns match `refusal_cues.guard_judges.guard_messages` for a non-Llama-Guard model. |
| `src/activation_tools.py` | The 4-judge capture tools plus a `qwen3guard` decision position, view expansion in the index, and a patching example. Llama Guard and ShieldGemma turns call `refusal_cues.guard_judges`. |
| `src/judge_metrics.py` | DFR@5, DFR@3, FN/FP, B→N, repeat and view disagreement, the Oct 26 gate, two-way flips with McNemar p, and N-referenced DFR. It reproduces the 4-model `analysis.json` and the audit exactly. |
| `tests/test_units.py` | Parser, prompt shape, plan size, and INVALID handling. |
| `tests/test_durability.py` | Chunked Drive writes under an abrupt-close simulation. |
| `tests/units_full.py` | Optional archive parity: all 3,290 prompts, the real 8B tokenizer, and metrics against `all_results.csv`. |
| `tests/notebook_cpu_full.py` | Optional notebook execution on CPU with mocked Colab and `Qwen3Guard-Gen-0.6B`. |

## Protocol
- **Conversation:** `[user: request, assistant: response]` with string contents. This is identical to `refusal_cues.guard_judges.guard_messages` for a non-Llama-Guard guard. Cell 1 writes that module into the extracted `baseline.zip` package before the preflight imports it.
- **Response:** the body for B; `cue + "\n\n" + body` for the other conditions.
- **Template:** the model's own chat template (response-moderation branch, fixed native policy).
- **Decoding:** greedy, 256-token budget, seed `20261004 + repeat`.
- **Output and parse:** the output is `Safety: Safe|Unsafe|Controversial` / `Categories: …` / `Refusal: Yes|No`. Parsing is strict: the first line must be exactly `Safety: <label>`, and anything else is INVALID (kept as INVALID).
- **Views** (stored in the `mapping` column):
  - `controversial_unsafe` (strict): Controversial → UNSAFE.
  - `controversial_safe` (loose): Controversial → SAFE.
  - The native label is in `raw_label`.
- **Layers:** `refusal_cues.protocol.depths(36)` = [8, 12, 16, 19, 23, 26].
- **Decision position:** generation step 2, where the label token after `Safety:` is emitted.

## Tests
Offline checks (parser, plan, chunked writes) run with the repo suite:

```sh
python3 -m pytest experiments/qwen3guard_judge/tests -q
```

Archive parity against the 470-record screen needs local files that are not in git (`inputs/baseline.zip`, `summaries/all_results.csv`, `summaries/jailjudge-full-470-results.zip` under `experiments/behavioral_refusal_screen/`):

```sh
BASELINE_SRC=/tmp/bz FOUR_MODEL_CSV=experiments/behavioral_refusal_screen/summaries/all_results.csv python experiments/qwen3guard_judge/tests/units_full.py
BASELINE_ZIP=experiments/behavioral_refusal_screen/inputs/baseline.zip FOUR_MODEL_ZIP=experiments/behavioral_refusal_screen/summaries/jailjudge-full-470-results.zip python experiments/qwen3guard_judge/tests/notebook_cpu_full.py
```

## Persistence on Colab Drive (chunked-verdicts-v2)

Colab's Drive client uploads a file when it is **closed**; `flush()`/`fsync()` do not upload. The first
version kept `behavioral/qwen3guard/verdicts.jsonl` (and the logs) open in append mode for the whole run, so
when Colab moved the runtime to a new VM on 2026-11-11 (~15:55 SGT) none of those rows had reached Drive.

Now nothing on Drive is held open:

- Behavioral verdicts: immutable closed chunks `behavioral/qwen3guard/verdicts/part-NNNNN.jsonl`, written every
  25 judgments or 60 s (whichever first) to a temp name, closed, then renamed. An interrupt flushes the buffer.
  Worst-case loss on abrupt VM death: 24 finished judgments + the one in progress.
- Reader merges a legacy `verdicts.jsonl` (if any; never modified; a truncated last line is ignored) and all
  chunks; identical duplicates (ignoring `seconds`) are dropped, conflicting duplicates raise.
- Logs: one file per session `logs/<name>-<stamp>-<runtime>.log`, replaced atomically at most every 60 s.
- Index/agreement/mismatch tables, analysis, README, run manifest, heartbeat, lock, shards: write-local then atomic replace.
- Cells 3 and 6 print `Persisted on Drive at start: N unique verdicts (...)`; cell 1 prints the notebook version.

Tests: `tests/test_durability.py` emulates sync-on-close and kills a child process with `os._exit`.
