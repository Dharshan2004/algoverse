"""Score original development responses or the six prefixed control conditions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from refusal_cues.judges import REVISIONS, evaluate, judge_messages, load_judge
from refusal_cues.protocol import MAPPINGS, MODELS, PREFIXES, validate_records
from refusal_cues.refusal_screen import DEFAULT_CONFIG

DATA_SHA256 = "beb131113f4c0cfa7ffb142f074469abe08dc66689b5a55fb70c1f3e6475c012"


def stable_manifest(metadata):
    runtime = {k: v for k, v in metadata["runtime"].items() if k != "gpu"}
    return {**metadata, "runtime": runtime}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=REVISIONS)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    raw = args.data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("Unexpected dataset revision")
    records = validate_records([json.loads(line) for line in raw.splitlines()])
    if len(records) != 470 or sum(r["mapped_unsafe"] for r in records) != 270:
        raise ValueError("Unexpected development cohort")
    config = json.loads(DEFAULT_CONFIG.read_text())
    cues = {"B": ""}
    if args.controls:
        cues = {"N": PREFIXES["N"], **{k: v for k, v in config["cues"].items() if k != "B"}}
    name = args.model
    mappings = list(MAPPINGS) if name in MODELS else ["native"]
    folder = args.output / name
    folder.mkdir(parents=True, exist_ok=True)
    runner = load_judge(name)
    manifest = {"runtime": runner.meta, "input_sha256": DATA_SHA256, "repetitions": config["repetitions"],
                "cues": cues, "seed": config["seed"], "budget": config["max_new_tokens"]}
    path = folder / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if stable_manifest(previous) != stable_manifest(manifest):
            raise ValueError("Run provenance changed. Choose a new output folder.")
    else:
        if (folder / "records").exists() or (folder / "verdicts.jsonl").exists():
            raise ValueError("Existing judgments have no manifest. Choose a new output folder.")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(path)
    # Each chunk is a complete record. Interrupted writes leave only a disposable temp file.
    chunks = folder / "records"
    chunks.mkdir(exist_ok=True)
    counts = Counter()
    verdicts = folder / "verdicts.jsonl"
    with verdicts.with_suffix(".tmp").open("w") as stream:
        for i, record in enumerate(records):
            chunk = chunks / f"{i:04d}.json"
            if not chunk.exists():
                rows = []
                for repeat in range(config["repetitions"]):
                    for condition, cue in cues.items():
                        response = (cue + "\n\n" if cue else "") + record["original_response"]
                        for mapping in mappings:
                            runner.torch.manual_seed(config["seed"] + repeat)
                            turns = judge_messages(name, record["original_prompt"], response, mapping)
                            views, _ = evaluate(runner, name, turns, mapping, config["max_new_tokens"])
                            for view, result in views:
                                rows.append(dict(source_id=record["source_id"], mapping=view, repeat=repeat,
                                                 harmful=record["mapped_unsafe"], verdict=result["verdict"],
                                                 condition=condition, condition_text=cue))
                tmp = chunk.with_suffix(".tmp")
                tmp.write_text(json.dumps(rows))
                tmp.replace(chunk)
            for row in json.loads(chunk.read_text()):
                stream.write(json.dumps(row) + "\n")
                counts[row["condition"], row["mapping"], row["repeat"], row["harmful"], row["verdict"]] += 1
    verdicts.with_suffix(".tmp").replace(verdicts)
    summary = []
    for condition, mapping, repeat in sorted({k[:3] for k in counts}):
        group = condition, mapping, repeat
        summary.append(dict(condition=condition, mapping=mapping, repeat=repeat, harmful=270, safe=200,
                            harmful_missed=counts[group + (True, "SAFE")], safe_flagged=counts[group + (False, "UNSAFE")],
                            invalid_harmful=counts[group + (True, "INVALID")], invalid_safe=counts[group + (False, "INVALID")]))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
