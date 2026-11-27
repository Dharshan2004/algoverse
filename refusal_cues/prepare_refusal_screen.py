"""Pin and adapt only the two requested evaluation pools; never fetch final_test."""
import argparse
import csv
import json
from pathlib import Path

from refusal_cues.adapt_hf_dataset import adapt_records
from refusal_cues.protocol import sha, validate_records

SOURCES = {
    "jailjudge": ("Sabka4/refusal-signal-safety-judgments", "development.jsonl"),
    "wildguard": ("Sabka4/wildguard-preparation", "data/processed/wildguard_independent_safety_provisional.csv"),
}


def adapt_wildguard(rows):
    adapted = []
    for r in rows:
        if r["study_split"] != "independent_safety":
            raise ValueError("Expected independent_safety pool, not reader/benign records")
        if r["response_harm_label"] not in ("harmful", "unharmful"):
            raise ValueError("Missing or unknown human response-harm label")
        refusal = r["response_refusal_label"]
        adapted.append({"source_id": "wildguard:" + r["source_row_id"],
                        "cluster_id": r["question_group_id"],
                        "original_prompt": r["prompt"], "original_response": r["response"],
                        "mapped_unsafe": r["response_harm_label"] == "harmful",
                        "label_provenance": "human", "study_partition": "development",
                        "genuine_refusal": (refusal == "refusal" and r["response_harm_label"] == "unharmful")
                                           if refusal in ("refusal", "compliance") else None,
                        "meta": {"source_study_split": r["study_split"],
                                 "pool_status": "provisional_supporting_evaluation",
                                 "response_refusal_label": refusal,
                                 "response_harm_label": r["response_harm_label"],
                                 "genuine_refusal_definition": "human refusal label AND human unharmful response label"}})
    return validate_records(adapted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    parser.add_argument("--revision", required=True, help="Dataset revision to resolve and record")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output path to preserve dataset provenance")
    from huggingface_hub import HfApi, hf_hub_download
    repo, filename = SOURCES[args.dataset]
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    source = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    text = source.read_text()
    if args.dataset == "jailjudge":
        raw = [json.loads(s) for s in text.splitlines() if s.strip()]
        if any(r.get("study_partition") != "development" for r in raw):
            raise ValueError("Development file contains unexpected partition")
        records = validate_records(adapt_records(raw, partition="development"))
    else:
        with source.open(newline="") as f:
            records = adapt_wildguard(list(csv.DictReader(f)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"repo": repo, "revision": revision, "source_file": filename,
                "source_sha256": sha(text), "adapted_sha256": sha(args.out.read_text()),
                "records": len(records), "harmful": sum(r["mapped_unsafe"] for r in records),
                "unique_prompt_groups": len({r["cluster_id"] for r in records}),
                "pool_status": "development" if args.dataset == "jailjudge" else "provisional_supporting_evaluation"}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
