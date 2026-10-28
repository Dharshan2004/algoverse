"""Adapt Sabika's Hugging Face release to the refusal_cues runner JSONL schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from refusal_cues.paths import REPO_ROOT
from refusal_cues.protocol import validate_records

DEFAULT_REPO = "Sabka4/refusal-signal-safety-judgments"
SPLIT_FILES = {
    "development": "development.jsonl",
    "final_test": "final_test.jsonl",
    "control_subset": "control_subset.jsonl",
    "replication_subset": "replication_subset.jsonl",
    "unused": "unused.jsonl",
}
RUNNER_FIELDS = (
    "source_id",
    "cluster_id",
    "original_prompt",
    "original_response",
    "mapped_unsafe",
    "label_provenance",
    "study_partition",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_hf_split(repo: str, split: str) -> list[dict[str, Any]]:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; choose from {sorted(SPLIT_FILES)}")
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo, SPLIT_FILES[split], repo_type="dataset"))
    return _load_jsonl(path)


def adapt_row(row: dict[str, Any], *, expected_partition: str | None = None) -> dict[str, Any]:
    """Map one HF record to runner schema; preserve extras under ``meta``."""
    prompt = row.get("user_prompt")
    response = row.get("model_response")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing user_prompt for source_id={row.get('source_id')!r}")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Missing model_response for source_id={row.get('source_id')!r}")

    partition = row.get("study_partition")
    if expected_partition is not None and partition != expected_partition:
        raise ValueError(
            f"source_id={row.get('source_id')!r} has study_partition={partition!r}, "
            f"expected {expected_partition!r}"
        )

    adapted = {
        "source_id": row["source_id"],
        "cluster_id": row["cluster_id"],
        "original_prompt": prompt,
        "original_response": response,
        "mapped_unsafe": row["mapped_unsafe"],
        "label_provenance": row.get("label_provenance", ""),
        "study_partition": partition,
    }
    meta = {k: v for k, v in row.items() if k not in RUNNER_FIELDS and k not in ("user_prompt", "model_response")}
    if meta:
        adapted["meta"] = meta
    return adapted


def adapt_records(
    rows: list[dict[str, Any]],
    *,
    partition: str,
) -> list[dict[str, Any]]:
    adapted = [adapt_row(r, expected_partition=partition) for r in rows if r.get("study_partition") == partition]
    if not adapted:
        raise ValueError(f"No rows with study_partition={partition!r}")
    seen: set[str] = set()
    for r in adapted:
        sid = r["source_id"]
        if sid in seen:
            raise ValueError(f"Duplicate source_id in adapted output: {sid}")
        seen.add(sid)
    return adapted


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    unsafe = sum(1 for r in rows if r["mapped_unsafe"])
    return {
        "n": len(rows),
        "unsafe": unsafe,
        "source_safe": len(rows) - unsafe,
        "unique_source_ids": len({r["source_id"] for r in rows}),
    }


def write_adapted(
    rows: list[dict[str, Any]],
    out: Path,
    *,
    partition: str,
    validate_runner: bool = True,
) -> dict[str, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    if partition == "final_test":
        print(
            "\n*** WARNING: final_test partition ***\n"
            "Output is for inspection/export only. refusal_cues.protocol.validate_records and "
            "scripts/run_inference.py REFUSE final_test until the protocol freeze gate passes. "
            "Do not bypass this check.\n",
            file=sys.stderr,
        )
        validate_runner = False
    elif validate_runner and partition == "development":
        validate_records(rows)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = summarize(rows)
    print(json.dumps({"out": str(out.resolve()), "partition": partition, **stats}, indent=2))
    if stats["n"] != stats["unique_source_ids"]:
        raise ValueError("Duplicate source_id in written file")
    return stats


def default_out_path(partition: str) -> Path:
    name = "development_inputs.jsonl" if partition == "development" else f"{partition}_inputs.jsonl"
    return REPO_ROOT / "data" / "processed" / name


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Local HF-shaped JSONL (default: download from --repo)",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo id")
    parser.add_argument(
        "--split",
        default="development",
        choices=sorted(SPLIT_FILES),
        help="Dataset file / study_partition filter (default: development)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output runner JSONL (default: data/processed/<split>_inputs.jsonl)",
    )
    args = parser.parse_args(argv)
    out = args.out or default_out_path(args.split)

    if args.input:
        raw = _load_jsonl(args.input.resolve())
    else:
        print(f"Downloading {args.repo} / {SPLIT_FILES[args.split]} …", flush=True)
        raw = load_hf_split(args.repo, args.split)

    adapted = adapt_records(raw, partition=args.split)
    if len(adapted) != len(raw):
        print(
            f"Note: kept {len(adapted)} / {len(raw)} rows with study_partition={args.split!r}",
            flush=True,
        )
    write_adapted(adapted, out.resolve(), partition=args.split)


if __name__ == "__main__":
    main()
