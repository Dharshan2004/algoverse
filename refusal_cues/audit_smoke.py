"""Audit completed synthetic smoke artifacts; never infer research accuracy."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import torch
from refusal_cues.protocol import MAPPINGS, extract_leading_ab, messages, normalize_native_output, parse, sha, variant

def audit(directory):
    root = Path(directory)
    config = json.loads((root / "run_config.json").read_text())
    identity = json.loads((root / "identity_check.json").read_text())
    rows = [json.loads(s) for s in (root / "results.jsonl").read_text().splitlines()]
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    require(identity == {"passed": True, "calls": 2}, "Identity check missing or failed")
    require(len(rows) == 40, "Default B/R smoke must contain 40 completed rows")
    require(len({r["source_id"] for r in rows}) == 10, "Expected ten synthetic sources")
    require(len({(r["source_id"], r["variant"], r["mapping"]) for r in rows}) == 40, "Duplicate conditions")
    dimensions = set()
    for i, row in enumerate(rows):
        require(row["study_partition"] == "synthetic", "Audit is for synthetic smoke only")
        require(row["run_index"] == i and row["status"] == "ok", "Incomplete row")
        require(row["variant"] in ("B", "R") and row["mapping"] in MAPPINGS, "Unexpected condition")
        expected = variant(row["original_response"], row["variant"])
        require(all(row[k] == v for k, v in expected.items()), "Body or variant changed")
        require(row["model_revision"] == config["revision"], "Model revision mismatch")
        parser_version = config.get("output_parser_version", "legacy_eos_only")
        if parser_version == "native_empty_channel_v1":
            normalized, method = normalize_native_output(row["raw_parsed_text"], config["model_id"])
            require(row["parsed_text"] == normalized and row["output_normalization"] == method,
                    "Native normalization mismatch")
        elif parser_version == "first_ab_token_v1":
            code, _method = extract_leading_ab(row["raw_parsed_text"])
            require(row["parsed_text"] == code and row["output_normalization"] == "first_ab_token_v1",
                    "First-token answer mismatch")
        require(row["verdict"] == parse(row["parsed_text"], row["mapping"]), "Verdict mismatch")
        turns = messages(row["original_prompt"], row["assistant_response"], row["mapping"],
                         config["baseline"], policy=config["policy"],
                         format_reminder=config.get("prompt_format", "native") == "reminder_v1")
        require(row["prompt_sha256"] == sha(json.dumps(turns, ensure_ascii=False)), "Prompt reconstruction mismatch")
        require(row["input_tokens"] > 0, "Invalid input length")
        require(all(math.isfinite(row[k]) for k in ("logp_A", "logp_B", "ab_mass", "semantic_margin", "seconds")), "Nonfinite measurement")
        require(0 <= row["ab_mass"] <= 1.00001, "Invalid probability mass")
        mass = math.exp(row["logp_A"]) + math.exp(row["logp_B"])
        require(abs(mass - row["ab_mass"]) < 1e-7, "Probability mass mismatch")
        sign = 1 if row["mapping"] == "A_unsafe" else -1
        require(abs(sign * (row["logp_A"] - row["logp_B"]) - row["semantic_margin"]) < 1e-6, "Semantic margin mismatch")
        states = torch.load(root / row["states_file"], map_location="cpu", weights_only=True)
        require(sorted(states) == config["depths"] and len(states) == 6, "Missing boundary states")
        for state in states.values():
            require(state.ndim == 1 and bool(torch.isfinite(state).all()), "Invalid state vector")
            dimensions.add(state.numel())
    require(len(dimensions) == 1, "Inconsistent hidden dimensions")
    formatting = {}
    for mapping in MAPPINGS:
        group = [r for r in rows if r["mapping"] == mapping]
        invalid = sum(r["verdict"] == "INVALID" for r in group)
        formatting[mapping] = {"invalid": invalid, "rows": len(group), "rate": invalid / len(group)}
    result = {"artifact_audit_passed": True,
        "output_parser_version": config.get("output_parser_version", "legacy_eos_only"),
        "native_wrapper_normalizations": sum(r.get("output_normalization", "none") != "none" for r in rows),
        "raw_exact_code_invalids": sum(parse(r.get("raw_parsed_text", r["parsed_text"]), r["mapping"]) == "INVALID" for r in rows),
        "length_cap_rows": sum(r["length_cap"] for r in rows),
        "ab_mass_min": min(r["ab_mass"] for r in rows),
        "prompt_format": config.get("prompt_format", "native"),
        "protocol_prompt_match": config.get("protocol_prompt_match", True),
        "invalid_by_mapping": formatting,
        "synthetic_format_check_passed": all(v["rate"] <= 0.01 for v in formatting.values()), "rows": len(rows), "identity_calls": 2,
        "verdict_counts": dict(Counter(r["verdict"] for r in rows)),
        "input_tokens_min": min(r["input_tokens"] for r in rows),
        "input_tokens_max": max(r["input_tokens"] for r in rows),
        "peak_gpu_gib": max(r["peak_gpu_bytes"] for r in rows) / 2**30,
        "six_finite_states_per_row": True, "hidden_size": dimensions.pop(),
        "model_revision": config["revision"], "compute_dtype": config["compute_dtype"],
        "results_sha256": sha((root / "results.jsonl").read_text()),
        "research_accuracy_claim": False}
    (root / "artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    audit(parser.parse_args().directory)


if __name__ == "__main__":
    main()
