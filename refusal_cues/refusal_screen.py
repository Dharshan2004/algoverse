"""Five-cue development screen: generated JSON chat judges and native guards."""
import argparse
import json
import random
import subprocess
from pathlib import Path

from refusal_cues.paths import REPO_ROOT
from refusal_cues.protocol import MODELS, MAPPINGS, POLICY, sha, validate_records
from refusal_cues.judges import REVISIONS, load_judge, judge_messages, evaluate

DEFAULT_CONFIG = REPO_ROOT / "configs/refusal_screen.json"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sample_records(records, limit, seed):
    if not records:
        raise ValueError("Input is empty")
    if limit is None:
        return records
    if limit < 1 or limit > len(records):
        raise ValueError("Sample size must be positive and no larger than the dataset")
    # Balance labels where possible; retain a fixed sample across every model.
    rng = random.Random(seed)
    pools = [[r for r in sorted(records, key=lambda r: r["source_id"])
              if r["mapped_unsafe"] == label] for label in (False, True)]
    for pool in pools:
        rng.shuffle(pool)
    selected = []
    while len(selected) < limit:
        for pool in pools:
            if pool and len(selected) < limit:
                selected.append(pool.pop())
    return selected


def make_plans(records, model, config):
    cues = config["cues"]
    if len(cues) != 6 or cues.get("B") != "" or any(not v for k, v in cues.items() if k != "B"):
        raise ValueError("Expected baseline B and exactly five nonempty refusal cues")
    if config["repetitions"] < 1 or config["max_new_tokens"] < 1 or config["cue_position"] != "prefix":
        raise ValueError("Invalid repetitions, budget or cue position")
    for repeat in range(config["repetitions"]):
        for record in records:
            for cue_id, cue in cues.items():
                body = record["original_response"]
                lead = cue + "\n\n" if cue else ""
                response = lead + body
                for mapping in (MAPPINGS if model in MODELS else ["native"]):
                    turns = judge_messages(model, record["original_prompt"], response, mapping)
                    yield {"source_id": record["source_id"], "cluster_id": record["cluster_id"],
                           "mapped_unsafe": record["mapped_unsafe"],
                           "genuine_refusal": record.get("genuine_refusal", record.get("meta", {}).get("genuine_refusal")),
                           "study_partition": record["study_partition"],
                           "variant": cue_id, "cue_text": cue, "repeat": repeat, "mapping": mapping,
                           "body_sha256": sha(body), "variant_sha256": sha(response),
                           "body_byte_start": len(lead.encode()), "body_byte_end": len(response.encode()),
                           "messages": turns, "prompt_sha256": sha(json.dumps(turns, ensure_ascii=False))}


def run_screen(args):
    config = json.loads(args.config.read_text())
    if args.repetitions is not None:
        config["repetitions"] = args.repetitions
    records = validate_records(read_jsonl(args.input))
    records = sample_records(records, args.limit, config["seed"])
    plans = list(make_plans(records, args.model, config))
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {"config": config, "model": args.model, "dataset": args.dataset,
                "input_file_sha256": sha(args.input.read_text()),
                "source_ids": [r["source_id"] for r in records],
                "n_records": len(records), "n_calls": len(plans), "n_rows": len(plans) * (2 if args.model == "qwen3guard" else 1), "git_commit": commit,
                "dry_run": args.dry_run, "policy_sha256": sha(POLICY),
                "selection": "all" if args.limit is None else "balanced_fixed_seed",
                "decoding": "greedy", "chat_output_protocol": "json_ab_constrained_v1", "repetitions_are_independent_samples": False}
    manifest["code_sha256"] = {str(p.relative_to(REPO_ROOT)): sha(p.read_text())
                               for p in sorted((REPO_ROOT / "refusal_cues").rglob("*.py"))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (out / "planned_inputs.jsonl").open("w") as f:
        for plan in plans:
            f.write(json.dumps(plan, ensure_ascii=False) + "\n")
    if args.dry_run:
        print(f"Planned {len(plans)} calls across {len(records)} records. NO model was run.")
        return
    runner = load_judge(args.model, args.revision)
    manifest["runtime"] = runner.meta
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    budget = config["max_new_tokens"]
    # Fail on any context overflow before beginning this run. Never truncate evidence.
    for plan in plans:
        encoded = runner.encode(plan["messages"])
        limit = (runner.model_max_position_embeddings if args.model in MODELS
                 else runner.model.config.max_position_embeddings)
        reserve = 1 if args.model == "shieldgemma" else budget
        if limit and encoded["input_ids"].shape[-1] + reserve > limit:
            raise ValueError(f"Context overflow for {plan['source_id']}; input left intact")
    with (out / "results.jsonl").open("w") as f:
        for index, plan in enumerate(plans):
            runner.torch.manual_seed(config["seed"] + plan["repeat"])
            views, states = evaluate(runner, args.model, plan["messages"], plan["mapping"], budget)
            for mapping, result in views:
                row = {k: v for k, v in plan.items() if k != "messages"}
                row.update(result, model=args.model, dataset=args.dataset, mapping=mapping,
                           model_revision=runner.revision, run_index=index)
                if states is not None and args.save_states:
                    row["states_file"] = f"states_{index:06d}.pt"
                    runner.torch.save(states, out / row["states_file"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    (out / "complete.json").write_text(json.dumps({"completed_calls": len(plans)}))
    from refusal_cues.screen_report import summarize
    summary = summarize(read_jsonl(out / "results.jsonl"), config)
    summary["run_complete"] = True
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", choices=["jailjudge", "wildguard", "synthetic"], required=True)
    parser.add_argument("--model", choices=REVISIONS, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--revision", help="Override the pinned model revision")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-states", action="store_true")
    run_screen(parser.parse_args())


if __name__ == "__main__":
    main()
