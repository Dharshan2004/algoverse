"""BeaverTails pool loading for the interpretability workstream.

Loads Sabika's prepared manifests from a pinned Hugging Face commit, adds IDs and
hashes, picks one response per judge-cue family with the plan's fingerprint rule,
and checks overlap with the plan's duplicate rule (Datasets & labels, Section 2,
"Exact grouping, fingerprint and split rules").

The benign pool now comes from WildGuardTest (the team's decision), loaded from Sabika's
WildGuard preparation CSVs. The original BeaverTails benign files are superseded and skipped.
Pool selection itself is Sabika's work and is not changed here.

Typical use in Colab:
    from refusal_cues.directions import pools
    paths = pools.download_manifests()                     # BeaverTails unsafe-leaning + judge-cue
    data, report = pools.load_pools(paths)
    wg, wg_report = pools.load_wildguard_benign(WG_FOLDER) # folder with the three benign CSVs
    data.update(wg)
    print(report, wg_report)
    print(pools.check_counts(data))
    print(pools.overlap_summary(pools.find_duplicates(pools.prompt_groups(data))))
    pools.write_manifest(data, "/content/pools_manifest.jsonl")
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict

REPO_ID = "Sabka4/beavertails-preparation"
REVISION = "433a9d15bf6d347da2c865cb17ae1e8b715ba381"   # commit that created the manifests
SEED = 20261004
JACCARD = 0.90
SHINGLE = 5

# manifest file -> (role, class, source)
MANIFESTS = {
    "evaluation_estimation_candidate.jsonl":      ("estimation", "unsafe_leaning", "BeaverTails-Evaluation"),
    "evaluation_layer_selection_candidate.jsonl": ("layer_selection", "unsafe_leaning", "BeaverTails-Evaluation"),
    "evaluation_validation_candidate.jsonl":      ("validation", "unsafe_leaning", "BeaverTails-Evaluation"),
    "training_benign_estimation.jsonl":           ("estimation", "benign_candidate", "BeaverTails 330k_train"),
    "training_benign_layer_selection.jsonl":      ("layer_selection", "benign_candidate", "BeaverTails 330k_train"),
    "training_benign_validation.jsonl":           ("validation", "benign_candidate", "BeaverTails 330k_train"),
    "judging_source_safe.jsonl":                  ("judge_cue", "source_safe", "BeaverTails 330k_train"),
    "judging_source_unsafe.jsonl":                ("judge_cue", "source_unsafe", "BeaverTails 330k_train"),
}

# Planned sizes (Learn the refusal signal, Section 1, and the instructions' pool table)
EXPECTED = {
    "unsafe_leaning:estimation": 200, "unsafe_leaning:layer_selection": 50, "unsafe_leaning:validation": 100,
    "benign_candidate:estimation": 200, "benign_candidate:layer_selection": 50, "benign_candidate:validation": 100,
    "source_safe:judge_cue": 200, "source_unsafe:judge_cue": 200,
}

# The BeaverTails benign files were replaced by WildGuardTest benign prompts
SUPERSEDED = {"training_benign_estimation.jsonl", "training_benign_layer_selection.jsonl",
              "training_benign_validation.jsonl"}

# Sabika's WildGuard benign files -> (role, value of her "pool" column)
WILDGUARD_BENIGN = {
    "wildguard_benign_direction_estimation.csv": ("estimation", "direction_estimation"),
    "wildguard_benign_layer_selection.csv":      ("layer_selection", "layer_selection"),
    "wildguard_benign_held_out_validation.csv":  ("validation", "held_out_validation"),
}
WILDGUARD_SOURCE = "WildGuardTest (allenai/wildguardmix, wildguardtest, test)"
WILDGUARD_MANIFEST = "wildguard_reallocation_manifest.json"


# ---------------------------------------------------------------------------
# Text rules from the plan
# ---------------------------------------------------------------------------

def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text):
    """Matching only: NFKC, case-folding, outer-whitespace removal, whitespace collapse.
    Never applied to the text the model actually sees."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold().strip())


def shingles(text, n=SHINGLE):
    """Sets of five consecutive lowercase word tokens. Empty for texts under five tokens,
    which then use exact matching only."""
    words = re.findall(r"\w+", normalize(text))
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def fingerprint(revision, source_id, response_sha256, seed=SEED):
    """The plan's selection fingerprint: SHA-256 of seed|source_revision|source_id|response_sha256."""
    return sha256(f"{seed}|{revision}|{source_id}|{response_sha256}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def download_manifests(local_dir="/content/beavertails", revision=REVISION, include_superseded=False):
    """Download the BeaverTails manifests from the pinned commit. Returns {file name: local path}.
    The superseded BeaverTails benign files are skipped unless asked for."""
    from huggingface_hub import hf_hub_download
    return {name: hf_hub_download(REPO_ID, f"manifests/{name}", repo_type="dataset",
                                  revision=revision, local_dir=local_dir)
            for name in MANIFESTS if include_superseded or name not in SUPERSEDED}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _field(record, names, what, where):
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"{where}: no {what} field (looked for {names}). Fields present: {sorted(record)}")


def load_pools(paths, revision=REVISION, include_superseded=False):
    """Read every manifest, add IDs and hashes, and pick one response per judge-cue family.

    Returns (data, report). data maps "class:role" to a list of records. Each record keeps
    its text plus prompt_id, group_id, prompt_sha256, source_id, role, class, source and
    manifest revision. Judge-cue records also carry response_id, response_sha256,
    source_label and the selection fingerprint.
    """
    data, report = defaultdict(list), {}
    for name, (role, cls, source) in MANIFESTS.items():
        if name in SUPERSEDED and not include_superseded:
            continue
        rows = _read_jsonl(paths[name])
        records = []
        for line_no, raw in enumerate(rows):
            prompt = _field(raw, ("prompt", "question", "user_prompt"), "prompt", name)
            rec = {
                "prompt": prompt,
                "prompt_sha256": sha256(prompt),
                "prompt_id": "bt-" + sha256(prompt)[:16],
                "group_id": "g-" + sha256(normalize(prompt))[:16],
                "source_id": f"{name}:{line_no}",
                "role": role,
                "class": cls,
                "source_dataset": source,
                "manifest_repo": REPO_ID,
                "manifest_revision": revision,
            }
            if "category" in raw:
                rec["category"] = raw["category"]
            if role == "judge_cue":
                response = _field(raw, ("response", "answer"), "response", name)
                rec.update(response=response,
                           response_sha256=sha256(response),
                           response_id=rec["source_id"],
                           source_label=raw.get("is_safe"))
                rec["fingerprint"] = fingerprint(revision, rec["source_id"], rec["response_sha256"])
            records.append(rec)

        key = f"{cls}:{role}"
        if role == "judge_cue":
            families = defaultdict(list)
            for rec in records:
                families[rec["group_id"]].append(rec)
            chosen = [min(fam, key=lambda r: r["fingerprint"]) for fam in families.values()]
            chosen.sort(key=lambda r: r["group_id"])
            expected_label = cls == "source_safe"
            mismatched = sum(1 for r in records
                             if r["source_label"] is not None and r["source_label"] != expected_label)
            report[key] = {"records": len(records), "families": len(families),
                           "chosen": len(chosen), "label_mismatches": mismatched}
            data[key] = chosen
        else:
            dupes = len(records) - len({r["group_id"] for r in records})
            report[key] = {"records": len(records), "exact_duplicates_within_file": dupes}
            data[key] = records
    return dict(data), report


def _read_csv(path):
    import csv
    csv.field_size_limit(10 ** 9)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_wildguard_benign(folder):
    """Load Sabika's three WildGuard benign CSVs as the benign-candidate pools.

    Returns (data, report). data has "benign_candidate:estimation", ":layer_selection" and
    ":validation", with the same record fields as the BeaverTails pools plus source_group_id
    (Sabika's question_group_id), adversarial and prompt_harm_label. If the reallocation
    manifest is in the same folder, its selected group IDs are checked against the files.
    """
    from pathlib import Path
    folder = Path(folder)
    data, report, seen_rows = {}, {}, set()
    for name, (role, pool_value) in WILDGUARD_BENIGN.items():
        rows = _read_csv(folder / name)
        problems = Counter()
        records = []
        for raw in rows:
            prompt = raw.get("prompt") or ""
            if not prompt.strip():
                problems["blank_prompt"] += 1
                continue
            if raw.get("pool") and raw["pool"] != pool_value:
                problems["wrong_pool_value"] += 1
            if str(raw.get("prompt_harm_label", "")).lower() != "unharmful":
                problems["not_unharmful"] += 1
            if raw["source_row_id"] in seen_rows:
                problems["repeated_source_row"] += 1
            seen_rows.add(raw["source_row_id"])
            records.append({
                "prompt": prompt,
                "prompt_sha256": sha256(prompt),
                "prompt_id": f"wg-{raw['source_row_id']}",
                "group_id": "g-" + sha256(normalize(prompt))[:16],
                "source_group_id": raw.get("question_group_id"),
                "source_id": f"wildguardtest:{raw['source_row_id']}",
                "role": role,
                "class": "benign_candidate",
                "source_dataset": WILDGUARD_SOURCE,
                "adversarial": str(raw.get("adversarial")).strip().lower() == "true",
                "prompt_harm_label": raw.get("prompt_harm_label"),
            })
        key = f"benign_candidate:{role}"
        data[key] = records
        report[key] = {"records": len(records),
                       "adversarial": sum(r["adversarial"] for r in records),
                       "problems": dict(problems) or "none"}

    manifest_path = folder / WILDGUARD_MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        selected = set(manifest["benign_generation"]["selected_question_group_ids"])
        loaded = {r["source_group_id"] for recs in data.values() for r in recs}
        seed = manifest.get("random_seed")
        for recs in data.values():
            for r in recs:
                r["selection_seed"] = seed
        report["manifest"] = {"seed": seed, "group_ids_match_files": selected == loaded}
    else:
        report["manifest"] = "not found in folder (group IDs not cross-checked)"
    return data, report


def texts_from_csv(path, field="prompt"):
    """Pull one text field out of a CSV, for example WildGuard's reader-validation prompts."""
    return [r[field] for r in _read_csv(path) if r.get(field)]


def check_counts(data):
    """Compare loaded pool sizes with the planned sizes. Returns a list of problems."""
    problems = []
    for key, n in EXPECTED.items():
        got = len(data.get(key, []))
        if got != n:
            problems.append(f"{key}: expected {n}, got {got}")
    return problems or ["all pool sizes match the plan"]


# ---------------------------------------------------------------------------
# Overlap checks
# ---------------------------------------------------------------------------

def prompt_groups(data, external=None):
    """Collect prompt texts per pool, plus optional external pools {name: [texts]}."""
    groups = {key: [r["prompt"] for r in recs] for key, recs in data.items()}
    for name, texts in (external or {}).items():
        groups[f"external:{name}"] = list(texts)
    return groups


def texts_from_jsonl(path, field):
    """Pull one text field out of a JSONL file, for example an external pool's prompts."""
    return [str(r[field]) for r in _read_jsonl(path) if r.get(field)]


def find_duplicates(groups):
    """Find duplicate pairs with the plan's rule: exact normalized match, or Jaccard of
    five-word shingles at least 0.90. Returns a list of (group_a, i, group_b, j, kind).
    Pairs inside one group are included, since a pool should not contain two members
    of the same family either."""
    items = [(g, i, text) for g, texts in groups.items() for i, text in enumerate(texts)]
    pairs = set()

    exact = defaultdict(list)
    for k, (g, i, text) in enumerate(items):
        exact[normalize(text)].append(k)
    for ks in exact.values():
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                pairs.add((ks[x], ks[y], "exact"))

    sh = [shingles(text) for _, _, text in items]
    index = defaultdict(list)
    for k, s in enumerate(sh):
        for piece in s:
            index[piece].append(k)
    for k, s in enumerate(sh):
        if not s:
            continue
        shared = Counter(j for piece in s for j in index[piece] if j > k)
        for j, n in shared.items():
            if n / (len(s) + len(sh[j]) - n) >= JACCARD:
                pairs.add((k, j, "near"))

    seen, out = set(), []
    for k, j, kind in sorted(pairs, key=lambda p: (p[0], p[1], p[2] != "exact")):
        if (k, j) in seen:
            continue
        seen.add((k, j))
        (ga, ia, _), (gb, ib, _) = items[k], items[j]
        out.append((ga, ia, gb, ib, kind))
    return out


def overlap_summary(pairs):
    """Count duplicate pairs per pool combination. Empty means no overlap found."""
    counts = Counter(tuple(sorted((ga, gb))) for ga, _, gb, _, _ in pairs)
    return {f"{a} <-> {b}": n for (a, b), n in sorted(counts.items())} or {"no duplicates": 0}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

ID_FIELDS = ("prompt_id", "group_id", "prompt_sha256", "source_id", "role", "class",
             "source_dataset", "manifest_repo", "manifest_revision", "category",
             "response_id", "response_sha256", "source_label", "fingerprint",
             "source_group_id", "adversarial", "prompt_harm_label", "selection_seed")


def write_manifest(data, path, include_text=False):
    """One JSONL line per record with its IDs, hashes and role. Text-free by default,
    so it can go into the repository."""
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(data):
            for rec in data[key]:
                row = {k: rec[k] for k in ID_FIELDS if k in rec}
                if include_text:
                    row["prompt"] = rec["prompt"]
                    if "response" in rec:
                        row["response"] = rec["response"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Self test (no network, no GPU)
# ---------------------------------------------------------------------------

def self_test():
    import os
    import tempfile

    # Text rules
    assert normalize("  Ｈｅｌｌｏ   WORLD ") == "hello world"
    a = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    b = a + " twentyone"
    assert jaccard(shingles(a), shingles(b)) >= JACCARD
    assert shingles("too short text") == set()

    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for name, (role, cls, _) in MANIFESTS.items():
            rows = []
            if role == "judge_cue":
                for fam in range(3):
                    for resp in range(4):
                        rows.append({"prompt": f"{cls} family {fam} question here",
                                     "response": f"answer {resp} for {fam}",
                                     "is_safe": cls == "source_safe"})
            else:
                rows = [{"prompt": f"{cls} {role} prompt number {i}", "category": "x"} for i in range(3)]
            path = os.path.join(tmp, name)
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")
            paths[name] = path

        data, report = load_pools(paths, revision="test")
        # One response per family, chosen by the smallest fingerprint, deterministically
        safe = data["source_safe:judge_cue"]
        assert len(safe) == 3 and report["source_safe:judge_cue"]["records"] == 12
        again, _ = load_pools(paths, revision="test")
        assert [r["response_id"] for r in safe] == [r["response_id"] for r in again["source_safe:judge_cue"]]
        for rec in safe:
            fam = [r for r in again["source_safe:judge_cue"] if r["group_id"] == rec["group_id"]]
            assert fam and rec["fingerprint"] == fingerprint("test", rec["source_id"], rec["response_sha256"])
        assert report["source_safe:judge_cue"]["label_mismatches"] == 0
        assert all(len(r["prompt_id"]) == 19 and r["role"] for recs in data.values() for r in recs)
        assert check_counts(data)[0].startswith("unsafe_leaning:estimation")   # fake sizes differ

        # Overlap: plant one exact and one near duplicate in an external pool
        groups = prompt_groups(data, external={
            "jailjudge": ["UNSAFE_LEANING ESTIMATION prompt number 1",          # exact after normalizing
                          a, "something unrelated"],
            "other": [b]})                                                     # near duplicate of a
        pairs = find_duplicates(groups)
        kinds = {(ga, gb, kind) for ga, _, gb, _, kind in pairs}
        assert ("unsafe_leaning:estimation", "external:jailjudge", "exact") in kinds
        assert ("external:jailjudge", "external:other", "near") in kinds
        assert len(pairs) == 2

        # Manifest is text-free by default
        out = write_manifest(data, os.path.join(tmp, "manifest.jsonl"))
        rows = _read_jsonl(out)
        assert len(rows) == sum(len(v) for v in data.values())
        assert all("prompt" not in r and "response" not in r for r in rows)
    # Superseded BeaverTails benign files are skipped by default, loadable on request
    assert not any(k.startswith("benign_candidate") for k in data)
    with tempfile.TemporaryDirectory() as tmp2:
        p2 = {}
        for name, (role, cls, _) in MANIFESTS.items():
            rows = ([{"prompt": f"{cls} fam {f}", "response": "r", "is_safe": cls == "source_safe"}
                     for f in range(2)] if role == "judge_cue"
                    else [{"prompt": f"{cls} {role} {i}"} for i in range(2)])
            p2[name] = os.path.join(tmp2, name)
            open(p2[name], "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
        old_data, _ = load_pools(p2, revision="test", include_superseded=True)
        assert "benign_candidate:estimation" in old_data

    # WildGuard benign CSVs
    import csv
    with tempfile.TemporaryDirectory() as wg_dir:
        fields = ["source_row_id", "prompt", "adversarial", "prompt_harm_label", "question_group_id", "pool"]
        groups, row_id = [], 0
        for name, (role, pool_value) in WILDGUARD_BENIGN.items():
            with open(os.path.join(wg_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for i in range(4):
                    gid = sha256(f"g{row_id}")
                    groups.append(gid)
                    w.writerow({"source_row_id": row_id,
                                "prompt": f"Plain harmless question {row_id}, with a comma\nand a newline",
                                "adversarial": "True" if i % 2 else "False",
                                "prompt_harm_label": "unharmful", "question_group_id": gid,
                                "pool": pool_value})
                    row_id += 1
        json.dump({"random_seed": 20261031,
                   "benign_generation": {"selected_question_group_ids": sorted(groups)}},
                  open(os.path.join(wg_dir, WILDGUARD_MANIFEST), "w"))
        wg, wg_report = load_wildguard_benign(wg_dir)
        est = wg["benign_candidate:estimation"]
        assert len(est) == 4 and est[0]["prompt_id"] == "wg-0" and est[0]["role"] == "estimation"
        assert [r["adversarial"] for r in est] == [False, True, False, True]
        assert est[0]["prompt"].endswith("and a newline") and est[0]["selection_seed"] == 20261031
        assert wg_report["manifest"]["group_ids_match_files"] is True
        assert all(v["problems"] == "none" for k, v in wg_report.items() if k != "manifest")

        # Merging gives the full set of pools, and overlap checks cover WildGuard too
        merged = {**data, **wg}
        merged["benign_candidate:validation"].append(dict(merged["unsafe_leaning:estimation"][0]))
        dup = find_duplicates(prompt_groups(merged))
        assert any({ga, gb} == {"unsafe_leaning:estimation", "benign_candidate:validation"}
                   for ga, _, gb, _, _ in dup)
        rows = _read_jsonl(write_manifest(merged, os.path.join(wg_dir, "m.jsonl")))
        assert any(r.get("adversarial") is True for r in rows) and all("prompt" not in r for r in rows)

        # Problems in the files are reported, not hidden
        bad = os.path.join(wg_dir, "bad")
        os.mkdir(bad)
        for name, (role, pool_value) in WILDGUARD_BENIGN.items():
            with open(os.path.join(bad, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow({"source_row_id": 1, "prompt": "x", "adversarial": "False",
                            "prompt_harm_label": "harmful", "question_group_id": "g", "pool": "wrong"})
        _, bad_report = load_wildguard_benign(bad)
        est_p = bad_report["benign_candidate:estimation"]["problems"]
        assert est_p["not_unharmful"] == 1 and est_p["wrong_pool_value"] == 1
        assert bad_report["benign_candidate:layer_selection"]["problems"]["repeated_source_row"] == 1
        assert isinstance(bad_report["manifest"], str)
    print("pools.self_test: all checks passed")


if __name__ == "__main__":
    self_test()
