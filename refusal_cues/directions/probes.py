"""Probe-building helpers for the refusal-wording study (interpretability workstream).

Builds on Runner from inference.py without modifying it. Development and practice
use only: directions built on a T4/FP16 runner are flagged as practice, because the
frozen probes must come from the same BF16 setup used for the edits.

Typical use in Colab (with `runner` already loaded):
    from refusal_cues.directions import probes; probes.self_test()
    recs_u = probes.capture_states(runner, unsafe_prompts)
    recs_b = probes.capture_states(runner, benign_prompts)
    rG = {L: probes.build_direction(recs_u, recs_b, L) for L in runner.depths}

Full answers for refusal behavior (native, then edited conditions), resumable:
    probes.run_condition(runner, records, "native", "answers.jsonl", states_path="native_states.pt")
    native = torch.load("native_states.pt")
    probes.run_condition(runner, records, "remove", "answers.jsonl", layer=L, rG=r)
    probes.run_condition(runner, records, "random_remove:6201", "answers.jsonl", layer=L, rG=r,
                         randoms=probes.random_directions(r), native_states=native)

Judge-cue direction from a saved judging run (e.g. the smoke test):
    rows = probes.load_judging_rows("/content/ministral-smoke")
    pairs = probes.pair_judging_rows(rows)
    rJ = {L: probes.build_judge_cue_direction(pairs, L) for L in runner.depths}
"""
import json
import re
from pathlib import Path

import torch

from refusal_cues.protocol import MAPPINGS, sha

SYSTEM = "You are a helpful assistant."
SEEDS = (6201, 6202, 6203, 6204, 6205)


# ---------------------------------------------------------------------------
# Prompts and token positions
# ---------------------------------------------------------------------------

def make_turns(prompt, system=SYSTEM):
    """Fixed system message plus one user request. The fixed system message stops
    Ministral's template from inserting its own dated default prompt."""
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def _ids(runner, turns):
    return runner.encode(turns)["input_ids"][0].tolist()


def post_instruction_length(runner, system=SYSTEM):
    """Number of template tokens after the user's text (e.g. [/INST] for Ministral).

    The template's closing tokens are the same for every prompt, so encoding two
    different user messages and counting their shared ending gives that number.
    """
    a = _ids(runner, make_turns("alpha", system))
    b = _ids(runner, make_turns("omega seven", system))
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    if n == 0:
        raise RuntimeError("Could not find the template's closing tokens")
    return n


def _letters(text):
    return re.sub(r"\W", "", text)


def _check_user_position(runner, ids, t_inst, prompt):
    """Decode the tokens ending at t_inst and confirm they end the user's text."""
    tail = runner.tokenizer.decode(ids[max(0, t_inst - 3):t_inst + 1])
    if not _letters(tail) or _letters(tail)[-3:] != _letters(prompt)[-3:]:
        raise RuntimeError(f"User-content position looks wrong: tail={tail!r}, "
                           f"prompt ends with {prompt[-20:]!r}")
    return tail


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_states(runner, prompts, system=SYSTEM, layers=None):
    """Run each prompt once and return its states at two positions.

    'final' is the final template token, read by the runner itself (used for rG).
    'user'  is the last token of the user's text, read by an extra hook (used for rH).
    Both come from the same prompt pass, so they are directly comparable.
    """
    layers = list(layers or runner.depths)
    if not set(layers) <= set(runner.depths):
        raise ValueError("Layers must be among runner.depths")
    n_post = post_instruction_length(runner, system)
    records = []
    for prompt in prompts:
        turns = make_turns(prompt, system)
        ids = _ids(runner, turns)
        length = len(ids)
        t_inst = length - n_post - 1
        tail = _check_user_position(runner, ids, t_inst, prompt)
        user, fired, handles = {}, set(), []

        def make_hook(index):
            def hook(module, args, output):
                if index in fired:          # only the prompt pass, like the runner
                    return None
                fired.add(index)
                x = output[0] if isinstance(output, tuple) else output
                if x.ndim != 3 or x.shape[1] != length:
                    raise RuntimeError("Expected the complete prompt pass")
                user[index] = x[0, t_inst].float().detach().cpu()
                return None                 # read only, never edit
            return hook

        try:
            for index in layers:
                handles.append(runner.layers[index].register_forward_hook(make_hook(index)))
            result, final = runner.run(turns, max_new_tokens=1)
        finally:
            for handle in handles:
                handle.remove()
        if set(user) != set(layers):
            raise RuntimeError("Not every user-content hook fired")
        records.append({
            "prompt": prompt,
            "final": {i: final[i] for i in layers},
            "user": user,
            "t_inst": t_inst,
            "length": length,
            "user_tail": tail,
            "input_ids_sha256": result["input_ids_sha256"],
        })
    return records


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

def build_direction(group_a, group_b, layer, position="final"):
    """Difference-in-means direction: mean(group_a) - mean(group_b), scaled to length one.

    Returns (unit_direction, length). The length is kept because it becomes the
    addition strength for the 'add' edit.
    """
    a = torch.stack([s[position][layer] for s in group_a]).float()
    b = torch.stack([s[position][layer] for s in group_b]).float()
    d = a.mean(0) - b.mean(0)
    length = d.norm()
    if not torch.isfinite(length) or length == 0:
        raise ValueError("Degenerate direction")
    return d / length, float(length)


def random_directions(r, seeds=SEEDS):
    """Five seeded Gaussian directions, each with its overlap with r removed and
    scaled to length one. Same seeds give the same directions on the same PyTorch version."""
    r = r.float() / r.float().norm()
    out = {}
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        u = torch.randn(r.shape, generator=g)
        u = u - (u @ r) * r
        out[seed] = u / u.norm()
    return out


# ---------------------------------------------------------------------------
# Judge-cue direction (rJ)
# ---------------------------------------------------------------------------

def load_judging_rows(run_dir):
    """Load results.jsonl from a judging run and attach each row's saved states."""
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()
            if line.strip()]
    for row in rows:
        row["states"] = torch.load(run_dir / row["states_file"])
    return rows


def pair_judging_rows(rows, cued="R", original="B", allowed_partitions=None):
    """Match each cued judging row to its original by family and mapping.

    Only unedited rows are used. A pair must share the model revision and the exact
    response body (body_sha256), so the only difference is the added prefix.
    For the real rJ, pass allowed_partitions with the judge-cue pool's partition name,
    so development and final-test data can never leak in.
    """
    index = {}
    for row in rows:
        if row.get("edit"):
            continue
        if allowed_partitions is not None and row["study_partition"] not in allowed_partitions:
            raise ValueError(f"Row {row['source_id']} is from partition {row['study_partition']!r}")
        key = (row["source_id"], row["mapping"], row["variant"])
        if key in index:
            raise ValueError(f"Duplicate unedited row for {key}")
        index[key] = row
    pairs = []
    for (source_id, mapping, name), cued_row in sorted(index.items()):
        if name != cued:
            continue
        orig = index.get((source_id, mapping, original))
        if orig is None:
            raise ValueError(f"No {original} row for {source_id} under {mapping}")
        if orig["model_revision"] != cued_row["model_revision"]:
            raise ValueError(f"Model revision differs for {source_id} under {mapping}")
        if orig["body_sha256"] != cued_row["body_sha256"]:
            raise ValueError(f"Response body differs for {source_id} under {mapping}")
        pairs.append((orig, cued_row))
    if not pairs:
        raise ValueError(f"No {cued}/{original} pairs found")
    return pairs


def build_judge_cue_direction(pairs, layer, require_both_mappings=True):
    """rJ: average (cued - original) state difference, scaled to length one.

    Differences are averaged over the two mappings within each family first, then
    across families, so every family counts equally.
    Returns (unit_direction, length, n_families). The length is rJ's addition strength.
    """
    by_family = {}
    for orig, cued_row in pairs:
        diff = cued_row["states"][layer].float() - orig["states"][layer].float()
        by_family.setdefault(orig["source_id"], {})[orig["mapping"]] = diff
    family_means = []
    for source_id, per_mapping in sorted(by_family.items()):
        if require_both_mappings and set(per_mapping) != set(MAPPINGS):
            raise ValueError(f"{source_id} is missing a mapping: has {sorted(per_mapping)}")
        family_means.append(torch.stack(list(per_mapping.values())).mean(0))
    d = torch.stack(family_means).mean(0)
    length = d.norm()
    if not torch.isfinite(length) or length == 0:
        raise ValueError("Degenerate direction")
    return d / length, float(length), len(family_means)


def cosine(a, b):
    """Cosine similarity between two directions (descriptive only)."""
    a, b = a.float(), b.float()
    return float(a @ b / (a.norm() * b.norm()))


# ---------------------------------------------------------------------------
# Readouts and edit amounts
# ---------------------------------------------------------------------------

def projections(records, r, layer, position="final"):
    """Each record's coordinate along r: one number per prompt."""
    return torch.stack([s[position][layer].float() @ r.float() for s in records])


def cue_shift(h_original, h_cued, r):
    """How far the cued coordinate must move to match the original (delta_G)."""
    return float(h_original.float() @ r.float() - h_cued.float() @ r.float())


def matched_removal_amount(h, r):
    """'add' amount for a random direction that moves h exactly as far as removing r."""
    return -float(h.float() @ r.float())


# ---------------------------------------------------------------------------
# Answering path (full generation, used to measure refusal behavior)
# ---------------------------------------------------------------------------

MAX_ANSWER_TOKENS = 256
RECORD_KEYS = ("prompt_id", "group_id", "role", "class", "category", "source_id")


def edit_for(condition, layer=None, rG=None, alpha=None, randoms=None, h=None):
    """Edit for a named condition. All edits act at one layer, final prompt token, prefill only.

    native               no edit
    remove               remove rG's component
    add                  add rG with strength alpha (the saved difference length)
    random_remove:<seed> random direction, moved exactly as far as removing rG would move h
    random_add:<seed>    random direction, added with the same strength alpha
    """
    if condition == "native":
        return None
    if condition == "remove":
        return {"layer": layer, "direction": rG, "mode": "remove"}
    if condition == "add":
        return {"layer": layer, "direction": rG, "mode": "add", "amount": float(alpha)}
    kind, _, seed = condition.partition(":")
    if kind in ("random_remove", "random_add"):
        u = randoms[int(seed)]
        if kind == "random_remove":
            if h is None:
                raise ValueError("random_remove needs the native prompt-pass state h at this layer")
            return {"layer": layer, "direction": u, "mode": "add", "amount": matched_removal_amount(h, rG)}
        return {"layer": layer, "direction": u, "mode": "add", "amount": float(alpha)}
    raise ValueError(f"Unknown condition {condition!r}")


def edit_audit(edit, h):
    """What an edit does to the edited direction's coordinate, from the pre-edit state h:
    coordinate before, signed change, coordinate after, displacement and direction hash."""
    d = torch.as_tensor(edit["direction"], dtype=torch.float32).cpu()
    h = h.float().cpu()
    before = float(h @ d)
    if edit["mode"] == "remove":
        change = -before
    elif edit["mode"] == "add":
        change = float(edit["amount"])
    elif edit["mode"] == "replace":
        change = float((torch.as_tensor(edit["donor"], dtype=torch.float32).cpu() - h) @ d)
    else:
        change = 0.0
    return {"edit_mode": edit["mode"], "edit_layer": edit["layer"],
            "edit_direction_sha256": direction_hash(d),
            "edit_coordinate_before": before, "edit_signed_change": change,
            "edit_coordinate_after": before + change, "edit_displacement": abs(change),
            "edit_donor_ref": edit.get("donor_ref")}


def answer(runner, record, edit=None, condition="native", system=SYSTEM,
           max_new_tokens=MAX_ANSWER_TOKENS):
    """Generate one full answer (up to 256 new tokens) and return (log row, prompt-pass states).

    The row keeps the raw output, output IDs, the answer text with only a recognized empty
    Gemma thought wrapper removed, and empty, length-cap and end-of-answer flags. A length
    cap alone does not make an answer broken. The refusal reader decides that later.
    """
    prompt = record["prompt"]
    result, states = runner.run(make_turns(prompt, system), max_new_tokens=max_new_tokens, edit=edit)
    text = result["parsed_text"]
    row = {k: record[k] for k in RECORD_KEYS if k in record}
    row.update({
        "status": "ok",
        "condition": condition,
        "answer_text": text,
        # Thought-channel tags left after the empty wrapper is removed mean the model wrote
        # real thinking text. Flag it for review rather than letting the reader see tags.
        "channel_tags_left": any(t in text for t in ("<|channel>", "<channel|>")),
        "raw_output": result["raw_output"],
        "output_ids": result["output_ids"],
        "output_normalization": result.get("output_normalization", "none"),
        "empty": not text.strip(),
        "length_cap": result["length_cap"],
        "end_of_answer": result["end_of_answer"],
        "n_output_tokens": len(result["output_ids"]),
        "input_ids_sha256": result["input_ids_sha256"],
        "runner_edit_log": result.get("edit", {}),
        "seconds": result["seconds"],
        "model_id": runner.model_id,
        "revision": runner.revision,
        "compute_dtype": runner.meta.get("compute_dtype"),
        "gpu": runner.meta.get("gpu"),
        "max_new_tokens": max_new_tokens,
        "system_prompt_sha256": sha(system),
    })
    if edit is not None:
        row.update(edit_audit(edit, states[edit["layer"]]))   # the runner saves the edit layer pre-edit
    return row, states


def load_answers(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_condition(runner, records, condition, out_path, layer=None, rG=None, alpha=None,
                  randoms=None, native_states=None, states_path=None, system=SYSTEM,
                  max_new_tokens=MAX_ANSWER_TOKENS):
    """Answer every record under one condition, appending one JSONL row per answer.

    Safe to rerun after a disconnect: records already answered under the same condition
    and layer are skipped. With states_path, each prompt's prompt-pass states are saved
    by prompt_id. Save them on the native run, because the matched random removals need
    each prompt's native state (pass them back in as native_states).
    """
    out_path = Path(out_path)
    done = {r["prompt_id"] for r in load_answers(out_path)
            if r["condition"] == condition and r.get("layer") == layer}
    saved = torch.load(states_path) if states_path and Path(states_path).exists() else {}
    unsaved = 0
    with out_path.open("a", encoding="utf-8") as f:
        for rec in records:
            if rec["prompt_id"] in done:
                continue
            h = None
            if native_states is not None and layer is not None:
                if rec["prompt_id"] not in native_states:
                    raise KeyError(f"No native state for {rec['prompt_id']}. Run the native condition first.")
                h = native_states[rec["prompt_id"]][layer]
            edit = edit_for(condition, layer, rG, alpha, randoms, h)
            try:
                row, states = answer(runner, rec, edit=edit, condition=condition,
                                     system=system, max_new_tokens=max_new_tokens)
            except ValueError as err:
                if "exceeds" not in str(err):
                    raise
                # Never truncate: log the prompt as skipped and move on
                row, states = {k: rec[k] for k in RECORD_KEYS if k in rec}, None
                row.update(status="input_too_long", condition=condition, error=str(err))
            row["layer"] = layer
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if states_path is not None and states is not None:
                saved[rec["prompt_id"]] = dict(states)
                unsaved += 1
                if unsaved >= 25:                     # save in batches, not after every prompt
                    torch.save(saved, states_path)
                    unsaved = 0
    if states_path is not None and unsaved:
        torch.save(saved, states_path)
    return out_path


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def direction_hash(r):
    """Same hash the runner logs for an edit, so a saved probe can be matched to the
    edits that used it."""
    return sha(json.dumps(r.float().cpu().tolist()))


def save_probes(path, runner, probes, note=""):
    """Save probes with everything needed to reproduce and audit them.

    probes: {name: {"direction": tensor, "layer": int, "position": "final" or "user",
                    "length": float, ...}}
    """
    practice = runner.meta.get("compute_dtype") != "bfloat16"
    payload = {
        "probes": {name: {**p, "direction": p["direction"].float().cpu(),
                          "sha256": direction_hash(p["direction"])}
                   for name, p in probes.items()},
        "model_id": runner.model_id,
        "revision": runner.revision,
        "compute_dtype": runner.meta.get("compute_dtype"),
        "decoder_path": runner.layer_path,
        "depths": list(runner.depths),
        "practice_only": practice,
        "note": note,
    }
    torch.save(payload, path)
    return payload


# ---------------------------------------------------------------------------
# GPU-free self test
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Word-level stand-in: one token per word or punctuation mark."""
    def __init__(self):
        self.vocab, self.words = {}, []

    def tok(self, text):
        ids = []
        for w in re.findall(r"\w+|[^\w\s]", text):
            if w not in self.vocab:
                self.vocab[w] = len(self.words)
                self.words.append(w)
            ids.append(self.vocab[w])
        return ids

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids)


class _FakeRunner:
    """Mimics the parts of Runner that capture_states uses."""
    def __init__(self, hidden=8, n_layers=8):
        self.tokenizer = _FakeTokenizer()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(hidden, hidden) for _ in range(n_layers))
        self.depths = [1, 3, 5]
        self.embed = torch.nn.Embedding(1000, hidden)
        self.model_id, self.revision, self.layer_path = "fake", "0", "layers"
        self.meta = {"compute_dtype": "float16"}

    def encode(self, turns):
        t = self.tokenizer.tok
        ids = t("<s> [SYS]") + t(turns[0]["content"]) + t("[/SYS] [INST]") \
            + t(turns[1]["content"]) + t("[/INST]")
        return {"input_ids": torch.tensor([ids])}

    def run(self, turns, max_new_tokens=1, edit=None):
        """Like Runner.run: saves states at the depths (and the edit layer) before editing,
        applies the edit at the final position once, and returns answer fields."""
        if "TOO_LONG" in turns[-1]["content"]:
            raise ValueError("Complete input exceeds 4096 tokens; never truncate the body")
        ids = self.encode(turns)["input_ids"]
        x, states, log = self.embed(ids), {}, {}
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i in self.depths or (edit and i == edit["layer"]):
                    states[i] = x[0, -1].float().clone()
                if edit and i == edit["layer"] and edit["mode"] != "identity":
                    h = x[0, -1].float()
                    r = torch.as_tensor(edit["direction"], dtype=torch.float32)
                    if edit["mode"] == "remove":
                        delta = -(h @ r)
                    elif edit["mode"] == "add":
                        delta = torch.tensor(float(edit["amount"]))
                    else:
                        delta = (torch.as_tensor(edit["donor"], dtype=torch.float32) - h) @ r
                    x = x.clone()
                    x[0, -1] = h + delta * r
                    log = {"mode": edit["mode"], "layer": i, "displacement": float(delta.abs()),
                           "direction_sha256": direction_hash(r)}
        n = min(4, max_new_tokens)
        text = "fake answer text" if max_new_tokens > 1 else ""
        return {"input_ids_sha256": "fake", "raw_output": text, "parsed_text": text,
                "output_ids": list(range(n)), "output_normalization": "none",
                "length_cap": n == max_new_tokens, "end_of_answer": n < max_new_tokens,
                "seconds": 0.0, "edit": log}, states


def self_test():
    torch.manual_seed(0)

    # Edit math, using the plan's toy examples
    h, r = torch.tensor([3., 5., 1.]), torch.tensor([1., 0., 0.])
    assert torch.allclose(h + matched_removal_amount(h, r) * r, torch.tensor([0., 5., 1.]))
    h_cued, h_orig, r2 = torch.tensor([4.5, 8.]), torch.tensor([2., 6.]), torch.tensor([1., 0.])
    assert torch.allclose(h_cued + cue_shift(h_orig, h_cued, r2) * r2, torch.tensor([2., 8.]))

    # Difference-in-means recovers a planted direction
    e = torch.nn.functional.normalize(torch.randn(64), dim=0)
    grp_a = [{"final": {0: torch.randn(64) + 3 * e}} for _ in range(200)]
    grp_b = [{"final": {0: torch.randn(64)}} for _ in range(200)]
    d, length = build_direction(grp_a, grp_b, 0)
    assert float(d @ e) > 0.9 and abs(float(d.norm()) - 1) < 1e-5 and length > 2

    # Random directions: unit length, orthogonal to r, reproducible
    rand = random_directions(d)
    again = random_directions(d)
    for seed, u in rand.items():
        assert abs(float(u.norm()) - 1) < 1e-5 and abs(float(u @ d)) < 1e-5
        assert torch.equal(u, again[seed])

    # Matched random move has the same size as removing r
    hv = torch.randn(64)
    u = rand[SEEDS[0]]
    moved = hv + matched_removal_amount(hv, d) * u
    removed = hv + matched_removal_amount(hv, d) * d
    assert abs(float((moved - hv).norm() - (removed - hv).norm())) < 1e-4

    # Projections
    assert torch.allclose(projections(grp_a[:2], d, 0),
                          torch.stack([grp_a[0]["final"][0] @ d, grp_a[1]["final"][0] @ d]))

    # Capture: user-content position and hooks, on a fake runner
    fake = _FakeRunner()
    assert post_instruction_length(fake) == len(fake.tokenizer.tok("[/INST]"))
    recs = capture_states(fake, ["Write a note please", "How do I bake bread ?"])
    for rec in recs:
        assert rec["user_tail"].split()[-1] == rec["prompt"].split()[-1]
        assert set(rec["final"]) == set(rec["user"]) == set(fake.depths)
        assert not torch.equal(rec["final"][3], rec["user"][3])

    # Judge-cue direction: pairing, provenance checks and a planted cue shift
    cue = torch.nn.functional.normalize(torch.randn(64), dim=0)
    rows = []
    for f in range(30):
        base = torch.randn(64)
        for mapping in MAPPINGS:
            for name, shift in (("B", 0.0), ("R", 2.0)):
                rows.append({"source_id": f"fam-{f}", "mapping": mapping, "variant": name,
                             "model_revision": "rev", "body_sha256": f"body-{f}",
                             "study_partition": "synthetic", "edit": {},
                             "states": {0: base + shift * cue + 0.1 * torch.randn(64)}})
    pairs = pair_judging_rows(rows)
    assert len(pairs) == 30 * len(MAPPINGS)
    rj, rj_length, n_fam = build_judge_cue_direction(pairs, 0)
    assert n_fam == 30 and cosine(rj, cue) > 0.95 and abs(rj_length - 2.0) < 0.3
    bad = [dict(r) for r in rows]
    bad[1]["body_sha256"] = "different"
    try:
        pair_judging_rows(bad)
        raise AssertionError("body mismatch was not caught")
    except ValueError:
        pass
    try:
        pair_judging_rows(rows, allowed_partitions={"judge_cue"})
        raise AssertionError("wrong partition was not caught")
    except ValueError:
        pass
    edited = rows + [{**rows[1], "edit": {"mode": "remove"}}]
    assert len(pair_judging_rows(edited)) == len(pairs)   # edited rows are ignored

    # Answering path: edits, audit, matched random size, logging and resume
    import tempfile
    fake = _FakeRunner()
    L = fake.depths[1]
    recs_a = [{"prompt_id": f"p{i}", "prompt": f"please answer question {i} now", "role": "validation",
               "class": "unsafe_leaning"} for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        out, spath = Path(tmp) / "answers.jsonl", Path(tmp) / "native_states.pt"
        run_condition(fake, recs_a, "native", out, states_path=spath)
        native_states = torch.load(spath)
        assert set(native_states) == {"p0", "p1", "p2"}
        rows = load_answers(out)
        assert len(rows) == 3 and not rows[0]["empty"] and not rows[0]["length_cap"]
        assert rows[0]["max_new_tokens"] == MAX_ANSWER_TOKENS and "edit_mode" not in rows[0]

        direction = torch.nn.functional.normalize(torch.randn(8), dim=0)
        rands = random_directions(direction)
        run_condition(fake, recs_a, "remove", out, layer=L, rG=direction)
        run_condition(fake, recs_a, f"random_remove:{SEEDS[0]}", out, layer=L, rG=direction,
                      randoms=rands, native_states=native_states)
        run_condition(fake, recs_a, "add", out, layer=L, rG=direction, alpha=2.5)
        rows = load_answers(out)
        assert len(rows) == 12
        rm = {r["prompt_id"]: r for r in rows if r["condition"] == "remove"}
        rr = {r["prompt_id"]: r for r in rows if r["condition"].startswith("random_remove")}
        for pid in rm:
            assert abs(rm[pid]["edit_coordinate_after"]) < 1e-5            # removal zeroes the coordinate
            assert abs(rm[pid]["edit_displacement"] - rr[pid]["edit_displacement"]) < 1e-5   # matched size
            assert abs(rm[pid]["edit_displacement"] - rm[pid]["runner_edit_log"]["displacement"]) < 1e-5
        add_rows = [r for r in rows if r["condition"] == "add"]
        assert all(abs(r["edit_signed_change"] - 2.5) < 1e-6 for r in add_rows)

        assert all(r["status"] == "ok" and r["channel_tags_left"] is False for r in rows)

        # Rerunning skips work already done
        run_condition(fake, recs_a, "remove", out, layer=L, rG=direction)
        assert len(load_answers(out)) == 12

        # An over-long prompt is logged as skipped, never truncated, and not retried
        long_rec = [{"prompt_id": "p_long", "prompt": "TOO_LONG prompt", "role": "validation"}]
        run_condition(fake, long_rec, "native", out, states_path=spath)
        run_condition(fake, long_rec, "native", out, states_path=spath)
        skipped = [r for r in load_answers(out) if r["prompt_id"] == "p_long"]
        assert len(skipped) == 1 and skipped[0]["status"] == "input_too_long"
        assert "p_long" not in torch.load(spath)
        try:
            edit_for("random_remove:6201", L, direction, randoms=rands)
            raise AssertionError("random_remove without h was not caught")
        except ValueError:
            pass

    # Hash matches the runner's convention
    assert direction_hash(r) == sha(json.dumps(r.cpu().tolist()))
    print("probes.self_test: all checks passed")


if __name__ == "__main__":
    self_test()
