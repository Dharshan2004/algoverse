"""Offline checks for the Qwen3Guard judge. Archive parity lives in units_full.py."""
import base64
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "src"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    judge = load("qwen3guard_judge")
    metrics = load("judge_metrics")
    spec = importlib.util.spec_from_file_location("qwen_activation_tools", SRC / "activation_tools.py")
    tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools)
    return judge, metrics, tools


def test_parser_views_and_guard_prompt_shape():
    from refusal_cues.guard_judges import guard_messages
    from refusal_cues.protocol import depths

    judge, _, tools = modules()
    cases = {
        "Safety: Safe\nCategories: None\nRefusal: Yes": ("SAFE", True, "Yes"),
        "Safety: Unsafe\nCategories: Violent, PII\nRefusal: No": ("UNSAFE", True, "No"),
        "Safety: Controversial\nCategories: Unethical Acts\nRefusal: No": ("CONTROVERSIAL", True, "No"),
        "Safety: Unsafe\nCategories: Violent": ("UNSAFE", False, None),
        "  Safety: Safe  \n": ("SAFE", False, None),
        "Safety: unsafe\nCategories: None\nRefusal: No": ("INVALID", False, "No"),
        "The response is Safety: Unsafe": ("INVALID", False, None),
        "Safety: Safe and sound\nCategories: None\nRefusal: No": ("INVALID", False, "No"),
        "": ("INVALID", False, None),
    }
    for text, expected in cases.items():
        parsed = judge.parse_output(text)
        assert (parsed["raw_label"], parsed["format_ok"], parsed["refusal"]) == expected
    assert judge.parse_output("Safety: Unsafe\nCategories: Violent, PII\nRefusal: No")["categories"] == ["Violent", "PII"]
    assert judge.view_verdict("CONTROVERSIAL", "controversial_unsafe") == "UNSAFE"
    assert judge.view_verdict("CONTROVERSIAL", "controversial_safe") == "SAFE"
    assert judge.view_verdict("INVALID", "controversial_unsafe") == "INVALID"
    turns = judge.guard_messages("request", "cue\n\nbody")
    assert turns == guard_messages("request", "cue\n\nbody", "qwen3guard")
    assert tools.build_turns("qwen3guard", "request", "cue\n\nbody", "native") == turns
    assert tools.build_turns("shieldgemma", "request", "body", "native") == guard_messages("request", "body", "shieldgemma")
    assert depths(36) == [8, 12, 16, 19, 23, 26]


def test_behavioral_plan_size():
    judge, _, tools = modules()
    records = [{"source_id": f"s{i}", "original_prompt": "p", "original_response": "b", "mapped_unsafe": i < 270}
               for i in range(470)]
    keys = judge.behavioral_keys(records)
    assert len(keys) == len(set(keys)) == 6580
    plan = tools.build_plan(["qwen3guard"], judge.CONDITIONS, records, 47)
    assert len(plan) == 70
    assert sum(len(unit["source_ids"]) for unit in plan) == 3290
    assert {unit["mapping"] for unit in plan} == {"native"}


def test_invalid_baseline_is_not_eligible():
    _, metrics, _ = modules()
    records = [
        {"source_id": "h1", "mapped_unsafe": True, "cluster_id": "c1"},
        {"source_id": "h2", "mapped_unsafe": True, "cluster_id": "c2"},
        {"source_id": "s1", "mapped_unsafe": False, "cluster_id": "c3"},
    ]
    conditions = metrics.CONDITIONS
    rows = []
    for record in records:
        for repeat in (0, 1):
            for condition in conditions:
                verdict = "UNSAFE" if record["mapped_unsafe"] else "SAFE"
                if record["source_id"] == "h1" and condition == "R":
                    verdict = "SAFE"
                rows.append({
                    "model": "llama_guard", "mapping": "native", "repeat": repeat,
                    "condition": condition, "source_id": record["source_id"],
                    "harmful": record["mapped_unsafe"], "verdict": verdict,
                })
    result = metrics.analyze(rows, records)
    lead = result["lead"][0]
    assert lead["eligible_harmful"] == 2
    assert lead["DFR5"]["numerator"] == 1
    assert lead["DFR5"]["denominator"] == 2
    flipped = next(row for row in rows if row["source_id"] == "h2" and row["condition"] == "B" and row["repeat"] == 0)
    flipped["verdict"] = "INVALID"
    again = metrics.analyze(rows, records)
    lead0 = next(item for item in again["lead"] if item["repeat"] == 0)
    assert lead0["eligible_harmful"] == 1
    assert lead0["invalid"] == 1
    assert again["integrity"]["invalid"] == 1
    assert len(metrics.markdown(result, "synthetic")) > 200


def test_notebook_code_cells_parse():
    notebook = json.loads((HERE / "qwen3guard_experiment.ipynb").read_text())
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code
    for index, source in enumerate(code):
        if source.startswith("%%writefile"):
            source = source.split("\n", 1)[1]
        compile(source, f"cell{index}", "exec")
    install = next(source for source in code if "guard_judges.py" in source and "b64decode" in source)
    imported = next(source for source in code if "from refusal_cues.guard_judges import guard_messages" in source)
    assert code.index(install) < code.index(imported)
    encoded = re.search(r"base64\.b64decode\('([^']+)'\)", install).group(1)
    repo_module = Path(__file__).resolve().parents[3] / "refusal_cues" / "guard_judges.py"
    assert base64.b64decode(encoded) == repo_module.read_bytes()


def test_resume_rejects_changed_records_or_budget(tmp_path):
    judge, _, _ = modules()
    records = [{"source_id": f"s{i}", "original_prompt": f"prompt {i}", "original_response": f"body {i}",
                "mapped_unsafe": i == 0} for i in range(4)]
    cue = {name: f"cue {name}" for name in ("N", "R", "R2", "paper_R1", "paper_R2", "paper_Rgreet")}

    class Runner:
        def __init__(self):
            self.torch = types.SimpleNamespace(manual_seed=lambda seed: None)
            self.meta = {"model_id": "fake", "revision": "0", "software": {}}

        def run(self, turns, budget):
            text = "Safety: Safe\nCategories: None\nRefusal: Yes"
            result = {"raw_output": text, "parsed_text": text, "output_ids": [1, 2],
                      "output_protocol": judge.OUTPUT_PROTOCOL, "budget_reached": False,
                      **judge.parse_output(text), "decision_step": 2, "logprob_safe": -1.0,
                      "logprob_unsafe": -2.0, "logprob_controversial": -3.0, "label_mass": 0.5,
                      "unsafe_margin": -1.0, "margin_controversial_unsafe": 0.1,
                      "margin_controversial_safe": -0.1, "input_tokens": 10,
                      "input_ids_sha256": "x", "seconds": 0.0}
            for view in judge.VIEWS:
                result[f"verdict_{view}"] = judge.view_verdict(result["raw_label"], view)
            return result

    judge.run_behavioral(tmp_path, records, cue, Runner, "rt", max_new=3, log=lambda message: None)
    folder = tmp_path / "behavioral" / "qwen3guard"
    saved_manifest = (folder / "manifest.json").read_text()
    saved_chunks = sorted(path.read_bytes() for path in (folder / "verdicts").glob("part-*.jsonl"))
    manifest = json.loads(saved_manifest)
    assert manifest["budget"] == judge.BUDGET
    assert manifest["input_sha256"] == judge.input_digest(records)

    changed = [dict(records[0], original_response="different body"), *records[1:]]
    with pytest.raises(RuntimeError, match="input records"):
        judge.run_behavioral(tmp_path, changed, cue, Runner, "rt", log=lambda message: None)
    with pytest.raises(RuntimeError, match="decoding budget"):
        judge.run_behavioral(tmp_path, records, cue, Runner, "rt", budget=1, log=lambda message: None)
    assert (folder / "manifest.json").read_text() == saved_manifest
    assert sorted(path.read_bytes() for path in (folder / "verdicts").glob("part-*.jsonl")) == saved_chunks

    judge.run_behavioral(tmp_path, list(reversed(records)), cue, Runner, "rt2", log=lambda message: None)
    assert len(judge.read_verdicts(folder)) == len(records) * len(judge.CONDITIONS) * judge.REPEATS
