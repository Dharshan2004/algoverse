"""Offline checks for the Qwen3Guard judge. Archive parity lives in units_full.py."""
import importlib.util
import json
import sys
from pathlib import Path

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
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code
    for index, cell in enumerate(code):
        source = "".join(cell["source"])
        if source.startswith("%%writefile"):
            source = source.split("\n", 1)[1]
        compile(source, f"cell{index}", "exec")
    joined = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "refusal_cues.guard_judges" in joined
