"""Checks on the committed 470-record refusal-screen summaries."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path("experiments/behavioral_refusal_screen")
HERE = ROOT / EXPERIMENT
SUMMARIES = HERE / "summaries"


def test_summary_table_covers_every_group():
    with (ROOT / "archive" / EXPERIMENT / "summaries" / "all_results_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 84
    assert len({row["model"] for row in rows}) == 4
    assert all(row["resolved"] == "True" for row in rows)
    assert all(int(row["n"]) == 470 for row in rows)
    assert all(row["INVALID"] == "0" for row in rows)


def test_analysis_matches_the_audit():
    analysis = json.loads((SUMMARIES / "analysis.json").read_text())
    audit = json.loads((SUMMARIES / "audit" / "independent_check.json").read_text())
    assert analysis["input_records"] == audit["n_records"] == 470
    assert analysis["harmful_records"] == audit["harmful_count"] == 270
    assert analysis["safe_records"] == 200
    assert audit["dups"] == 0
    assert audit["label_mismatch"] == 0
    assert audit["ids_not_in_dataset"] == 0
    assert audit["n_groups"] == 84
    assert audit["group_sizes"] == [470]
    assert len(analysis["lead"]) == 6
    report = (SUMMARIES / "analysis.md").read_text()
    assert "22/240 (9.17%)" in report
    assert "84/108 (77.78%)" in report
    for row in analysis["lead"]:
        assert f"{row['DFR5']['numerator']}/{row['DFR5']['denominator']}" in report


def test_notebook_is_the_resume_screen():
    notebook = json.loads((HERE / "behavioral_refusal_screen.ipynb").read_text())
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    joined = "\n".join(code)
    for name in ("llama_guard", "shieldgemma", "qwen3guard", "gemma"):
        assert name in joined
    assert "scripts.run_jailjudge_baseline" in joined
    assert "git', 'clone" in joined
    for index, source in enumerate(code):
        compile(source, f"cell{index}", "exec")
