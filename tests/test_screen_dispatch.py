"""Regression checks for the public four-judge screen."""
from unittest.mock import Mock

import pytest

from refusal_cues.judges import REVISIONS, evaluate, judge_messages
from refusal_cues.prepare_refusal_screen import main as prepare
from refusal_cues.qwen3guard_judge import VIEWS


def test_public_judges():
    assert set(REVISIONS) == {"gemma", "llama_guard", "shieldgemma", "qwen3guard"}


def test_qwen_scores_both_views_from_one_native_generation():
    turns = judge_messages("qwen3guard", "question", "response", "native")
    runner = Mock()
    runner.run.return_value = {"raw_label": "CONTROVERSIAL", "verdict_controversial_unsafe": "UNSAFE",
                               "verdict_controversial_safe": "SAFE"}
    views, states = evaluate(runner, "qwen3guard", turns, "native", 256)
    runner.run.assert_called_once_with(turns, 256)
    assert states is None
    assert {view: row["verdict"] for view, row in views} == {v: labels["CONTROVERSIAL"] for v, labels in VIEWS.items()}


def test_preparation_requires_revision_before_download(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["prepare", "--dataset", "jailjudge", "--out", str(tmp_path / "input")])
    with pytest.raises(SystemExit) as exc:
        prepare()
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())
