"""Offline tests for adapting Hugging Face records and selecting dataset revisions."""
import json
from unittest.mock import Mock

import pytest

from refusal_cues import adapt_hf_dataset as adapter


def test_hugging_face_row_maps_to_runner_schema():
    source = dict(source_id="s1", cluster_id="c1", study_partition="development",
                  user_prompt="  question", model_response="café\n", mapped_unsafe=False,
                  label_provenance="human", language="en")
    rows = adapter.adapt_records([source], partition="development")
    assert rows == [dict(source_id="s1", cluster_id="c1", study_partition="development",
                         original_prompt="  question", original_response="café\n", mapped_unsafe=False,
                         label_provenance="human", meta={"language": "en"})]
    adapter.validate_records(rows)


def test_final_test_export_warns_and_skips_runner_validation(tmp_path, capsys, monkeypatch):
    validate = Mock()
    monkeypatch.setattr(adapter, "validate_records", validate)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(dict(source_id="fixture", cluster_id="c", study_partition="final_test",
                                      user_prompt="question", model_response="answer", mapped_unsafe=False)) + "\n")
    out = tmp_path / "adapted.jsonl"
    adapter.main(["--input", str(source), "--split", "final_test", "--out", str(out)])
    assert "WARNING: final_test partition" in capsys.readouterr().err
    assert json.loads(out.read_text())["study_partition"] == "final_test"
    validate.assert_not_called()


def test_download_without_revision_errors_before_fetch(tmp_path, capsys, monkeypatch):
    download = Mock()
    monkeypatch.setattr(adapter, "load_hf_split", download)
    out = tmp_path / "adapted.jsonl"
    with pytest.raises(SystemExit) as error:
        adapter.main(["--out", str(out)])
    assert error.value.code == 2
    assert "--revision is required for downloads" in capsys.readouterr().err
    download.assert_not_called()
    assert not out.exists()
