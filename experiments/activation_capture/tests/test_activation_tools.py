"""Offline checks for the 4-judge activation capture helpers."""
import base64
import importlib.util
import json
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "src" / "activation_tools.py"


def load_tools():
    spec = importlib.util.spec_from_file_location("activation_capture_tools", SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_turns_use_guard_judges():
    from refusal_cues.guard_judges import guard_messages

    tools = load_tools()
    request, response = "please help", "I cannot help with that request.\n\nbody"
    for model in ("llama_guard", "shieldgemma"):
        assert tools.build_turns(model, request, response, "native") == guard_messages(request, response, model)


def test_byte_spans_preserve_body():
    tools = load_tools()
    body = '  café "quoted"\n'
    cue = "I cannot help with that request."
    spans = tools.byte_spans(body, cue)
    text = tools.judge_response(body, cue).encode("utf-8")
    assert text[spans["body_byte_start"]:spans["body_byte_end"]] == body.encode("utf-8")
    assert spans["body_sha256"] == tools.sha(body)
    bare = tools.byte_spans(body, "")
    assert bare["cue_byte_end"] == 0
    assert bare["body_byte_start"] == 0


def test_plan_covers_four_judges():
    from refusal_cues.protocol import depths

    tools = load_tools()
    records = [{"source_id": f"s{i}"} for i in range(470)]
    conditions = ["B", "N", "R", "R2", "paper_R1", "paper_R2", "paper_Rgreet"]
    models = ["llama_guard", "shieldgemma", "ministral", "gemma"]
    plan = tools.build_plan(models, conditions, records, 47)
    assert sum(len(unit["source_ids"]) for unit in plan) == 19740
    assert len(plan) == 420
    assert depths(16) == [3, 5, 7, 8, 10, 11]
    assert depths(26) == [6, 9, 11, 14, 16, 19]
    assert depths(34) == [8, 11, 15, 18, 22, 25]
    assert depths(48) == [11, 16, 21, 26, 31, 35]


def test_token_spans_and_lock(tmp_path):
    tools = load_tools()
    ids = [1, 2, 3, 9, 9, 4, 5]
    sentinel = [1, 2, 7, 4, 5]
    cue_sentinel = [1, 2, 3, 8, 4, 5]
    spans = tools.token_spans(ids, sentinel, cue_sentinel, baseline_len=6)
    assert spans["span_ok"]
    assert spans["response_tok_start"] == 2
    assert spans["cue_tok_end"] == 3
    assert spans["body_tok_start"] == 3
    assert spans["offset_vs_baseline"] == 1
    tools.acquire_lock(tmp_path, "runtime-a")
    tools.heartbeat(tmp_path, "runtime-a", stage="test")
    with pytest.raises(RuntimeError, match="Another runtime"):
        tools.acquire_lock(tmp_path, "runtime-b", stale_seconds=900)
    held = json.loads((tmp_path / "runtime-lock.json").read_text())
    assert held["runtime_id"] == "runtime-a"


def test_notebook_code_cells_parse():
    notebook = json.loads((HERE / "activation_capture.ipynb").read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code
    for index, cell in enumerate(code):
        source = "".join(cell["source"])
        if source.startswith("%%writefile"):
            source = source.split("\n", 1)[1]
        compile(source, f"cell{index}", "exec")
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    joined = "\n".join(sources)
    assert "from refusal_cues.guard_judges import GuardRunner" in joined
    assert "from refusal_cues.guard_judges import guard_messages" in joined
    assert "from refusal_cues.guard_judges import parse_guard" in joined
    install = next(source for source in sources if "guard_judges.py" in source and "b64decode" in source)
    imported = next(source for source in sources if "from refusal_cues.guard_judges import GuardRunner" in source)
    assert sources.index(install) < sources.index(imported)
    encoded = re.search(r"base64\.b64decode\('([^']+)'\)", install).group(1)
    repo_module = Path(__file__).resolve().parents[3] / "refusal_cues" / "guard_judges.py"
    assert base64.b64decode(encoded) == repo_module.read_bytes()
