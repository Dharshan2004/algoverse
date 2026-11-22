"""Dispatch each checkpoint to its own prompt, decoder and verdict parser."""
from refusal_cues.guard_judges import GUARDS, GuardRunner, guard_messages
from refusal_cues.inference import Runner
from refusal_cues.protocol import MODELS
from refusal_cues.qwen3guard_judge import Qwen3GuardRunner, VIEWS
from refusal_cues.structured_output import parse_structured, structured_messages

REVISIONS = {
    "gemma": "43c9d014b231474fa7212dbb48a762566c2b255d",
    "llama_guard": "acf7aafa60f0410f8f42b1fa35e077d705892029",
    "shieldgemma": "d1dffc9c8c9237a90aab09c61383791e718ef9e8",
    "qwen3guard": "4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb",
}


def load_judge(name, revision=None):
    revision = revision or REVISIONS[name]
    if name in MODELS:
        return Runner(MODELS[name], revision)
    if name in GUARDS:
        return GuardRunner(name, revision)
    if name == "qwen3guard":
        return Qwen3GuardRunner(revision=revision)
    raise ValueError(f"Unknown judge {name}")


def judge_messages(name, request, response, mapping):
    if name in MODELS:
        return structured_messages(request, response, mapping)
    return guard_messages(request, response, name)


def evaluate(runner, name, turns, mapping, budget):
    """Return scored views and optional chat states after a single generation."""
    if name in MODELS:
        result, states = runner.run(turns, max_new_tokens=budget)
        result.update(parse_structured(result["parsed_text"], mapping))
        return [(mapping, result)], states
    result = runner.run(turns, budget)
    if name == "qwen3guard":
        return [(view, {**result, "verdict": result[f"verdict_{view}"]}) for view in VIEWS], None
    return [(mapping, result)], None
