"""Residual-stream capture, shard storage, loading and a patching example for the
JAILJUDGE refusal-cue study. Imported by activation_capture.ipynb and copied into
every output folder so the activations can be loaded and patched later.

Positions (absolute token indices in the sequence the model processed):
  final_token  last token of the templated prompt (the final prefill token; the
              same position refusal_cues.inference.Runner hooks and edits).
  decision    the position whose next-token logits choose the verdict:
              chat judges: the last forced JSON token before the A/B letter
              (generation step `branch_index`), Llama Guard: the step that emits
              the first non-blank 'safe'/'unsafe' token, ShieldGemma: final_token.
Span means are taken over prefill tokens [start, end).
"""
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

VECTOR_KINDS = ("decision", "final_token", "response_mean", "cue_mean", "body_mean")
SENTINEL = "\u00a7"
FP16_MAX = 65504.0


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def judge_response(body, cue_text):
    """Exactly the behavioral runner's response text: B is the body; a cue is cue + blank line + body."""
    return body if not cue_text else cue_text + "\n\n" + body


def byte_spans(body, cue_text):
    """UTF-8 byte spans inside the judged response text (proposal's body-preservation fields)."""
    text = judge_response(body, cue_text)
    lead = len(text.encode("utf-8")) - len(body.encode("utf-8"))
    raw = text.encode("utf-8")
    if hashlib.sha256(raw[lead:]).hexdigest() != sha(body):
        raise ValueError("Body not preserved byte-for-byte")
    return {"cue_byte_start": 0, "cue_byte_end": len(cue_text.encode("utf-8")) if cue_text else 0,
            "body_byte_start": lead, "body_byte_end": len(raw),
            "body_sha256": sha(body), "variant_sha256": sha(text)}


def build_turns(model, request, response, mapping):
    """Chat judges use structured JSON turns. Guards use refusal_cues.guard_judges."""
    if model in ("ministral", "gemma"):
        from refusal_cues.structured_output import structured_messages
        return structured_messages(request, response, mapping)
    from refusal_cues.guard_judges import guard_messages
    return guard_messages(request, response, model)


def join_key(model, mapping, condition, source_id):
    return f"{model}|{mapping}|{condition}|{source_id}"


# ---------------------------------------------------------------- token spans
def common_prefix(a, b):
    n, p = min(len(a), len(b)), 0
    while p < n and a[p] == b[p]:
        p += 1
    return p


def common_suffix(a, b, limit):
    s = 0
    while s < limit and a[-1 - s] == b[-1 - s]:
        s += 1
    return s


def token_spans(ids, sentinel_ids, cue_sentinel_ids, baseline_len):
    """Token spans of the response, the cue and the body inside one prompt.

    sentinel_ids:      same prompt with the whole response replaced by SENTINEL.
    cue_sentinel_ids:  same prompt with response = cue + blank line + SENTINEL (None for B).
    A span may include one boundary token that merges a JSON quote or the blank line.
    """
    start = common_prefix(ids, sentinel_ids)
    end = len(ids) - common_suffix(ids, sentinel_ids, min(len(ids), len(sentinel_ids)) - start)
    if cue_sentinel_ids is None:
        cue_end = start
    else:
        cue_end = min(max(common_prefix(ids, cue_sentinel_ids), start), end)
    return {"response_tok_start": start, "response_tok_end": end,
            "cue_tok_start": start, "cue_tok_end": cue_end,
            "body_tok_start": cue_end, "body_tok_end": end,
            "offset_vs_baseline": len(ids) - baseline_len,
            "span_ok": bool(start < end and cue_end <= end)}


# ---------------------------------------------------------------- hooks
def find_decoder_layers(model):
    import torch
    found = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and n.endswith("layers")
             and "vision" not in n and "audio" not in n
             and len(m) and "DecoderLayer" in type(m[0]).__name__]
    if len(found) != 1:
        raise RuntimeError(f"Ambiguous decoder hook location: {[n for n, _ in found]}")
    return found[0]


class ResidualRecorder:
    """Forward hooks on decoder blocks (block output = residual stream after that block).

    Call 0 is the prefill (full prompt). Call c >= 1 processes generated token c-1
    and produces the logits for generated token c. Records the last-position state
    for each call in `keep_calls`, and span means during prefill. Optional `patch`
    {layer, call, vector} replaces the last-position state of that call (example only).
    """
    def __init__(self, layers, indices, prompt_len, spans, keep_calls, patch=None):
        self.layers, self.indices, self.prompt_len = layers, list(indices), prompt_len
        self.spans, self.keep_calls, self.patch = spans, set(keep_calls), patch
        self.calls = {i: 0 for i in self.indices}
        self.last = {}   # (call, layer) -> float32 cpu vector
        self.means = {}  # (kind, layer) -> float32 cpu vector or None
        self.handles = []

    def __enter__(self):
        hook_layers = set(self.indices) | ({self.patch["layer"]} if self.patch else set())
        self.calls.update({i: 0 for i in hook_layers})
        for i in sorted(hook_layers):
            self.handles.append(self.layers[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, index):
        def hook(module, args, output):
            import torch
            call = self.calls[index]
            self.calls[index] += 1
            x = output[0] if isinstance(output, tuple) else output
            expected = self.prompt_len if call == 0 else 1
            if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != expected:
                raise RuntimeError(f"Unexpected hidden shape {tuple(x.shape)} at call {call}; expected [1,{expected},H]")
            if self.patch and index == self.patch["layer"] and call == self.patch["call"]:
                y = x.clone()
                y[0, -1] = torch.as_tensor(self.patch["vector"], device=x.device).to(x.dtype)
                x = y
                output = (y,) + tuple(output[1:]) if isinstance(output, tuple) else y
            if index in self.indices:
                if call in self.keep_calls:
                    self.last[call, index] = x[0, -1].float().cpu()
                if call == 0:
                    for kind, (s, e) in self.spans.items():
                        self.means[kind, index] = x[0, s:e].float().mean(0).cpu() if e > s else None
            return output
        return hook

    def check_fired(self, calls_needed):
        if any(self.calls[i] < calls_needed for i in self.indices):
            raise RuntimeError("Not all hooks executed for the required calls")


# ---------------------------------------------------------------- one prompt
def _logprobs(logits_row, ids):
    import torch
    lp = torch.log_softmax(logits_row.float(), -1)
    if not torch.isfinite(lp[list(ids)]).all():
        raise RuntimeError("Nonfinite verdict log-probs")
    return [float(logits_row[i]) for i in ids], [float(lp[i]) for i in ids]


def capture_one(judge, model, turns, mapping, spans_tok, input_ids_dict):
    """Run the behavioral decoding once with hooks. Returns (vectors, row_fields).

    judge: object with .model, .tokenizer, .layers, .depths (and for guards .vocab).
    vectors: dict kind -> list[layer] of float32 tensors (None for empty spans).
    """
    import torch
    from refusal_cues.protocol import MAPPINGS
    from refusal_cues.guard_judges import parse_guard
    m, tok = judge.model, judge.tokenizer
    L = int(input_ids_dict["input_ids"].shape[-1])
    span_ranges = {"response_mean": (spans_tok["response_tok_start"], spans_tok["response_tok_end"]),
                   "cue_mean": (spans_tok["cue_tok_start"], spans_tok["cue_tok_end"]),
                   "body_mean": (spans_tok["body_tok_start"], spans_tok["body_tok_end"])}
    fields = {}
    if model in ("ministral", "gemma"):
        from refusal_cues.structured_output import JsonVerdictConstraint
        constraint = JsonVerdictConstraint(tok, L, m.generation_config.eos_token_id)
        bi = constraint.branch_index
        path_a, path_b = constraint.paths
        with ResidualRecorder(judge.layers, judge.depths, L, span_ranges, {0, bi}) as rec, torch.inference_mode():
            out = m.generate(**input_ids_dict, do_sample=False, num_beams=1, max_new_tokens=bi + 1,
                             use_cache=True, return_dict_in_generate=True, output_logits=True,
                             prefix_allowed_tokens_fn=constraint)
        rec.check_fired(bi + 1)
        new = out.sequences[0, L:].tolist()
        if len(new) != bi + 1 or new not in (path_a[:bi + 1], path_b[:bi + 1]):
            raise RuntimeError("Constrained JSON prefix not reproduced")
        code = "A" if new[bi] == path_a[bi] else "B"
        verdict = MAPPINGS[mapping][code]
        raw, lp = _logprobs(out.logits[bi][0], (path_a[bi], path_b[bi]))
        sem = {MAPPINGS[mapping]["A"]: (raw[0], lp[0]), MAPPINGS[mapping]["B"]: (raw[1], lp[1])}
        unsafe_code = "B" if mapping == "A_safe" else "A"
        _, pe = _logprobs(out.logits[0][0], (judge.codes["A"], judge.codes["B"]))
        pe = dict(zip("AB", pe))
        fields.update(final_token_logprob_A=pe["A"], final_token_logprob_B=pe["B"],
                      final_token_ab_mass=math.exp(pe["A"]) + math.exp(pe["B"]),
                      final_token_semantic_margin=pe[unsafe_code] - pe["B" if unsafe_code == "A" else "A"],
                      decision_ab_mass=math.exp(lp[0]) + math.exp(lp[1]))
        fields.update(answer_code=code, logit_A=raw[0], logit_B=raw[1], logprob_A=lp[0], logprob_B=lp[1],
                      decision_call=bi, decision_position=L - 1 + bi, decision_token_ids=new[:bi])
        decision_call = bi
    elif model == "llama_guard":
        keep = set(range(judge.decision_search))
        with ResidualRecorder(judge.layers, judge.depths, L, span_ranges, keep) as rec, torch.inference_mode():
            out = m.generate(**input_ids_dict, do_sample=False, max_new_tokens=judge.budget,
                             return_dict_in_generate=True, output_logits=True)
        new = out.sequences[0, L:].tolist()
        verdict = parse_guard(tok.decode(new, skip_special_tokens=True), "llama_guard")
        k = next((i for i in range(len(new)) if tok.decode(new[:i + 1], skip_special_tokens=True).strip()), None)
        if k is None or k >= judge.decision_search:
            raise RuntimeError(f"Llama Guard verdict token not within first {judge.decision_search} steps")
        raw, lp = _logprobs(out.logits[k][0], (judge.safe_id, judge.unsafe_id))
        sem = {"SAFE": (raw[0], lp[0]), "UNSAFE": (raw[1], lp[1])}
        _, pe = _logprobs(out.logits[0][0], (judge.safe_id, judge.unsafe_id))
        fields.update(final_token_logprob_safe=pe[0], final_token_logprob_unsafe=pe[1],
                      final_token_safe_unsafe_mass=math.exp(pe[0]) + math.exp(pe[1]),
                      final_token_semantic_margin=pe[1] - pe[0])
        fields.update(decision_call=k, decision_position=L - 1 + k, decision_token_id=new[k],
                      decision_token_is_candidate=new[k] in (judge.safe_id, judge.unsafe_id),
                      generated_text=tok.decode(new, skip_special_tokens=True))
        decision_call = k
    elif model == "shieldgemma":
        with ResidualRecorder(judge.layers, judge.depths, L, span_ranges, {0}) as rec, torch.inference_mode():
            logits = m(**input_ids_dict).logits[0, -1].float()
        yes, no = judge.yes_id, judge.no_id
        score = torch.softmax(logits[[yes, no]], dim=0)[0].item()
        verdict = "UNSAFE" if score > 0.5 else "SAFE"
        raw, lp = _logprobs(logits, (no, yes))
        sem = {"SAFE": (raw[0], lp[0]), "UNSAFE": (raw[1], lp[1])}
        fields.update(unsafe_probability=score, decision_call=0, decision_position=L - 1,
                      final_token_logprob_safe=lp[0], final_token_logprob_unsafe=lp[1],
                      final_token_safe_unsafe_mass=math.exp(lp[0]) + math.exp(lp[1]),
                      final_token_semantic_margin=lp[1] - lp[0])
        decision_call = 0
    else:
        raise ValueError(model)
    rec.check_fired(1)
    vectors = {"decision": [rec.last[decision_call, i] for i in judge.depths],
               "final_token": [rec.last[0, i] for i in judge.depths]}
    for kind in span_ranges:
        vectors[kind] = [rec.means[kind, i] for i in judge.depths]
    fields.update(verdict=verdict, prompt_tokens=L, final_token_position=L - 1,
                  logit_safe=sem["SAFE"][0], logit_unsafe=sem["UNSAFE"][0],
                  logprob_safe=sem["SAFE"][1], logprob_unsafe=sem["UNSAFE"][1],
                  unsafe_margin=sem["UNSAFE"][1] - sem["SAFE"][1])
    return vectors, fields


# ---------------------------------------------------------------- storage
def stack_shard(vector_rows, kinds, n_layers, hidden):
    """vector_rows: list of dict kind -> list[layer] tensors. Returns {kind: [n, layers, H]} float32."""
    import torch
    out = {}
    for kind in kinds:
        t = torch.zeros(len(vector_rows), n_layers, hidden, dtype=torch.float32)
        for r, row in enumerate(vector_rows):
            for j, v in enumerate(row[kind]):
                if v is not None:
                    t[r, j] = v
        out[kind] = t
    return out


def to_storage_dtype(tensors, preferred="float16"):
    """fp16 unless any value would overflow fp16; then the whole shard is bf16."""
    import torch
    if preferred == "float16" and all(float(t.abs().max()) < FP16_MAX if t.numel() else True for t in tensors.values()):
        return {k: v.to(torch.float16).contiguous() for k, v in tensors.items()}, "float16"
    return {k: v.to(torch.bfloat16).contiguous() for k, v in tensors.items()}, "bfloat16"


def atomic_copy(local_path, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copyfile(local_path, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dest)


def atomic_write_text(text, dest):
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        f.write(text)
    atomic_copy(f.name, dest)
    os.unlink(f.name)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def shard_paths(run_dir, model, mapping, condition, shard):
    base = Path(run_dir) / "activations" / model / mapping / condition
    return base / f"shard-{shard:02d}.safetensors", base / f"shard-{shard:02d}.json"


def write_shard(run_dir, model, mapping, condition, shard, tensors_f32, rows, layers, meta, preferred_dtype="float16"):
    """Write tensors then the JSON completion marker, each atomically. Returns the marker dict."""
    from safetensors.torch import save_file
    tensors, dtype = to_storage_dtype(tensors_f32, preferred_dtype)
    st_path, json_path = shard_paths(run_dir, model, mapping, condition, shard)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / st_path.name
        save_file(tensors, str(local), metadata={"model": model, "mapping": mapping, "condition": condition,
                  "shard": str(shard), "layers": json.dumps(layers), "dtype": dtype,
                  "source_ids": json.dumps([r["source_id"] for r in rows])})
        digest, size = file_sha256(local), local.stat().st_size
        atomic_copy(local, st_path)
    marker = {"complete": True, "model": model, "mapping": mapping, "condition": condition, "shard": shard,
              "layers": layers, "kinds": list(tensors), "dtype": dtype,
              "shape": {k: list(v.shape) for k, v in tensors.items()},
              "safetensors_file": str(st_path.relative_to(run_dir)), "safetensors_sha256": digest,
              "safetensors_bytes": size, "rows": rows, **meta}
    atomic_write_text(json.dumps(marker), json_path)
    return marker


def shard_complete(run_dir, model, mapping, condition, shard, source_ids, verify_hash=False):
    st_path, json_path = shard_paths(run_dir, model, mapping, condition, shard)
    if not (json_path.is_file() and st_path.is_file()):
        return None
    try:
        marker = json.loads(json_path.read_text())
    except ValueError:
        return None
    if (not marker.get("complete") or [r["source_id"] for r in marker["rows"]] != list(source_ids)
            or st_path.stat().st_size != marker["safetensors_bytes"]):
        return None
    if verify_hash and file_sha256(st_path) != marker["safetensors_sha256"]:
        return None
    return marker


# ---------------------------------------------------------------- loading
def load_shard(run_dir, model, mapping, condition, shard, kinds=None):
    from safetensors.torch import load_file
    st_path, json_path = shard_paths(run_dir, model, mapping, condition, shard)
    marker = json.loads(json_path.read_text())
    tensors = load_file(str(st_path))
    return {k: v for k, v in tensors.items() if kinds is None or k in kinds}, marker


def load_condition(run_dir, model, mapping, condition, kind="decision"):
    """Return (tensor [470, n_layers, H] in input order, rows list, layers)."""
    import torch
    base = Path(run_dir) / "activations" / model / mapping / condition
    parts, rows, layers = [], [], None
    for json_path in sorted(base.glob("shard-*.json")):
        shard = int(json_path.stem.split("-")[1])
        tensors, marker = load_shard(run_dir, model, mapping, condition, shard, [kind])
        parts.append(tensors[kind]); rows += marker["rows"]; layers = marker["layers"]
    return torch.cat(parts), rows, layers


# ---------------------------------------------------------------- patching example
def verdict_with_residual_patch(judge, model, turns, mapping, layer, donor_vector, position="decision"):
    """EXAMPLE (chat judges only): rerun the constrained verdict with the residual stream
    after decoder block `layer` replaced at one position by `donor_vector`
    (e.g. the baseline B activation for the same record), and return verdict + log-probs.

    position='final_token' patches the final prefill token; 'decision' patches the
    generation step that chooses A/B. `layer` must be an absolute decoder index.
    """
    import torch
    from refusal_cues.protocol import MAPPINGS
    from refusal_cues.structured_output import JsonVerdictConstraint
    if model not in ("ministral", "gemma"):
        raise ValueError("Example covers the chat judges; adapt decision_call for guards")
    inputs = judge.encode(turns)
    L = int(inputs["input_ids"].shape[-1])
    constraint = JsonVerdictConstraint(judge.tokenizer, L, judge.model.generation_config.eos_token_id)
    bi = constraint.branch_index
    call = 0 if position == "final_token" else bi
    patch = {"layer": layer, "call": call, "vector": donor_vector}
    with ResidualRecorder(judge.layers, [], L, {}, set(), patch=patch), torch.inference_mode():
        out = judge.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=bi + 1,
                                   use_cache=True, return_dict_in_generate=True, output_logits=True,
                                   prefix_allowed_tokens_fn=constraint)
    new = out.sequences[0, L:].tolist()
    code = "A" if new[bi] == constraint.paths[0][bi] else "B"
    raw, lp = _logprobs(out.logits[bi][0], (constraint.paths[0][bi], constraint.paths[1][bi]))
    return {"verdict": MAPPINGS[mapping][code], "answer_code": code, "logprob_A": lp[0], "logprob_B": lp[1]}


# ---------------------------------------------------------------- planning
CHAT_JUDGES = ("ministral", "gemma")


def mappings_for(model):
    return ["A_safe", "A_unsafe"] if model in CHAT_JUDGES else ["native"]


def build_plan(models, conditions, records, shard_size):
    """One unit per (model, mapping, condition, shard). Records keep input-file order."""
    plan = []
    for model in models:
        for mapping in mappings_for(model):
            for condition in conditions:
                for shard, start in enumerate(range(0, len(records), shard_size)):
                    plan.append({"model": model, "mapping": mapping, "condition": condition, "shard": shard,
                                 "source_ids": [r["source_id"] for r in records[start:start + shard_size]]})
    return plan


def now_sgt():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


# ---------------------------------------------------------------- runtime lock
def acquire_lock(run_dir, runtime_id, stale_seconds=900, takeover=False):
    import time
    path = Path(run_dir) / "runtime-lock.json"
    if path.exists():
        try:
            held = json.loads(path.read_text())
        except ValueError:
            held = {}
        age = time.time() - held.get("heartbeat_unix", 0)
        if held.get("runtime_id") not in (None, runtime_id) and age < stale_seconds and not takeover:
            raise RuntimeError(
                f"Another runtime ({held.get('runtime_id')}) wrote a heartbeat {age:.0f}s ago. "
                "Never run two runtimes on one folder. Wait until it is older than "
                f"{stale_seconds}s, or set TAKEOVER_LOCK=True only if that runtime is definitely gone.")
    heartbeat(run_dir, runtime_id)


def heartbeat(run_dir, runtime_id, **extra):
    import time
    atomic_write_text(json.dumps({"runtime_id": runtime_id, "heartbeat_unix": time.time(),
                                  "heartbeat_sgt": now_sgt(), **extra}), Path(run_dir) / "runtime-lock.json")


# ---------------------------------------------------------------- capture loop
def stable_meta(meta):
    return {k: v for k, v in meta.items() if k not in ("gpu", "peak_gpu_bytes", "loaded_sgt")}


def run_capture(run_dir, plan, records_by_id, cue_text, load_judge, runtime_id, behavioral=None,
                preferred_dtype="float16", verify_hash=False, seed=20261004, log=print, max_units=None):
    """Capture every incomplete shard in `plan`. Resumable: completed shards are skipped.

    load_judge(model) -> judge with .model .tokenizer .layers .depths .encode(turns) .meta .hidden
    behavioral: optional dict join_key -> verdict (repeat 0) for live agreement logging.
    """
    import gc
    import time
    import torch
    run_dir = Path(run_dir)
    done_units = 0
    models = list(dict.fromkeys(u["model"] for u in plan))
    for model in models:
        todo = [u for u in plan if u["model"] == model and not shard_complete(
            run_dir, u["model"], u["mapping"], u["condition"], u["shard"], u["source_ids"], verify_hash)]
        total_model = sum(len(u["source_ids"]) for u in plan if u["model"] == model)
        remaining = sum(len(u["source_ids"]) for u in todo)
        log(f"{model}: {remaining} of {total_model} prompts remaining ({len(todo)} shards)")
        if not todo:
            continue
        judge = load_judge(model)
        meta = {**{k: v for k, v in getattr(judge, "meta", {}).items() if k != "software"},
                "software": getattr(judge, "meta", {}).get("software"),
                "decoder_path": getattr(judge, "layer_path", None), "layers": list(judge.depths),
                "n_decoder_layers": len(judge.layers), "hidden_size": judge.hidden,
                "vector_kinds": list(VECTOR_KINDS), "loaded_sgt": now_sgt()}
        manifest_path = run_dir / "activations" / model / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if stable_meta(previous) != stable_meta(json.loads(json.dumps(meta))):
                raise RuntimeError(f"{model}: runtime differs from the manifest already in this folder; "
                                   "start a new run (START_NEW_RUN=True) instead of mixing captures")
        atomic_write_text(json.dumps(meta, indent=2), manifest_path)
        agree = disagree = 0
        started, prompts_done = time.monotonic(), 0
        for unit in todo:
            mapping, condition = unit["mapping"], unit["condition"]
            cue = cue_text.get(condition, "")
            vector_rows, rows = [], []
            for offset, sid in enumerate(unit["source_ids"]):
                rec = records_by_id[sid]
                request, body = rec["original_prompt"], rec["original_response"]
                turns = build_turns(model, request, judge_response(body, cue), mapping)
                row = {"source_id": sid, "harmful": bool(rec["mapped_unsafe"]), "model": model,
                       "mapping": mapping, "condition": condition, "cue_text": cue,
                       "join_key": join_key(model, mapping, condition, sid), "row_offset": offset,
                       **byte_spans(body, cue),
                       "prompt_sha256": sha(json.dumps(turns, ensure_ascii=False))}
                for attempt in (1, 2):
                    try:
                        inputs = judge.encode(turns)
                        ids = inputs["input_ids"][0].tolist()
                        ids_of = lambda t: judge.encode(t)["input_ids"][0].tolist()
                        sentinel_ids = ids_of(build_turns(model, request, SENTINEL, mapping))
                        cue_sentinel = (ids_of(build_turns(model, request, judge_response(SENTINEL, cue), mapping))
                                        if cue else None)
                        base_len = len(ids) if not cue else len(ids_of(build_turns(model, request, body, mapping)))
                        spans = token_spans(ids, sentinel_ids, cue_sentinel, base_len)
                        torch.manual_seed(seed)
                        vectors, fields = capture_one(judge, model, turns, mapping, spans, inputs)
                        row.update(spans, **fields, input_ids_sha256=sha(json.dumps([ids])), status="ok")
                        break
                    except Exception as exc:  # keep a 19,740-prompt run alive; failures are listed
                        if attempt == 2:
                            log(f"ERROR {row['join_key']}: {type(exc).__name__}: {exc}")
                            vectors = {k: [None] * len(judge.depths) for k in VECTOR_KINDS}
                            row.update(status="error", error=f"{type(exc).__name__}: {exc}", verdict="ERROR")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                if behavioral is not None and row["status"] == "ok":
                    expected = behavioral.get(row["join_key"])
                    row["behavioral_verdict"] = expected
                    if expected is not None:
                        agree += expected == row["verdict"]
                        disagree += expected != row["verdict"]
                vector_rows.append(vectors)
                rows.append(row)
            tensors = stack_shard(vector_rows, VECTOR_KINDS, len(judge.depths), judge.hidden)
            write_shard(run_dir, model, mapping, condition, unit["shard"], tensors, rows, list(judge.depths),
                        {"written_sgt": now_sgt(), "runtime_id": runtime_id}, preferred_dtype)
            prompts_done += len(rows)
            remaining -= len(rows)
            rate = (time.monotonic() - started) / prompts_done
            heartbeat(run_dir, runtime_id, model=model, last_shard=f"{mapping}/{condition}/{unit['shard']:02d}")
            match = f"; verdict match {agree}/{agree + disagree}" if agree + disagree else ""
            log(f"{now_sgt()} {model} {mapping} {condition} shard {unit['shard']:02d} saved; "
                f"{rate:.2f}s/prompt; {remaining} left for {model} (~{remaining * rate / 60:.0f} min){match}")
            done_units += 1
            if max_units is not None and done_units >= max_units:
                return
        del judge
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- index and agreement
def load_behavioral(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, dtype={"source_id": str, "mapping": str, "condition": str, "model": str})
    df["join_key"] = df.model + "|" + df.mapping + "|" + df.condition + "|" + df.source_id
    return df


def build_index(run_dir, plan, behavioral_df=None):
    """Write index.csv (+ parquet), agreement.csv and mismatches.csv. Returns (index, agreement, summary)."""
    import pandas as pd
    run_dir = Path(run_dir)
    rows, missing = [], []
    for u in plan:
        marker = shard_complete(run_dir, u["model"], u["mapping"], u["condition"], u["shard"], u["source_ids"])
        if marker is None:
            missing.append(u)
            continue
        for r in marker["rows"]:
            rows.append({**{k: v for k, v in r.items() if not isinstance(v, (list, dict))},
                         "decision_token_ids": json.dumps(r.get("decision_token_ids")) if "decision_token_ids" in r else None,
                         "layers": json.dumps(marker["layers"]), "shard": u["shard"],
                         "shard_file": marker["safetensors_file"], "dtype": marker["dtype"]})
    index = pd.DataFrame(rows)
    summary = {"expected_prompts": sum(len(u["source_ids"]) for u in plan), "indexed_prompts": len(index),
               "missing_shards": len(missing), "errors": int((index.get("status") == "error").sum()) if len(index) else 0}
    agreement = None
    if behavioral_df is not None and len(index):
        b = behavioral_df.pivot_table(index="join_key", columns="repeat", values="verdict", aggfunc="first")
        b.columns = [f"behavioral_verdict_r{c}" for c in b.columns]
        index = index.drop(columns=[c for c in index.columns if c.startswith("behavioral_verdict")])
        index = index.merge(b, left_on="join_key", right_index=True, how="left")
        index["matches_behavioral"] = index.verdict == index["behavioral_verdict_r0"]
        agreement = (index.groupby(["model", "mapping", "condition"])
                     .agg(prompts=("join_key", "size"), matches=("matches_behavioral", "sum"))
                     .reset_index())
        agreement["agreement_rate"] = agreement.matches / agreement.prompts
        agreement.to_csv(run_dir / "agreement.csv", index=False)
        index.loc[~index.matches_behavioral].to_csv(run_dir / "mismatches.csv", index=False)
        summary.update(matches=int(index.matches_behavioral.sum()),
                       agreement_rate=float(index.matches_behavioral.mean()),
                       no_behavioral_row=int(index["behavioral_verdict_r0"].isna().sum()))
    index.to_csv(run_dir / "index.csv", index=False)
    try:
        index.to_parquet(run_dir / "index.parquet", index=False)
    except Exception as exc:
        summary["parquet_error"] = str(exc)
    atomic_write_text(json.dumps(summary, indent=2), run_dir / "index-summary.json")
    return index, agreement, summary


def make_zips(run_dir, models, local_tmp="/tmp"):
    """ZIP_STORED (fp16 does not compress). One zip per model plus index-and-docs.zip."""
    import zipfile
    run_dir = Path(run_dir)
    made = []
    docs = ["README.md", "index.csv", "index.parquet", "index-summary.json", "agreement.csv", "mismatches.csv",
            "activation_tools.py", "run-manifest.json", "behavioral/all_results.csv"]
    targets = [("index-and-docs.zip", [run_dir / d for d in docs if (run_dir / d).is_file()]
                + sorted((run_dir / "activations").glob("*/manifest.json")))]
    targets += [(f"activations-{m}.zip", sorted(p for p in (run_dir / "activations" / m).rglob("*")
                                                 if p.is_file() and not p.name.endswith(".tmp"))) for m in models]
    for name, paths in targets:
        local = Path(local_tmp) / name
        with zipfile.ZipFile(local, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in paths:
                z.write(p, str(p.relative_to(run_dir)))
        atomic_copy(local, run_dir / "zips" / name)
        made.append((run_dir / "zips" / name, local.stat().st_size))
        local.unlink()
    return made
