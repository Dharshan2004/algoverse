"""Colab step-zero inference; real GPU validation remains required."""
import argparse
import importlib.metadata
import json
import math
import re
import time
from pathlib import Path
from protocol import MODELS, MAPPINGS, POLICY, depths, messages, normalize_native_output, parse, sha, validate_records, variant

class Runner:
    def __init__(self, model_id, revision="main", compute_dtype="bfloat16"):
        import torch
        import transformers as tr
        from huggingface_hub import model_info
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required")
        if compute_dtype not in ("bfloat16", "float16"):
            raise ValueError("Unsupported compute dtype")
        if compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("BF16 requires a compatible GPU. Use --compute-dtype float16 only for the approved T4 compatibility smoke test.")
        dtype = getattr(torch, compute_dtype)
        self.torch, self.model_id = torch, model_id
        print(f"Resolving checkpoint: {model_id}", flush=True)
        self.revision = model_info(model_id, revision=revision).sha
        torch.manual_seed(20261004)
        config = tr.AutoConfig.from_pretrained(model_id, revision=self.revision, trust_remote_code=False)
        arch = config.architectures[0]
        cls = getattr(tr, arch, None)
        if cls is None:
            raise RuntimeError(f"Installed transformers lacks native architecture {arch}; update then rerun smoke checks")
        self.processor = (tr.MistralCommonBackend.from_pretrained(model_id, revision=self.revision)
                          if model_id.startswith("mistralai/Ministral") else
                          tr.AutoProcessor.from_pretrained(model_id, revision=self.revision, trust_remote_code=False))
        self.tokenizer = (self.processor if model_id.startswith("mistralai/Ministral")
                          else self.processor.tokenizer)
        print(f"Loading {arch} at {self.revision} with NF4/{compute_dtype}", flush=True)
        self.model = cls.from_pretrained(model_id, revision=self.revision, trust_remote_code=False,
            device_map={"": 0}, dtype=dtype,
            quantization_config=tr.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)).eval()
        self.model.requires_grad_(False)
        print("Model loaded; validating decoder hooks and answer tokens", flush=True)
        # Find exactly one language decoder ModuleList, never vision encoder layers.
        candidates = [(name, mod) for name, mod in self.model.named_modules()
                      if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers")
                      and "vision" not in name and "audio" not in name
                      and len(mod) and "DecoderLayer" in type(mod[0]).__name__]
        if len(candidates) != 1:
            raise RuntimeError(f"Ambiguous decoder hook location: {[n for n, _ in candidates]}")
        self.layer_path, self.layers = candidates[0]
        self.depths = depths(len(self.layers))
        self.codes = {}
        for code in ("A", "B"):
            ids = self.tokenizer.encode(code, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.decode(ids).strip() != code:
                raise RuntimeError(f"Code {code} is not a verified single token")
            self.codes[code] = ids[0]
        template = (getattr(self.processor, "chat_template", None)
                    or getattr(self.tokenizer, "chat_template", None))
        self.meta = {"model_id": model_id, "revision": self.revision,
            "architecture": arch, "decoder_path": self.layer_path, "depths": self.depths,
            "code_ids": self.codes, "seed": 20261004, "max_input_tokens": 4096,
            "quantization": "NF4 double quantization", "compute_dtype": compute_dtype,
            "protocol_precision_match": compute_dtype == "bfloat16",
            "run_purpose": "protocol_smoke" if compute_dtype == "bfloat16" else "T4_FP16_compatibility_only",
            "thinking": False, "output_parser_version": "native_empty_channel_v1",
            "gpu": torch.cuda.get_device_name(0),
            "chat_template_sha256": sha(str(template)) if template else None,
            "template_backend": type(self.processor).__name__,
            "mistral_common_version": importlib.metadata.version("mistral-common") if model_id.startswith("mistralai/") else None,
            "software": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface-hub")}}

    def encode(self, turns):
        data = self.processor.apply_chat_template(turns, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt", return_dict=True)
        if data["input_ids"].shape[-1] > 4096:
            raise ValueError("Complete input exceeds 4096 tokens; never truncate the body")
        return {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in data.items()}

    def run(self, turns, max_new_tokens=4, edit=None):
        """edit: {layer, direction, mode: replace/remove/add/identity, donor/amount}.

        Direction estimation/validation belongs to the interpretability workstream.
        The optional operator acts once at the final prefill token, never during decoding.
        """
        t = self.torch
        inputs = self.encode(turns)
        length = inputs["input_ids"].shape[-1]
        states, handles, fired = {}, [], set()
        edit_log = {}
        selected = set(self.depths)
        if edit:
            if edit["layer"] not in range(len(self.layers)):
                raise ValueError("Invalid edit layer")
            selected.add(edit["layer"])

        def hook_at(index):
            def hook(module, args, output):
                if index in fired:
                    return output
                fired.add(index)
                x = output[0] if isinstance(output, tuple) else output
                if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != length:
                    raise RuntimeError("Expected a batch-one complete prefill")
                h = x[0, -1].float()
                if not t.isfinite(h).all():
                    raise RuntimeError("Nonfinite activation")
                states[index] = h.detach().cpu()
                if not edit or index != edit["layer"] or edit["mode"] == "identity":
                    return output
                r = t.as_tensor(edit["direction"], dtype=t.float32, device=h.device)
                if r.shape != h.shape or not t.isfinite(r).all() or not t.isclose(r.norm(), r.new_tensor(1.), atol=1e-4):
                    raise ValueError("Direction must be finite, unit length, and match hidden size")
                mode = edit["mode"]
                if mode == "replace":
                    donor = t.as_tensor(edit["donor"], dtype=t.float32, device=h.device)
                    if donor.shape != h.shape or not t.isfinite(donor).all():
                        raise ValueError("Invalid donor")
                    delta = t.dot(donor-h, r)
                elif mode == "remove":
                    delta = -t.dot(h, r)
                elif mode == "add":
                    delta = h.new_tensor(edit["amount"])
                else:
                    raise ValueError("Unknown edit")
                updated = h + delta*r
                if not t.isfinite(updated).all():
                    raise ValueError("Nonfinite edit")
                edit_log.update(mode=mode, layer=index, displacement=float(delta.abs()),
                                direction_sha256=sha(json.dumps(r.cpu().tolist())))
                y = x.clone()
                y[0, -1] = updated.to(x.dtype)
                return (y,) + output[1:] if isinstance(output, tuple) else y
            return hook
        try:
            for index in sorted(selected):
                handles.append(self.layers[index].register_forward_hook(hook_at(index)))
            t.cuda.reset_peak_memory_stats()
            t.cuda.synchronize()
            start = time.perf_counter()
            with t.inference_mode():
                out = self.model.generate(**inputs, do_sample=False, num_beams=1,
                    max_new_tokens=max_new_tokens, use_cache=True,
                    return_dict_in_generate=True, output_logits=True)
            t.cuda.synchronize()
            seconds = time.perf_counter()-start
        finally:
            for handle in handles:
                handle.remove()
        if fired != selected:
            raise RuntimeError("Not all hooks executed")
        new_ids = out.sequences[0, length:].tolist()
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        # Strip only known end-of-turn/EOS/PAD ids at the outside; do not erase
        # arbitrary special tokens or reasoning. A separate narrow normalizer below
        # recognizes Gemma's empty channel wrapper before strict A/B parsing.
        eos = self.model.generation_config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos])
        stop_ids.add(self.model.generation_config.pad_token_id)
        cleaned_ids = list(new_ids)
        while cleaned_ids and cleaned_ids[-1] in stop_ids:
            cleaned_ids.pop()
        cleaned = self.tokenizer.decode(cleaned_ids, skip_special_tokens=False)
        parsed_text, normalization = normalize_native_output(cleaned, self.model_id)
        logp = out.logits[0][0].float().log_softmax(-1)
        if not t.isfinite(logp).all():
            raise RuntimeError("Nonfinite next-token scores")
        scores = {c: float(logp[i]) for c, i in self.codes.items()}
        return {"raw_output": raw, "raw_parsed_text": cleaned,
            "parsed_text": parsed_text, "output_normalization": normalization, "output_ids": new_ids,
            "input_tokens": length, "input_ids_sha256": sha(json.dumps(inputs["input_ids"].tolist())),
            "logp_A": scores["A"], "logp_B": scores["B"],
            "ab_mass": sum(math.exp(v) for v in scores.values()),
            "end_of_answer": bool(new_ids and new_ids[-1] in stop_ids),
            "length_cap": len(new_ids) == max_new_tokens and new_ids[-1] not in stop_ids,
            "seconds": seconds, "peak_gpu_bytes": t.cuda.max_memory_allocated(),
            "edit": edit_log, "status": "ok"}, states

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="gemma")
    ap.add_argument("--compute-dtype", choices=["bfloat16", "float16"], default="bfloat16",
                    help="float16 is an explicitly non-protocol T4 compatibility run")
    ap.add_argument("--revision", default="main", help="Resolved to immutable HF SHA at run start")
    ap.add_argument("--input", default="examples.jsonl")
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--variants", nargs="+", choices=["B", "R", "N", "R2"], default=["B", "R"])
    ap.add_argument("--baseline", choices=["native", "content", "strict"], default="native")
    ap.add_argument("--format-reminder", action="store_true",
                    help="Experimental synthetic-only prompt adjustment; not the frozen research prompt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    records = validate_records([json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()])
    if args.format_reminder and any(r["study_partition"] != "synthetic" for r in records):
        raise ValueError("Format reminder is an unfrozen diagnostic; synthetic fixtures only")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=False)  # Never mix or overwrite runs.
    plans = []
    for r in records:
        for v in args.variants:
            value = variant(r["original_response"], v)
            for mapping in MAPPINGS:
                turns = messages(r["original_prompt"], value["assistant_response"], mapping, args.baseline, format_reminder=args.format_reminder)
                plans.append((r, value, mapping, turns))
    if args.dry_run:
        with (outdir/"planned_inputs.jsonl").open("w") as f:
            for r, value, mapping, turns in plans:
                f.write(json.dumps({"source_id": r["source_id"], **value, "mapping": mapping, "messages": turns}, ensure_ascii=False)+"\n")
        print(f"Validated {len(records)} records; planned {len(plans)} calls. NO model was run.")
        return
    if args.compute_dtype == "float16" and any(r["study_partition"] != "synthetic" for r in records):
        raise ValueError("T4/FP16 compatibility mode only permits synthetic fixtures; protocol experiments require BF16")
    runner = Runner(MODELS[args.model], args.revision, args.compute_dtype)
    runner.meta.update(prompt_format="reminder_v1" if args.format_reminder else "native",
                       protocol_prompt_match=not args.format_reminder)
    # Preflight all selected inputs before model calls; cross-model length
    # eligibility is a separate dataset intake gate, not implied by this check.
    for _, _, _, turns in plans:
        runner.encode(turns)
    print(f"Preflight passed for {len(plans)} inputs; running identity check", flush=True)
    (outdir/"run_config.json").write_text(json.dumps({**runner.meta, "args": vars(args),
        "policy": POLICY, "policy_sha256": sha(POLICY), "baseline": args.baseline,
        "input_file_sha256": sha(Path(args.input).read_text()), "validated_gpu": False}, indent=2))
    # Native/identity equality checks reproducibility and prefill hook execution.
    turns = plans[0][3]
    native, native_states = runner.run(turns)
    identity, identity_states = runner.run(turns, edit={"layer": runner.depths[0], "mode": "identity"})
    if native["output_ids"] != identity["output_ids"] or any(abs(native[k]-identity[k]) > 1e-5 for k in ("logp_A", "logp_B")):
        raise RuntimeError("Identity smoke check failed; stop this model")
    if any(not runner.torch.equal(native_states[k], identity_states[k]) for k in native_states):
        raise RuntimeError("Boundary state reproducibility failed")
    (outdir/"identity_check.json").write_text(json.dumps({"passed": True, "calls": 2}))
    with (outdir/"results.jsonl").open("w") as f:
        for i, (r, value, mapping, turns) in enumerate(plans):
            result, states = runner.run(turns)
            unsafe_code = "B" if mapping == "A_safe" else "A"
            safe_code = "A" if mapping == "A_safe" else "B"
            state_file = f"states_{i:05d}.pt"
            runner.torch.save(states, outdir/state_file)
            row = {**r, **value, **result, "mapping": mapping,
                "verdict": parse(result["parsed_text"], mapping),
                "semantic_margin": result[f"logp_{unsafe_code}"]-result[f"logp_{safe_code}"],
                "model_revision": runner.revision, "states_file": state_file,
                "prompt_sha256": sha(json.dumps(turns, ensure_ascii=False)), "run_index": i}
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
            f.flush()
            print(f"Completed input {i+1}/{len(plans)}: {row['verdict']}", flush=True)
    print(f"Completed {len(plans)} calls plus two identity checks. Inspect {outdir}; synthetic outcomes are not research results.")

if __name__ == "__main__":
    main()
