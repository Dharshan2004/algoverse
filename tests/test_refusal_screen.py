import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from refusal_cues.guard_judges import parse_guard
from refusal_cues.prepare_refusal_screen import adapt_wildguard
from refusal_cues.refusal_screen import DEFAULT_CONFIG, make_plans, run_screen, sample_records
from refusal_cues.screen_report import summarize
from refusal_cues.structured_output import parse_structured, structured_messages

CONFIG = json.loads(DEFAULT_CONFIG.read_text())
RECORD = {"source_id": "fixture", "cluster_id": "group", "original_prompt": "A saved request",
          "original_response": "Saved response café\n", "mapped_unsafe": True,
          "study_partition": "synthetic", "label_provenance": "synthetic"}


class ScreenTests(unittest.TestCase):
    def test_json_mapping_swaps_semantics(self):
        self.assertEqual(parse_structured('{"verdict":"A"}', "A_safe")["verdict"], "SAFE")
        self.assertEqual(parse_structured('{"verdict":"A"}', "A_unsafe")["verdict"], "UNSAFE")

    def test_invalid_json_is_never_repaired_into_a_verdict(self):
        for text in ['A', '```json\n{"verdict":"A"}\n```', '{"verdict":"SAFE"}',
                     '{"verdict":"A","reason":"x"}', '{"verdict":"A","verdict":"B"}',
                     '{"verdict":1}', '{"verdict":"A"', 'null']:
            with self.subTest(text=text):
                self.assertEqual(parse_structured(text, "A_safe")["verdict"], "INVALID")

    def test_plan_preserves_body_swaps_and_repetitions(self):
        plans = list(make_plans([RECORD], "gemma", CONFIG))
        self.assertEqual(len(plans), 24)
        self.assertEqual({p["mapping"] for p in plans}, {"A_safe", "A_unsafe"})
        self.assertEqual({p["repeat"] for p in plans}, {0, 1})
        for plan in plans:
            evidence = json.loads(plan["messages"][1]["content"].split("\n", 1)[1])
            response = evidence["assistant_response"].encode()
            self.assertEqual(response[plan["body_byte_start"]:plan["body_byte_end"]], RECORD["original_response"].encode())
            self.assertNotIn("Return exactly one uppercase letter", plan["messages"][0]["content"])

    def test_native_guards_do_not_receive_chat_judge_json_prompt(self):
        plans = list(make_plans([RECORD], "llama_guard", CONFIG))
        self.assertEqual(len(plans), 12)
        self.assertEqual({p["mapping"] for p in plans}, {"native"})
        self.assertEqual([t["role"] for t in plans[0]["messages"]], ["user", "assistant"])

    def test_guard_plans_preserve_text_in_native_content_format(self):
        for model in ("llama_guard", "shieldgemma"):
            for plan in make_plans([RECORD], model, CONFIG):
                contents = [turn["content"] for turn in plan["messages"]]
                if model == "llama_guard":
                    self.assertTrue(all(isinstance(c, list) for c in contents))
                    contents = [c[0]["text"] for c in contents]
                self.assertEqual(contents[0], RECORD["original_prompt"])
                response = contents[1].encode()
                self.assertEqual(response[plan["body_byte_start"]:plan["body_byte_end"]],
                                 RECORD["original_response"].encode())

    def test_llama_guard_rejects_silently_dropped_content(self):
        from refusal_cues.guard_judges import GuardRunner
        runner = GuardRunner.__new__(GuardRunner)
        runner.name = "llama_guard"
        with self.assertRaisesRegex(ValueError, "typed text blocks"):
            runner.encode([{"role": "user", "content": "Hello"}])
        runner.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: "empty conversation")
        with self.assertRaisesRegex(ValueError, "omitted conversation"):
            runner.encode([{"role": "user", "content": [{"type": "text", "text": "Hello"}]}])

    def test_llama_guard_parser(self):
        for text, verdict in [("safe", "SAFE"), ("unsafe\nS1,S2", "UNSAFE"),
                              ("safe but actually unsafe", "INVALID"), ("not unsafe", "INVALID")]:
            self.assertEqual(parse_guard(text, "llama_guard"), verdict)

    def test_balanced_selection_is_deterministic(self):
        rows = [{**RECORD, "source_id": str(i), "mapped_unsafe": bool(i % 2)} for i in range(40)]
        first = sample_records(rows, 20, 42)
        self.assertEqual(first, sample_records(list(reversed(rows)), 20, 42))
        self.assertEqual(sum(r["mapped_unsafe"] for r in first), 10)
        with self.assertRaises(ValueError):
            sample_records(rows, 41, 42)

    def test_wildguard_uses_response_harm_not_prompt_harm(self):
        raw = {"source_row_id": "1", "question_group_id": "g", "prompt": "request", "response": "response",
               "study_split": "independent_safety", "response_harm_label": "unharmful",
               "prompt_harm_label": "harmful", "response_refusal_label": "refusal"}
        row = adapt_wildguard([raw])[0]
        self.assertFalse(row["mapped_unsafe"])
        self.assertTrue(row["genuine_refusal"])
        with self.assertRaises(ValueError):
            adapt_wildguard([{**raw, "study_split": "reader_validation"}])

    def test_live_orchestration_uses_json_constrained_generation(self):
        class FakeRunner:
            def __init__(self, *args):
                self.meta, self.revision = {}, "fixture_revision"
                self.model_max_position_embeddings = 1000
                self.torch = SimpleNamespace(manual_seed=lambda x: None)

            def encode(self, turns):
                return {"input_ids": SimpleNamespace(shape=(1, 20))}

            def run(self, turns, **kwargs):
                assert kwargs == {"max_new_tokens": 256}
                return {"parsed_text": '{"verdict":"A"}', "raw_output": '{"verdict":"A"}',
                        "logp_A": -10, "logp_B": -11, "ab_mass": 0.001}, {}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "input.jsonl"
            source.write_text(json.dumps(RECORD) + "\n")
            args = SimpleNamespace(config=DEFAULT_CONFIG, repetitions=1, input=source, limit=None,
                                   model="gemma", out=path / "run", dataset="synthetic", dry_run=False,
                                   revision="main", save_states=False)
            with patch("refusal_cues.judges.Runner", FakeRunner):
                run_screen(args)
            rows = [json.loads(x) for x in (args.out / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertEqual([r["verdict"] for r in rows[:2]], ["SAFE", "UNSAFE"])
            self.assertEqual(rows[0]["logp_A"], -10)
            self.assertTrue((args.out / "complete.json").exists())


class ReportTests(unittest.TestCase):
    def rows(self):
        rows = []
        for i in range(4):
            for cue in CONFIG["cues"]:
                # Two baseline-detected harmful examples with disjoint cue successes.
                verdict = "UNSAFE" if i < 2 else "SAFE"
                if (i == 0 and cue == "R") or (i == 1 and cue == "R2"):
                    verdict = "SAFE"
                rows.append({"source_id": str(i), "model": "gemma", "dataset": "synthetic",
                             "mapping": "A_safe", "repeat": 0, "variant": cue,
                             "mapped_unsafe": i != 3, "body_sha256": str(i), "verdict": verdict})
        return rows

    def test_dfr_is_union_not_sum_and_baseline_denominator_is_explicit(self):
        report = summarize(self.rows(), CONFIG)
        stratum = report["strata"][0]
        self.assertEqual(stratum["gold_harmful"], 3)
        self.assertEqual(stratum["baseline_detected_harmful"], 2)
        self.assertEqual(stratum["per_cue"]["R"]["dfr"], 0.5)
        self.assertEqual(report["selected_trio"]["mean_dfr_at_3"], 1)
        self.assertEqual(len(stratum["all_trios"]), 10)
        self.assertEqual(stratum["per_cue"]["R"]["net_safe_rate_delta_pp"], 25)
        self.assertIsNone(stratum["per_cue"]["R"]["genuine_refusal_safe_rate"])

    def test_invalid_family_excluded_consistently_for_all_trios(self):
        rows = self.rows()
        rows[1]["verdict"] = "INVALID"
        report = summarize(rows, CONFIG)
        self.assertEqual(report["families_excluded_for_invalid"], 1)
        self.assertEqual(report["invalid_rows"], 1)
        self.assertEqual(report["strata"][0]["baseline_detected_harmful"], 1)

    def test_no_eligible_harmful_is_null_not_zero(self):
        rows = self.rows()
        for row in rows:
            row["verdict"] = "SAFE"
        self.assertIsNone(summarize(rows, CONFIG)["selected_trio"])

    def test_duplicates_and_body_changes_are_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]], CONFIG)
        rows[1]["body_sha256"] = "changed"
        with self.assertRaises(ValueError):
            summarize(rows, CONFIG)


if __name__ == "__main__":
    unittest.main()
