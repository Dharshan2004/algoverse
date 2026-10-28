import json
import tempfile
import unittest
from pathlib import Path

from refusal_cues.adapt_sabika import adapt_records, write_adapted
from refusal_cues.protocol import validate_records

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sabika_hf_development_sample.jsonl"


class AdaptSabikaTests(unittest.TestCase):
    def test_hf_rows_map_to_runner_schema(self):
        raw = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
        adapted = adapt_records(raw, partition="development")
        self.assertEqual(len(adapted), 2)
        validate_records(adapted)
        self.assertEqual(adapted[0]["original_prompt"], raw[0]["user_prompt"])
        self.assertEqual(adapted[0]["original_response"], raw[0]["model_response"])
        self.assertIn("meta", adapted[0])
        self.assertEqual(adapted[0]["meta"]["response_sha256"], "abc")
        self.assertNotIn("user_prompt", adapted[0])

    def test_final_test_writes_with_warning_no_validate(self):
        row = json.loads(FIXTURE.read_text().splitlines()[0])
        row = {**row, "source_id": "ft-01", "study_partition": "final_test"}
        adapted = adapt_records([row], partition="final_test")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "final_test_inputs.jsonl"
            with self.assertRaises(ValueError):
                validate_records(adapted)
            write_adapted(adapted, out, partition="final_test", validate_runner=False)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
