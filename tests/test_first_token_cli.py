"""CLI smoke for first-token A/B judging. No model is loaded."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "data" / "manifests" / "examples.jsonl"


class FirstTokenCliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_inference.py"), *args],
            cwd=ROOT, capture_output=True, text=True, check=False, env=env,
        )

    def test_help_documents_first_token_protocol(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-protocol", result.stdout)
        self.assertIn("ab_first_token_v1", result.stdout)
        self.assertIn("native_v1", result.stdout)

    def test_dry_run_plans_constrained_first_token_without_a_model(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "planned"
            result = self.run_cli(
                "--dry-run", "--model", "gemma", "--variants", "B", "R",
                "--compute-dtype", "bfloat16", "--input", str(EXAMPLES),
                "--output-protocol", "ab_first_token_v1", "--out", str(out),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("NO model was run", result.stdout)
            self.assertIn("ab_first_token_v1", result.stdout)
            lines = (out / "planned_inputs.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 40)
            row = json.loads(lines[0])
            self.assertEqual(row["output_protocol"], "ab_first_token_v1")
            self.assertEqual(row["variant"], "B")
            self.assertIn("messages", row)

    def test_dry_run_default_stays_native(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "planned"
            result = self.run_cli("--dry-run", "--input", str(EXAMPLES), "--out", str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("native_v1", result.stdout)
            row = json.loads((out / "planned_inputs.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["output_protocol"], "native_v1")
            self.assertEqual(set(row), {"source_id", "assistant_response", "variant", "cue_text",
                                         "body_sha256", "variant_sha256", "body_byte_start",
                                         "body_byte_end", "mapping", "output_protocol", "messages"})

    def test_unknown_protocol_is_rejected_before_a_model(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.run_cli("--dry-run", "--output-protocol", "answer-only", "--out", str(Path(d) / "planned"))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((Path(d) / "planned").exists())


if __name__ == "__main__":
    unittest.main()
