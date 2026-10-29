"""GPU dependency declarations. No packages are installed."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class InferenceDependencyTests(unittest.TestCase):
    def test_ministral_backend_package_is_declared_for_gpu_installs(self):
        requirements = [
            line.split("#", 1)[0].strip()
            for line in (ROOT / "requirements.txt").read_text().splitlines()
        ]
        self.assertIn("mistral-common", requirements)
        pyproject = (ROOT / "pyproject.toml").read_text()
        extra = pyproject.split("[project.optional-dependencies]", 1)[1]
        inference = extra.split("inference = [", 1)[1].split("]", 1)[0]
        self.assertIn('"mistral-common"', inference)


if __name__ == "__main__":
    unittest.main()
