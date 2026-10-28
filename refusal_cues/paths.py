"""Repository-root paths for manifests, controls, and default fixtures."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent

DEFAULT_EXAMPLES = REPO_ROOT / "data" / "manifests" / "examples.jsonl"
CONTROLS_AB_V1 = REPO_ROOT / "data" / "controls" / "ab_v1.jsonl"
