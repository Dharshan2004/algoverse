#!/usr/bin/env python3
"""Run hand-checkable metrics unit checks (legacy test_metrics runner)."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "tests" / "test_metrics.py"), run_name="__main__")
