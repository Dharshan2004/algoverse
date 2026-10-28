#!/usr/bin/env python3
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "refusal_cues" / "diagnostics" / "archive_ab_evidence.py"),
        run_name="__main__",
    )
