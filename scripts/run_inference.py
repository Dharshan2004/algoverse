#!/usr/bin/env python3
"""Thin CLI for GPU inference and dry-run input planning.

`--output-protocol ab_first_token_v1` forces the first generated token to A or B
and allows trailing text. The default `native_v1` path is unchanged.
"""
from refusal_cues.inference import main

if __name__ == "__main__":
    main()
