#!/usr/bin/env python3
"""Compatibility wrapper for the renamed R5 data verifier."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).with_name("verify_rdnet_r5_data.py")
    runpy.run_path(str(target), run_name="__main__")
