#!/usr/bin/env python3
"""Compatibility wrapper for the moved analysis entrypoint."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).resolve().parent / "scripts/analysis/aggregate_result_type_logs.py",
        run_name="__main__",
    )
