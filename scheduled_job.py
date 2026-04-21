#!/usr/bin/env python3
"""Cron-friendly scheduled runner for the drift pipeline."""

from __future__ import annotations

import datetime as dt
import fcntl
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOCK_PATH = ROOT / "scheduled_job.lock"
MAIN_PATH = ROOT / "main.py"


def get_python_command() -> list[str]:
    """Prefer project venv python on Linux, then fall back to current interpreter."""
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python)]
    return [sys.executable]


def append_log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "scheduled_job.log"
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def run_pipeline() -> int:
    cmd = get_python_command() + [str(MAIN_PATH)]
    append_log(f"Starting pipeline: {' '.join(cmd)}")

    # Use repo root as cwd so relative paths in project code behave consistently.
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.stdout:
        append_log("STDOUT:\n" + proc.stdout.strip())
    if proc.stderr:
        append_log("STDERR:\n" + proc.stderr.strip())

    append_log(f"Pipeline finished with exit code {proc.returncode}")
    return proc.returncode


def main() -> int:
    if not MAIN_PATH.exists():
        append_log(f"ERROR: {MAIN_PATH} not found")
        return 1

    with LOCK_PATH.open("w", encoding="utf-8") as lock_file:
        try:
            # Non-blocking lock: if another run is active, skip this trigger.
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            append_log("Another scheduled run is active; skipping this trigger")
            return 0

        return run_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
