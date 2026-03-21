"""Resilience utilities for the experiment runner.

Provides crash-safe file operations, heartbeat monitoring, and signal handling
so overnight runs survive API outages, power losses, and terminal kills.

Components:
  - atomic_write(): Crash-safe file writes (write to .tmp, fsync, rename)
  - validate_results_tsv(): Detect and fix corrupted results files
  - Heartbeat: Status file for external monitors (monitor.py)
  - install_signal_handlers(): Graceful shutdown on SIGTERM/SIGHUP/SIGINT
"""

import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Atomic file writer
# ---------------------------------------------------------------------------

def atomic_write(path: str, content: str) -> None:
    """Write content to a file atomically via tmp + rename.

    On a single Mac filesystem, os.rename() is atomic — so the file is
    either fully written or not changed at all. No corruption from
    mid-write crashes.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, path)


def atomic_append(path: str, line: str) -> None:
    """Append a line to a file atomically (read + append + atomic write).

    Slower than bare append but crash-safe. Acceptable for results.tsv
    which gets one append per ~5-minute experiment.
    """
    existing = ""
    if os.path.exists(path):
        with open(path) as f:
            existing = f.read()
    atomic_write(path, existing + line)


# ---------------------------------------------------------------------------
# TSV validation
# ---------------------------------------------------------------------------

EXPECTED_FIELDS = 9  # exp, description, val_bpb, peak_mem_gb, tok_sec, mfu, steps, status, notes

def validate_results_tsv(path: str) -> tuple[bool, list[str]]:
    """Validate a results.tsv file and fix minor corruption.

    Checks:
    - Header line matches expected format
    - Each data line has exactly 9 tab-separated fields
    - Numeric fields parse correctly
    - No incomplete trailing lines (truncated by crash)

    Returns (is_valid, list_of_warnings). Fixes issues in-place
    (truncates partial trailing lines).
    """
    if not os.path.exists(path):
        return True, []

    warnings = []

    with open(path) as f:
        content = f.read()

    if not content.strip():
        return True, []

    lines = content.split("\n")

    # Check for incomplete trailing line (no newline at end = mid-write crash)
    if content and not content.endswith("\n"):
        last_line = lines[-1]
        fields = last_line.split("\t")
        if len(fields) < EXPECTED_FIELDS:
            warnings.append(f"Truncated trailing line removed: '{last_line[:60]}...'")
            lines = lines[:-1]
            # Rewrite without the corrupt line
            fixed = "\n".join(lines)
            if not fixed.endswith("\n"):
                fixed += "\n"
            atomic_write(path, fixed)

    # Validate header
    if lines and lines[0].strip():
        header_fields = lines[0].strip().split("\t")
        if header_fields[0] != "exp":
            warnings.append(f"Unexpected header: {lines[0][:60]}")

    # Validate data lines
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != EXPECTED_FIELDS:
            warnings.append(f"Line {i}: expected {EXPECTED_FIELDS} fields, got {len(fields)}")
            continue
        # Validate numeric fields
        try:
            float(fields[2])  # val_bpb
            float(fields[3])  # peak_mem_gb
            int(fields[4])    # tok_sec
            float(fields[5])  # mfu
            int(fields[6])    # steps
        except (ValueError, IndexError):
            warnings.append(f"Line {i}: numeric parse error in '{fields[0]}'")

    is_valid = len(warnings) == 0
    return is_valid, warnings


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class Heartbeat:
    """Lightweight status file for external monitors.

    Writes a JSON file atomically at a configurable interval so that
    monitor.py (or any external tool) can check if the runner is alive,
    what experiment it's on, and when it last updated.

    Usage:
        hb = Heartbeat()
        hb.update(experiment=5, status="training", dataset="climbmix", best_bpb=1.296)
        ...
        hb.close()
    """

    DEFAULT_PATH = ".runner_status.json"

    def __init__(self, path: str | None = None):
        self._path = path or self.DEFAULT_PATH
        self._started_at = datetime.now().isoformat()
        self._pid = os.getpid()

    def update(
        self,
        experiment: int = 0,
        status: str = "running",
        dataset: str = "",
        best_bpb: float = float("inf"),
        model: str = "",
        total: int = 0,
        kept: int = 0,
        discarded: int = 0,
        crashes: int = 0,
    ) -> None:
        """Write current status to the heartbeat file."""
        data = {
            "pid": self._pid,
            "alive": True,
            "started_at": self._started_at,
            "last_updated": datetime.now().isoformat(),
            "experiment": experiment,
            "status": status,
            "dataset": dataset,
            "model": model,
            "best_bpb": best_bpb if best_bpb < float("inf") else None,
            "total": total,
            "kept": kept,
            "discarded": discarded,
            "crashes": crashes,
        }
        try:
            atomic_write(self._path, json.dumps(data, indent=2) + "\n")
        except Exception:
            pass  # Heartbeat is best-effort, never crash the runner

    def close(self) -> None:
        """Mark the runner as no longer alive."""
        try:
            if os.path.exists(self._path):
                with open(self._path) as f:
                    data = json.load(f)
                data["alive"] = False
                data["last_updated"] = datetime.now().isoformat()
                data["status"] = "stopped"
                atomic_write(self._path, json.dumps(data, indent=2) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_shutdown_requested = False
_shutdown_time = 0.0


def install_signal_handlers(stop_callback: Callable[[], None]) -> None:
    """Install graceful shutdown handlers for SIGINT, SIGTERM, SIGHUP.

    First signal: set graceful shutdown flag, call stop_callback.
    Second signal within 5s: force exit.

    Must be called from the main thread.
    """
    global _shutdown_requested, _shutdown_time

    def _handler(signum, frame):
        global _shutdown_requested, _shutdown_time
        sig_name = signal.Signals(signum).name

        if _shutdown_requested and (time.time() - _shutdown_time) < 5.0:
            # Second signal — force exit
            print(f"\n[SIGNAL] Received {sig_name} again — forcing exit", flush=True)
            sys.exit(1)

        _shutdown_requested = True
        _shutdown_time = time.time()
        print(f"\n[SIGNAL] Received {sig_name} — finishing current experiment, then stopping...", flush=True)
        stop_callback()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # SIGHUP = terminal disconnected (SSH timeout, etc.)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handler)
