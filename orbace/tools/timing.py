# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Lightweight, opt-in timing instrumentation for prepare/train/eval/md.

Sub-steps inside prepare/train/eval/md record their wall time here. Recording
is a NO-OP unless the environment variable ``ORBACE_TIMING_CSV`` points at a
file, so standalone runs are unaffected and never crash on timing.

CSV schema (one row per sub-step):

    method,phase,label,context,seconds,status

The model family is not a column; encode it in the CSV file name if several
runs are collected together. ``context`` is a short tag (e.g. output-dir or
checkpoint-dir basename) to trace a row back to a config.
"""
import contextlib
import csv
import os
import threading
import time

_LOCK = threading.Lock()
_HEADER = ["method", "phase", "label", "context", "seconds", "status"]


def _csv_path():
    return os.environ.get("ORBACE_TIMING_CSV")


def record_timing(method, phase, label, seconds, context="", status="ok"):
    """Append a single timing row when ORBACE_TIMING_CSV is set; else no-op."""
    path = _csv_path()
    if not path:
        return
    try:
        with _LOCK:
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(_HEADER)
                w.writerow([method, phase, label, context, f"{float(seconds):.3f}", status])
    except OSError:
        # Never let timing bookkeeping break a real run.
        pass


@contextlib.contextmanager
def timed(method, phase, label, context=""):
    """Context manager: record wall time of the enclosed block."""
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        record_timing(method, phase, label, time.perf_counter() - t0, context, status)
