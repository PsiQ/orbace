# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Pytest configuration for the OrbACE test suite.

Most files here are dual-purpose: runnable as scripts (python tests/<f>.py) and
collectible by pytest. The script form predates the pytest form, which is why
each one keeps its main() and its __main__ guard.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: runs a full prepare/train/eval cycle; deselect with -m 'not slow'",
    )
