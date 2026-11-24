# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Backward-compatibility shim.

``InvariantOrbACECalculator`` was renamed to :class:`AtomOrbACECalculator` and
moved to :mod:`orbace.calculators.atom_ase_calc`. Import from there in new code;
this alias is kept so existing imports keep working.
"""

from .atom_ase_calc import AtomOrbACECalculator  # noqa: F401

InvariantOrbACECalculator = AtomOrbACECalculator

__all__ = ["AtomOrbACECalculator", "InvariantOrbACECalculator"]
