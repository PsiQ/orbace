# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Backward-compatibility shim.

The atom-centered model was renamed ``InvariantOrbACE`` -> ``AtomOrbACE`` and
moved to :mod:`orbace.modules.atom_model`. This module is kept only so that
checkpoints saved *before* the rename (which pickle the whole model object and
therefore hard-code the class path ``orbace.modules.invariant_model.*``) still
unpickle. New code should import from :mod:`orbace.modules.atom_model`.
"""

from .atom_model import (  # noqa: F401
    AtomOrbACE,
    AtomFeatureEmbedding,
    FeatureStandardizer,
    MFScalarHead,
    InvariantEdgeEmbedding,
)

# Old pickles look up this exact name on this exact module.
InvariantOrbACE = AtomOrbACE

__all__ = [
    "AtomOrbACE",
    "InvariantOrbACE",
    "AtomFeatureEmbedding",
    "FeatureStandardizer",
    "MFScalarHead",
    "InvariantEdgeEmbedding",
]
