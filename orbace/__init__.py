# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""OrbACE: Equivariant Message-Passing Potential in Orbital Space.

Two production model families:

* ``OrbACE`` -- orbital-space E(3)-equivariant MACE (nodes = localized MOs),
  precomputed HF/localized orbital features, autograd forces through orbital
  centroids.
* ``AtomOrbACE`` -- atom-centered MACE on gauge-invariant tight-binding
  features, with ``regular``/``mf`` delta-learning and conservative forces.
"""

import torch as _torch
if hasattr(_torch, "serialization"):
    _torch.serialization.add_safe_globals([slice])

__version__ = "0.6.0"

from .modules.models import OrbACE  # noqa: F401, E402
from .modules.atom_model import AtomOrbACE  # noqa: F401, E402

__all__ = ["OrbACE", "AtomOrbACE"]
