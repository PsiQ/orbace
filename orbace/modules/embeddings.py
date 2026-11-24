# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Orbital embedding modules for OrbACE.

Maps (orbital_species, orbital_features, ao_shell_encoding) into e3nn
irreducible representations suitable for equivariant message passing.
"""

import torch
import torch.nn as nn

if hasattr(torch, "serialization"):
    torch.serialization.add_safe_globals([slice])

from e3nn import o3


class OrbitalEmbeddingBlock(nn.Module):
    """Full orbital embedding: species one-hot + feature projection -> irreps.

    The species one-hot is used as node_attrs (for symmetric contraction in
    MACE interaction blocks). The combined embedding of species + features
    produces the initial node_feats in scalar irreps.
    """

    def __init__(
        self,
        num_species: int,
        n_scalar_features: int,
        n_ao_shell_features: int,
        hidden_dim: int,
        node_feats_irreps: o3.Irreps,
    ):
        super().__init__()
        self.num_species = num_species
        self.node_feats_irreps = node_feats_irreps
        n_scalars = node_feats_irreps.count(o3.Irrep(0, 1))

        total_input = num_species + n_scalar_features + n_ao_shell_features
        self.embed_mlp = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_scalars),
        )

        self.node_attr_irreps = o3.Irreps([(num_species, (0, 1))])

    def forward(self, orbital_species, orbital_features, ao_shell_encoding):
        species_clamped = orbital_species.clamp(0, self.num_species - 1)
        node_attrs = torch.nn.functional.one_hot(
            species_clamped, num_classes=self.num_species
        ).to(orbital_features.dtype)

        combined = torch.cat([node_attrs, orbital_features, ao_shell_encoding], dim=-1)
        node_feats = self.embed_mlp(combined)
        return node_attrs, node_feats
