# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
# Portions adapted from MACE remain under the MIT License; see NOTICE.
"""OrbACE: Orbital-space MACE with quantum-chemistry features.

Node features (orbital-level, 10 scalar + 15 AO shell + 5 coeff norms):
  - Orbital energy F_ii, occupancy, h_ii, g_iiii, <r^2>,
    Mulliken partition energy n_i*(h_ii+F_ii)/2
  - Orbital spread, 3 quadrupole eigenvalues (shape invariants)
  - AO shell composition encoding (binary)
  - MO coefficient norms per angular momentum channel

Edge features (orbital-pair, 5 chemical + Bessel radial):
  - Fock matrix off-diagonal F_ij (orbital coupling strength)
  - Coulomb integrals (ii|jj) (density-density repulsion)
  - Exchange integrals (ij|ji) (same-spin interaction)
  - Core Hamiltonian h_ij (separates 1e from 2e effects)
  - Energy gap |eps_i - eps_j| (perturbation theory coupling)

Energy baseline: per-orbital-species constant E0 (no orbital-energy slope).
The NN learns all geometry-dependent energy from features and edges.

The equivariant backbone is assembled from the interaction, product and
readout blocks of MACE (https://github.com/ACEsuit/mace, MIT License,
Copyright (c) 2022 ACEsuit/mace), following the layout of MACE's own model
class; see NOTICE.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

if hasattr(torch, "serialization"):
    torch.serialization.add_safe_globals([slice])

from e3nn import o3
from e3nn.io import CartesianTensor
from e3nn.o3 import Linear as e3nn_Linear

from mace.modules.blocks import (
    EquivariantProductBasisBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.utils import (
    get_outputs,
)
from mace.tools.scatter import scatter_sum

# Multi-fidelity modes.
#   "regular" -- delta on a baseline that is RECOMPUTED at inference.
#   "mfdh"    -- multi-fidelity delta head: a learned base head replaces that
#                recompute and is ADDED into the total, so the total is
#                (readout + delta) + base.
#   "mfah"    -- multi-fidelity absolute head: a parallel head predicts the
#                cheap level's ABSOLUTE energy and never enters the total, so
#                the high-fidelity path is identical to a direct model and the
#                cheap level acts purely as an auxiliary task.
FIDELITY_MODES = ("regular", "mfdh", "mfah")

# What "mf" meant before the mfdh/mfah split. Checkpoints are pickled modules,
# so every model trained before the rename has this string in its __dict__ and
# restores WITHOUT going through __init__ -- forward() must therefore accept it.
_LEGACY_MF = "mf"


def _canonical_fidelity(mode: str) -> str:
    """Normalize a fidelity mode, mapping the legacy 'mf' onto 'mfdh'."""
    mode = str(mode)
    if mode == _LEGACY_MF:
        return "mfdh"
    if mode not in FIDELITY_MODES:
        raise ValueError(
            f"fidelity mode must be one of {FIDELITY_MODES} (or the legacy "
            f"'mf', an alias for 'mfdh'), got {mode!r}")
    return mode


def model_fidelity_mode(model) -> str:
    """Canonical fidelity mode of a loaded model.

    Use this at every inference site instead of reading ``delta_method``: a
    checkpoint is a pickled module, so pre-split models carry the legacy "mf"
    and models predating the flag carry nothing at all.
    """
    return _canonical_fidelity(getattr(model, "delta_method", "regular"))


def model_is_multifidelity(model) -> bool:
    """True when a loaded model predicts the cheap fidelity with a learned head.

    Both MF arms make the physics baseline redundant at inference, so callers
    that would otherwise recompute and re-add it must skip doing so.
    """
    return model_fidelity_mode(model) in ("mfdh", "mfah")


class OrbitalEnergiesBlock(nn.Module):
    """Per-orbital-species constant baseline energy (frozen).

    E0_i = a[species_i]

    One constant per orbital species, analogous to MACE's per-element E0.
    The orbital-energy slope (b * F_ii) was removed because sum(F_ii)
    double-counts electron repulsion and creates a geometry-dependent
    error in the frozen baseline that is invisible to force training.
    The NN already receives F_ii as input feature 0 and can learn the
    correct nonlinear mapping to the target energy.
    """

    def __init__(
        self,
        num_species: int,
        initial_a: Optional[torch.Tensor] = None,
        initial_b: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        a = torch.zeros(num_species, dtype=torch.get_default_dtype())
        if initial_a is not None:
            n = min(len(initial_a), num_species)
            a[:n] = initial_a[:n]
        self.register_buffer("e0_base", a)
        # Keep e0_slope buffer (zeroed) for backward compat with saved checkpoints
        self.register_buffer("e0_slope", torch.zeros(num_species, dtype=torch.get_default_dtype()))

    def forward(
        self, orbital_species: torch.Tensor, orbital_energy: torch.Tensor,
    ) -> torch.Tensor:
        # Clamp guards against orbital species indices that exceed the table
        # (the encoding Z*15+l*3+occ can exceed num_species for heavy elements);
        # without it indexing would raise. No-op when the table is large enough.
        species = orbital_species.clamp(max=self.e0_base.shape[0] - 1)
        return self.e0_base[species]


class OrbitalFeatureEmbedding(nn.Module):
    """Embed orbital species + scalar features + coeff norms into node features.

    Per-group normalization: instead of a single LayerNorm over the
    concatenated feature vector, each feature group is normalized separately
    to prevent scale mixing between incompatible units.

    Groups:
      A: orbital energies/integrals (first 6 scalar features, Hartree scale)
      B: binary AO shell encoding (15 features, {0,1})
      C: MO coefficient norms per l (5 features, positive, sum~1)
      D: orbital shape invariants (last 4 scalar features, Bohr^2 scale)
    """

    def __init__(
        self,
        num_species: int,
        n_scalar_features: int,
        n_ao_shell_features: int,
        n_coeff_norm_features: int,
        irreps_out: o3.Irreps,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_species = num_species
        irreps_in = o3.Irreps(f"{num_species}x0e")
        self.species_embedding = e3nn_Linear(
            irreps_in=irreps_in, irreps_out=irreps_out
        )
        feat_dim = n_scalar_features + n_ao_shell_features + n_coeff_norm_features
        num_features = irreps_out.count(o3.Irrep(0, 1))

        n_base_scalar = min(n_scalar_features, 6)
        n_shape = n_scalar_features - n_base_scalar
        self.n_base_scalar = n_base_scalar
        self.n_shape = n_shape
        self.n_ao_shell_features = n_ao_shell_features
        self.n_coeff_norm_features = n_coeff_norm_features

        self.norm_A = nn.LayerNorm(n_base_scalar) if n_base_scalar > 0 else nn.Identity()
        self.norm_B = nn.LayerNorm(n_ao_shell_features) if n_ao_shell_features > 0 else nn.Identity()
        self.norm_C = nn.LayerNorm(n_coeff_norm_features) if n_coeff_norm_features > 0 else nn.Identity()
        self.norm_D = nn.LayerNorm(n_shape) if n_shape > 0 else nn.Identity()

        layers = [
            nn.Linear(feat_dim, num_features * 2),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
        ]
        self.feature_proj = nn.Sequential(*layers)

    def forward(
        self,
        orbital_species: torch.Tensor,
        orbital_features: torch.Tensor,
        ao_shell_encoding: torch.Tensor,
        coeff_norms_per_l: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Clamp species into the one-hot range. The encoding Z*15+l*3+occ can
        # exceed num_species for elements heavier than ~Mg; F.one_hot would
        # otherwise raise. No-op when num_species covers all present species.
        species = orbital_species.clamp(min=0, max=self.num_species - 1)
        one_hot = torch.nn.functional.one_hot(
            species, self.num_species
        ).to(orbital_features.dtype)
        node_attrs = one_hot
        node_feats = self.species_embedding(one_hot)

        group_a = self.norm_A(orbital_features[:, :self.n_base_scalar])
        parts = [group_a]
        if self.n_shape > 0:
            group_d = self.norm_D(orbital_features[:, self.n_base_scalar:])
            parts.append(group_d)
        group_b = self.norm_B(ao_shell_encoding)
        group_c = self.norm_C(coeff_norms_per_l)
        parts.insert(1, group_b)
        parts.insert(2, group_c)

        extra = torch.cat(parts, dim=-1)
        node_feats = node_feats + self.feature_proj(extra)
        return node_attrs, node_feats


class EquivariantOrbitalEmbedding(nn.Module):
    """Embed per-orbital equivariant electronic features into node irreps.

    Takes two frozen HF quantities expressed in the lab frame:
      - polarization vector p_i (l=1): centroid minus dominant nucleus [Bohr]
      - traceless quadrupole Q_i (l=2): orbital shape orientation [Bohr^2]

    Each Cartesian quantity is mapped to its e3nn spherical (irrep) basis via a
    fixed CartesianTensor change-of-basis, rescaled by a fixed per-l constant
    for numerical conditioning, then mixed into ``n_channels`` copies of 1o/2e
    by an equivariant linear. The output irreps are ``n_channels x 1o +
    n_channels x 2e`` and are concatenated onto the scalar node features.

    The Cartesian inputs are treated as constants w.r.t. atomic positions
    (detached), matching how every other QM feature is handled, so the autograd
    force path is unchanged. Equivariance holds because each input transforms as
    a proper irrep and only equivariant ops (scalar rescale + e3nn Linear) act
    on it.
    """

    def __init__(
        self,
        n_channels: int,
        pol_scale: float = 1.0,
        quad_scale: float = 2.0,
    ):
        super().__init__()
        self.n_channels = n_channels

        # Fixed Cartesian -> spherical change-of-basis matrices (see
        # CartesianTensor.from_cartesian). Stored as plain buffers so they
        # follow the module across devices/dtypes.
        ct_vec = CartesianTensor("i")          # -> 1x1o
        ct_quad = CartesianTensor("ij=ji")     # -> 1x0e + 1x2e
        cob_vec = ct_vec.reduced_tensor_products().change_of_basis  # (3, 3)
        cob_quad = ct_quad.reduced_tensor_products().change_of_basis  # (6, 3, 3)
        # vector -> 1o (all 3 rows)
        self.register_buffer("cob_vec", cob_vec.reshape(3, 3).clone())
        # symmetric tensor -> keep only the 2e rows (drop row 0 = l=0 trace)
        self.register_buffer("cob_quad_2e", cob_quad.reshape(6, 9)[1:].clone())

        self.register_buffer("pol_scale", torch.tensor(float(pol_scale)))
        self.register_buffer("quad_scale", torch.tensor(float(quad_scale)))

        irreps_in = o3.Irreps("1x1o + 1x2e")
        self.irreps_out = o3.Irreps(f"{n_channels}x1o + {n_channels}x2e")
        self.equiv_linear = e3nn_Linear(irreps_in=irreps_in, irreps_out=self.irreps_out)

    def forward(
        self,
        orbital_quadrupole: torch.Tensor,
        orbital_polarization: torch.Tensor,
    ) -> torch.Tensor:
        # Frozen features: detach so forces flow only through positions.
        pol = orbital_polarization.detach().to(self.cob_vec.dtype)
        quad = orbital_quadrupole.detach().to(self.cob_quad_2e.dtype)

        # Cartesian -> spherical (irrep) basis: data @ change_of_basis.T
        pol_irrep = (pol / self.pol_scale) @ self.cob_vec.T          # (n, 3) -> 1o
        quad_2e = (quad / self.quad_scale).flatten(-2) @ self.cob_quad_2e.T  # (n, 5) -> 2e

        equiv_in = torch.cat([pol_irrep, quad_2e], dim=-1)
        return self.equiv_linear(equiv_in)


class ChemicalEdgeEmbedding(nn.Module):
    """Project quantum-chemical edge features into the radial feature space.

    Features (6 total):
      - Fock matrix element F_ij (orbital coupling)
      - Coulomb integral (ii|jj) (density-density repulsion)
      - Exchange integral (ij|ji) (same-spin interaction)
      - Core Hamiltonian h_ij (separates 1e from 2e effects)
      - Energy gap |eps_i - eps_j| (perturbation theory coupling)
      - MO overlap S_ij (localized-orbital non-orthogonality)

    Produces a learned embedding fused with Bessel radial basis via additive
    residual correction. This avoids hard multiplicative bottlenecks in the
    chemistry pathway while preserving expressive feature interactions.
    """

    N_CHEM_FEATURES = 6

    def __init__(self, radial_dim: int, n_chem_features: int = 6):
        super().__init__()
        self.n_chem_features = n_chem_features
        self.feat_norm = nn.LayerNorm(n_chem_features)
        self.base_proj = nn.Sequential(
            nn.Linear(n_chem_features, radial_dim),
            nn.SiLU(),
            nn.Linear(radial_dim, radial_dim),
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(n_chem_features, radial_dim),
            nn.SiLU(),
            nn.Linear(radial_dim, radial_dim),
        )
        self.max_residual = 0.5
        self.residual_mix_logit = nn.Parameter(torch.tensor(-2.2))
        with torch.no_grad():
            self.residual_proj[-1].weight.zero_()
            self.residual_proj[-1].bias.zero_()

    def forward(
        self,
        edge_fock: Optional[torch.Tensor],
        edge_coulomb: Optional[torch.Tensor],
        edge_exchange: Optional[torch.Tensor],
        edge_hcore: Optional[torch.Tensor],
        edge_energy_gap: Optional[torch.Tensor],
        n_edges: int,
        device: torch.device,
        dtype: torch.dtype,
        edge_overlap: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # F_ij, h_ij and S_ij change sign with the arbitrary phase of either
        # orbital; only their magnitudes are invariant.
        if edge_fock is not None:
            edge_fock = edge_fock.abs()
        if edge_hcore is not None:
            edge_hcore = edge_hcore.abs()
        if edge_overlap is not None:
            edge_overlap = edge_overlap.abs()
        parts = []
        for feat in [edge_fock, edge_coulomb, edge_exchange, edge_hcore,
                     edge_energy_gap, edge_overlap]:
            if feat is not None:
                parts.append(feat.unsqueeze(-1).to(dtype))
            else:
                parts.append(torch.zeros(n_edges, 1, device=device, dtype=dtype))
        # Keep only the configured number of channels (backward-compatible with
        # 5-feature checkpoints that predate the overlap edge feature).
        parts = parts[: self.n_chem_features]
        chem_feats = torch.cat(parts, dim=-1)
        chem_feats = self.feat_norm(chem_feats)
        base = self.base_proj(chem_feats)
        residual = self.residual_proj(chem_feats)
        residual_mix = self.max_residual * torch.sigmoid(self.residual_mix_logit)
        return base + residual_mix * residual


class OccupationGate(nn.Module):
    """Identity-centered occupation modulation for orbital energy contributions.

    Learns a bounded residual modulation from occupation number and orbital
    energy. The output stays centered at 1.0 so the readout remains stable at
    initialization and gradually learns to emphasize or de-emphasize orbitals.
    """

    def __init__(self, n_input: int = 2, max_residual: float = 0.5):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_input)
        self.gate = nn.Sequential(
            nn.Linear(n_input, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.max_residual = max_residual
        self.mix_logit = nn.Parameter(torch.tensor(-2.2))
        # Start exactly at identity and let the model learn small residual
        # deviations away from uniform orbital weighting.
        with torch.no_grad():
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.zero_()

    def forward(self, orbital_features: torch.Tensor) -> torch.Tensor:
        """Return per-orbital residual weight centered at 1.0.

        Args:
            orbital_features: (n_orb, n_feat) — columns 0=energy, 1=occupancy
        """
        occ_energy = self.input_norm(orbital_features[:, :2])
        residual = torch.tanh(self.gate(occ_energy)).squeeze(-1)
        mix = self.max_residual * torch.sigmoid(self.mix_logit)
        return 1.0 + mix * residual


class ZBLAtomPair(nn.Module):
    """Ziegler-Biersack-Littmark short-range pair repulsion on atom pairs.

    Nuclear-core repulsion that stabilizes the short-bond regime; adds a
    fixed physical contribution to the energy so the NN doesn't have to
    learn how hard nuclei repel each other at small r.

    Operates directly on atom numbers and positions (no node-attr one-hot
    detour). Cutoff envelope uses the sum of covalent radii per pair.

    Adapted from ``ZBLBasis`` in MACE's ``mace/modules/radial.py`` (MIT
    License, Copyright (c) 2022 ACEsuit/mace); see NOTICE.
    """

    def __init__(self, p: int = 6, trainable: bool = False):
        super().__init__()
        import ase.data
        self.register_buffer(
            "c",
            torch.tensor([0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(ase.data.covalent_radii, dtype=torch.get_default_dtype()),
        )
        if trainable:
            self.a_prefactor = nn.Parameter(torch.tensor(0.4543))
            self.a_exp = nn.Parameter(torch.tensor(0.300))
        else:
            self.register_buffer("a_prefactor", torch.tensor(0.4543))
            self.register_buffer("a_exp", torch.tensor(0.300))

    @staticmethod
    def _poly_envelope(r: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        p_f = p.float()
        x = r / r_max
        inside = (x < 1.0).float()
        env = (
            1.0
            - 0.5 * (p_f + 1.0) * (p_f + 2.0) * torch.pow(x, p_f)
            + p_f * (p_f + 2.0) * torch.pow(x, p_f + 1.0)
            - 0.5 * p_f * (p_f + 1.0) * torch.pow(x, p_f + 2.0)
        )
        return env * inside

    def forward(
        self,
        atom_positions: torch.Tensor,
        atom_numbers: torch.Tensor,
        batch_atom: torch.Tensor,
        num_graphs: int,
        cell: Optional[torch.Tensor] = None,
        has_pbc: bool = False,
    ) -> torch.Tensor:
        """Return per-graph ZBL pair energy summed over all intra-graph atom pairs."""
        n_atoms = atom_positions.shape[0]
        if n_atoms < 2:
            return torch.zeros(num_graphs, device=atom_positions.device, dtype=atom_positions.dtype)

        # Build intra-graph upper-triangular atom pairs. Non-periodic only;
        # for PBC we skip (long-range block handles that regime).
        same_graph = batch_atom.unsqueeze(0) == batch_atom.unsqueeze(1)
        upper = torch.triu(torch.ones(n_atoms, n_atoms, device=atom_positions.device, dtype=torch.bool), diagonal=1)
        mask = same_graph & upper
        i_idx, j_idx = torch.where(mask)
        if i_idx.numel() == 0:
            return torch.zeros(num_graphs, device=atom_positions.device, dtype=atom_positions.dtype)

        r_vec = atom_positions[j_idx] - atom_positions[i_idx]
        r = torch.linalg.norm(r_vec, dim=-1)

        Z_u = atom_numbers[i_idx].to(atom_positions.dtype)
        Z_v = atom_numbers[j_idx].to(atom_positions.dtype)

        a = (self.a_prefactor * 0.529) / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
        r_over_a = r / a.clamp(min=1e-6)
        phi = (
            self.c[0] * torch.exp(-3.2 * r_over_a)
            + self.c[1] * torch.exp(-0.9423 * r_over_a)
            + self.c[2] * torch.exp(-0.4028 * r_over_a)
            + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )
        r_safe = r.clamp(min=1e-3)
        v_pair = (14.3996 * Z_u * Z_v) / r_safe * phi
        # Clamp Z into the covalent-radii table (0..118) defensively; real
        # atomic numbers are always in range, this just avoids a hard crash on
        # malformed inputs.
        z_max_idx = self.covalent_radii.shape[0] - 1
        r_cut_pair = (
            self.covalent_radii[atom_numbers[i_idx].clamp(max=z_max_idx)]
            + self.covalent_radii[atom_numbers[j_idx].clamp(max=z_max_idx)]
        )
        envelope = self._poly_envelope(r_safe, r_cut_pair, self.p)
        v_pair = v_pair * envelope

        return scatter_sum(v_pair, batch_atom[i_idx], dim=0, dim_size=num_graphs)


class GradDetachedScaleShift(nn.Module):
    """Scale-shift block with gradient-detached scale factor.

    During backprop the scale is treated as a constant, preventing gradient
    explosion/vanishing when the target values span many orders of magnitude.
    """

    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor, heads: torch.Tensor = None) -> torch.Tensor:
        scale = self.scale.detach()
        return scale * x + self.shift


class OrbACE(nn.Module):
    """Orbital-space MACE with quantum-chemistry features.

    Key architectural choices:
      - 6-channel chemical edge features (incl. MO overlap) with residual fusion
      - Equivariant orbital node features (polarization 1o + quadrupole 2e)
      - Per-group LayerNorm on input features
      - Orbital shape invariants from quadrupole tensor
      - Identity-centered occupation modulation + orbital attention readout
      - Gradient-detached scale factor
      - Non-residual first interaction, residual rest
      - Multi-layer readout with per-interaction energy contribution
    """

    def __init__(
        self,
        r_max: float = 6.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 2,
        num_interactions: int = 2,
        num_species: int = 200,
        n_scalar_features: int = 10,
        n_ao_shell_features: int = 15,
        n_coeff_norm_features: int = 5,
        hidden_irreps: Union[str, o3.Irreps] = "64x0e",
        MLP_irreps: Union[str, o3.Irreps] = "16x0e",
        correlation: Union[int, List[int]] = 3,
        gate: Optional[Callable] = torch.nn.functional.silu,
        radial_MLP: Optional[List[int]] = None,
        avg_num_neighbors: float = 1.0,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
        orbital_e0_base: Optional[np.ndarray] = None,
        orbital_e0_slope: Optional[np.ndarray] = None,
        atom_e0_correction: Optional[np.ndarray] = None,
        use_chem_edge_features: bool = True,
        n_chem_edge_features: int = 6,
        use_equivariant_orbital_features: bool = True,
        n_equiv_channels: int = 8,
        dropout: float = 0.0,
        use_long_range: bool = False,
        use_multipole_lr: bool = True,
        lr_alpha: float = 0.2,
        lr_cutoff: float = 15.0,
        ewald_dl: float = 2.0,
        pair_repulsion: bool = True,
        pair_repulsion_p: int = 6,
        baseline_model: str = "none",
        baseline_basis: str = None,
        baseline_xc: str = None,
        atom_head_multi: bool = False,
        delta_method: str = "regular",
        base_atom_e0: Optional[np.ndarray] = None,
        mf_head_hidden: int = 64,
    ):
        super().__init__()
        self.baseline_model = baseline_model
        # Basis / functional used when the delta-learning baseline is recomputed
        # at inference (HF at a fixed basis, or KS-DFT). None => method defaults.
        self.baseline_basis = baseline_basis
        self.baseline_xc = baseline_xc
        self.atom_head_multi = atom_head_multi
        self.use_equivariant_orbital_features = use_equivariant_orbital_features
        # Multi-fidelity. In "mfdh" a learned per-atom BASE head predicts the
        # cheap baseline fidelity (xTB / HF) while the existing orbital readout +
        # atom-correction heads act as the high-fidelity DELTA, so TOTAL = E0 +
        # delta + base is trained against the ABSOLUTE reference and base is
        # soft-anchored to the baseline. In "mfah" the cheap head predicts the
        # cheap level's ABSOLUTE energy in parallel and never enters the total.
        # Neither mode recomputes a baseline at inference.
        self.delta_method = _canonical_fidelity(delta_method)

        if isinstance(hidden_irreps, str):
            hidden_irreps = o3.Irreps(hidden_irreps)
        if isinstance(MLP_irreps, str):
            MLP_irreps = o3.Irreps(MLP_irreps)

        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float32))
        self.use_long_range = use_long_range
        self.use_chem_edge_features = use_chem_edge_features
        self.num_species = num_species

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        # Scalar node features come from OrbitalFeatureEmbedding; when equivariant
        # orbital features are enabled, extend the node irreps with l=1/l=2
        # channels seeded from the orbital polarization/quadrupole.
        node_feats_scalar_irreps = o3.Irreps([(num_features, (0, 1))])
        if use_equivariant_orbital_features:
            self.n_equiv_channels = n_equiv_channels
            node_feats_irreps = o3.Irreps(
                [(num_features, (0, 1)), (n_equiv_channels, (1, -1)), (n_equiv_channels, (2, 1))]
            )
        else:
            self.n_equiv_channels = 0
            node_feats_irreps = node_feats_scalar_irreps

        a_t = torch.tensor(orbital_e0_base, dtype=torch.get_default_dtype()) if orbital_e0_base is not None else None
        b_t = torch.tensor(orbital_e0_slope, dtype=torch.get_default_dtype()) if orbital_e0_slope is not None else None
        self.orbital_energies_fn = OrbitalEnergiesBlock(num_species, initial_a=a_t, initial_b=b_t)

        if atom_e0_correction is not None:
            self.register_buffer(
                "atom_e0",
                torch.tensor(atom_e0_correction, dtype=torch.get_default_dtype()),
            )
        else:
            self.register_buffer("atom_e0", torch.zeros(1, dtype=torch.get_default_dtype()))

        self.orbital_embedding = OrbitalFeatureEmbedding(
            num_species=num_species,
            n_scalar_features=n_scalar_features,
            n_ao_shell_features=n_ao_shell_features,
            n_coeff_norm_features=n_coeff_norm_features,
            irreps_out=node_feats_scalar_irreps,
            dropout=dropout,
        )

        if use_equivariant_orbital_features:
            self.equiv_orbital_embedding = EquivariantOrbitalEmbedding(
                n_channels=n_equiv_channels,
            )

        # Per-orbital occupancy gate: modulates initial node features by a
        # bounded learned function of (orbital_energy, occupancy). Starts at
        # identity (output = 1.0) so pretrained checkpoints are preserved.
        self.occupation_gate = OccupationGate(n_input=2)

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        radial_out_dim = self.radial_embedding.out_dim
        edge_feats_irreps = o3.Irreps(f"{radial_out_dim}x0e")

        if use_chem_edge_features:
            self.chem_edge_embedding = ChemicalEdgeEmbedding(
                radial_dim=radial_out_dim, n_chem_features=n_chem_edge_features
            )

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        self.readouts = nn.ModuleList()

        for i in range(num_interactions):
            is_last = (i == num_interactions - 1)
            if i == 0:
                input_irreps = node_feats_irreps
            else:
                input_irreps = hidden_irreps

            hidden_irreps_out = o3.Irreps(str(hidden_irreps[0])) if is_last else hidden_irreps

            inter_cls = RealAgnosticInteractionBlock if i == 0 else RealAgnosticResidualInteractionBlock
            inter = inter_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=input_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)

            prod = EquivariantProductBasisBlock(
                node_feats_irreps=inter.target_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i],
                num_elements=num_species,
                use_sc=True,
                use_agnostic_product=True,
                use_reduced_cg=True,
            )
            self.products.append(prod)

            if is_last:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        irreps_in=hidden_irreps_out,
                        MLP_irreps=MLP_irreps,
                        gate=gate,
                    )
                )
            else:
                self.readouts.append(
                    LinearReadoutBlock(irreps_in=hidden_irreps_out)
                )

        self.scale_shift = GradDetachedScaleShift(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

        # Learned spread scale for dominant-atom anchored orbital positions:
        # position = atom[dominant] + spread_scale * (mw_sum - atom[dominant]).
        # Initialized to 1.0 so behavior matches the original MW-centroid
        # scheme at start, then the model can shrink the spread to avoid
        # degenerate positions if useful.
        self.orbital_spread_scale = nn.Parameter(torch.tensor(1.0))

        # Atom energy head consumes scalars concatenated from EVERY interaction
        # layer (MACE-style node_feats_out), not just the last layer. Each
        # layer contributes `num_scalar` scalar channels.
        num_scalar = hidden_irreps.count(o3.Irrep(0, 1))
        self.num_scalar_per_layer = num_scalar
        self.num_interactions = num_interactions
        head_in = num_scalar * num_interactions
        if atom_head_multi:
            # Shared trunk + per-element output: one linear weight and bias
            # per supported atomic number (Z). Gives the model element-aware
            # correction capacity without ~200 independent MLPs. The trunk is
            # shared so orbital-level feature learning doesn't bifurcate.
            self.atom_energy_trunk = nn.Sequential(
                nn.Linear(head_in, head_in),
                nn.SiLU(),
            )
            self.atom_head_out_weight = nn.Parameter(torch.zeros(num_species, head_in))
            self.atom_head_out_bias = nn.Parameter(torch.zeros(num_species))
            with torch.no_grad():
                nn.init.normal_(self.atom_head_out_weight, std=1.0 / max(1, head_in) ** 0.5)
        else:
            self.atom_energy_head = nn.Sequential(
                nn.Linear(head_in, head_in),
                nn.SiLU(),
                nn.Linear(head_in, 1),
            )

        # Multipole long-range needs the frozen orbital polarization direction,
        # which only exists when equivariant orbital features are enabled.
        self.use_multipole_lr = (
            use_multipole_lr and use_long_range and use_equivariant_orbital_features
        )
        if use_long_range:
            from .blocks import LongRangeBlock
            self.long_range = LongRangeBlock(
                hidden_dim=num_features,
                alpha=lr_alpha,
                lr_cutoff=lr_cutoff,
                ewald_dl=ewald_dl,
                use_multipoles=self.use_multipole_lr,
            )

        self.use_pair_repulsion = pair_repulsion
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLAtomPair(p=pair_repulsion_p)

        # ---- Multi-fidelity cheap-level head -------------------------------
        # Per-atom MLP on the same pooled scalar features the atom-correction
        # head uses. For mfdh the final layer is zero-initialized so a fresh MF
        # model starts identical to its direct/regular counterpart
        # (base_inter = 0) and then learns the cheap-baseline surrogate from the
        # base-energy anchor. For mfah the head is outside the total energy
        # entirely, so it needs no such guard and is left normally initialized.
        if self.delta_method in ("mfdh", "mfah"):
            head = nn.Sequential(
                nn.Linear(head_in, head_in),
                nn.SiLU(),
                nn.Linear(head_in, 1),
            )
            if self.delta_method == "mfdh":
                with torch.no_grad():
                    head[-1].weight.zero_()
                    head[-1].bias.zero_()
                self.mf_base_head = head
            else:
                self.mf_low_head = head
            if base_atom_e0 is not None:
                self.register_buffer(
                    "mf_base_atom_e0",
                    torch.tensor(base_atom_e0, dtype=torch.get_default_dtype()))
            else:
                self.register_buffer(
                    "mf_base_atom_e0",
                    torch.zeros(num_species, dtype=torch.get_default_dtype()))

    def fidelity_mode(self) -> str:
        """Canonical fidelity mode of this instance.

        Read through this rather than ``self.delta_method`` directly: a
        checkpoint is a pickled module, so a model trained before the
        mfdh/mfah split restores with the legacy ``"mf"`` in its ``__dict__``
        and never runs ``__init__`` to be normalized.
        """
        return model_fidelity_mode(self)

    def freeze_backbone(self, keep_last_interaction: bool = False):
        """Freeze all layers except readout heads for finetuning.

        Keeps trainable: readouts, scale_shift, atom energy head(s),
        and optionally the last interaction + product layer.
        """
        for param in self.parameters():
            param.requires_grad = False

        for param in self.scale_shift.parameters():
            param.requires_grad = True
        if self.atom_head_multi:
            for param in self.atom_energy_trunk.parameters():
                param.requires_grad = True
            self.atom_head_out_weight.requires_grad = True
            self.atom_head_out_bias.requires_grad = True
        else:
            for param in self.atom_energy_head.parameters():
                param.requires_grad = True
        for readout in self.readouts:
            for param in readout.parameters():
                param.requires_grad = True

        if keep_last_interaction:
            for param in self.interactions[-1].parameters():
                param.requires_grad = True
            for param in self.products[-1].parameters():
                param.requires_grad = True

        # Keep the MF cheap-level head trainable when finetuning a frozen backbone.
        for name in ("mf_base_head", "mf_low_head"):
            head = getattr(self, name, None)
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        return self._forward_impl(
            data, training, compute_force, compute_virials, compute_stress,
        )

    @torch.enable_grad()
    def _forward_impl(
        self,
        data: Dict[str, torch.Tensor],
        training: bool,
        compute_force: bool,
        compute_virials: bool,
        compute_stress: bool,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["num_graphs"]
        batch_orb = data["batch_orb"]
        has_pbc = data.get("has_pbc", False)

        atom_positions = data["atom_positions"]
        if compute_force or compute_virials or compute_stress:
            atom_positions = atom_positions.clone().requires_grad_(True)

        displacement = None
        cell = data.get("cell", None)
        if (compute_virials or compute_stress) and cell is not None and has_pbc:
            displacement = torch.zeros(
                (num_graphs, 3, 3),
                dtype=atom_positions.dtype,
                device=atom_positions.device,
                requires_grad=True,
            )

        mw_orb_idx = data["mw_orb_idx"]
        mw_atom_idx = data["mw_atom_idx"]
        mw_values = data["mw_values"]
        total_n_orb = data["total_n_orb"]
        n_atoms_total = atom_positions.shape[0]

        # Mulliken-weighted orbital centroid: mw_centroid[orb] = sum_a w[orb,a]*R_a.
        # Implemented as a scatter (index_add) rather than torch.sparse.mm: the
        # sparse CUDA kernel has no bfloat16 support (forward AND backward), which
        # broke bf16-autocast training; index_add has bf16 kernels and is the same
        # cost at these graph sizes. Positions are kept fp32 for force accuracy.
        with torch.autocast(device_type=atom_positions.device.type, enabled=False):
            pos32 = atom_positions.float()
            contrib = mw_values.float().unsqueeze(-1) * pos32[mw_atom_idx]
            mw_centroid = pos32.new_zeros((total_n_orb, 3))
            mw_centroid = mw_centroid.index_add(0, mw_orb_idx, contrib)
        mw_centroid = mw_centroid.to(atom_positions.dtype)

        # Dominant-atom anchored orbital positions: anchor the orbital at its
        # dominant atom so positions are degenerate-free across orbitals on
        # different atoms. Orbitals sharing a dominant atom still collapse
        # geometrically; those edges are handled by the min_edge_length mask.
        dominant_idx = data.get("dominant_atom_idx", None)
        if dominant_idx is not None:
            orbital_anchor = atom_positions[dominant_idx]
            mw_residual = mw_centroid - orbital_anchor
            orbital_positions = orbital_anchor + self.orbital_spread_scale * mw_residual
        else:
            orbital_positions = mw_centroid

        if displacement is not None:
            orbital_positions = orbital_positions + torch.einsum(
                "bi,gij->bj",
                orbital_positions,
                displacement[batch_orb],
            )

        edge_index = data["edge_index"]
        vectors = orbital_positions[edge_index[1]] - orbital_positions[edge_index[0]]

        unit_shifts = data.get("unit_shifts", None)
        if unit_shifts is not None and cell is not None and has_pbc:
            shift_vectors = torch.einsum(
                "ei,gij->ej", unit_shifts, cell[batch_orb[edge_index[0]]]
            )
            vectors = vectors + shift_vectors

        # After dominant-atom anchoring, orbitals on different atoms have
        # bond-scale separations. Only orbitals sharing the same dominant atom
        # can still collapse (e.g. 1s and 2s on the same nucleus). We must
        # drop those edges *before* computing lengths or spherical harmonics:
        # torch.linalg.norm and o3.SphericalHarmonics(normalize=True) both
        # have singular gradients at zero vectors (d(|v|)/dv = v/|v| = 0/0
        # and d(v/|v|)/dv blows up), and those NaNs propagate to the global
        # orbital_spread_scale parameter even through a post-hoc mask.
        min_edge_length = 1.0e-4
        sq_norm = (vectors * vectors).sum(dim=-1)
        edge_mask = sq_norm > (min_edge_length * min_edge_length)
        if not torch.all(edge_mask):
            edge_index = edge_index[:, edge_mask]
            vectors = vectors[edge_mask]
            if unit_shifts is not None:
                unit_shifts = unit_shifts[edge_mask]

        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)

        orbital_energy = data["orbital_features"][:, 0]
        node_e0 = self.orbital_energies_fn(data["orbital_species"], orbital_energy)
        e0 = scatter_sum(
            src=node_e0, index=batch_orb, dim=0, dim_size=num_graphs
        )

        if self.atom_e0.numel() > 1:
            # Clamp Z to the fitted table range; elements unseen in training
            # have no correction (and would otherwise index out of bounds).
            atom_z = data["atom_numbers"].clamp(max=self.atom_e0.shape[0] - 1)
            per_atom_e0 = self.atom_e0[atom_z]
            batch_atom_e0 = data["batch_atom"]
            atom_e0_sum = scatter_sum(
                per_atom_e0, batch_atom_e0, dim=0, dim_size=num_graphs
            )
            e0 = e0 + atom_e0_sum

        node_attrs, node_feats = self.orbital_embedding(
            data["orbital_species"],
            data["orbital_features"],
            data["ao_shell_encoding"],
            data["coeff_norms_per_l"],
        )

        if self.use_equivariant_orbital_features:
            n_orb_total = node_feats.shape[0]
            quad = data.get("orbital_quadrupole", None)
            pol = data.get("orbital_polarization", None)
            if quad is None:
                quad = torch.zeros(n_orb_total, 3, 3, device=node_feats.device, dtype=node_feats.dtype)
            if pol is None:
                pol = torch.zeros(n_orb_total, 3, device=node_feats.device, dtype=node_feats.dtype)
            equiv_feats = self.equiv_orbital_embedding(quad, pol)
            node_feats = torch.cat([node_feats, equiv_feats], dim=-1)

        gate = self.occupation_gate(data["orbital_features"])
        node_feats = node_feats * gate.unsqueeze(-1)

        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, _cutoff = self.radial_embedding(
            lengths,
            node_attrs,
            edge_index,
            torch.zeros(1, dtype=torch.int64, device=edge_index.device),
        )

        if self.use_chem_edge_features:
            cutoff_envelope = self.radial_embedding.cutoff_fn(lengths)

            def _get_edge_feat(key):
                v = data.get(key, None)
                if v is not None and not torch.all(edge_mask):
                    v = v[edge_mask]
                return v

            chem_embed = self.chem_edge_embedding(
                edge_fock=_get_edge_feat("edge_fock"),
                edge_coulomb=_get_edge_feat("edge_coulomb"),
                edge_exchange=_get_edge_feat("edge_exchange"),
                edge_hcore=_get_edge_feat("edge_hcore"),
                edge_energy_gap=_get_edge_feat("edge_energy_gap"),
                edge_overlap=_get_edge_feat("edge_overlap"),
                n_edges=edge_feats.shape[0],
                device=edge_feats.device,
                dtype=edge_feats.dtype,
            )
            edge_feats = edge_feats + chem_embed * cutoff_envelope

        node_es_list = []
        node_feats_list = []
        node_heads = torch.zeros(total_n_orb, dtype=torch.int64, device=node_feats.device)

        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=_cutoff,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs
            )
            node_feats_list.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(
                readout(node_feats_list[feat_idx], node_heads).squeeze(-1)
            )

        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es)

        node_energy = node_e0 + node_inter_es
        inter_e = scatter_sum(
            node_inter_es, batch_orb, dim=0, dim_size=num_graphs
        )

        # Concatenate scalar channels from every interaction layer so the
        # atom head sees features at every body order the model resolved.
        # Scalars occupy the leading channels of each layer's output irreps.
        layer_scalars = [nf[:, : self.num_scalar_per_layer] for nf in node_feats_list]
        all_scalars = torch.cat(layer_scalars, dim=-1)
        # Equivalent to torch.sparse.mm(mw_sparse.t(), all_scalars) but via
        # gather + weighted scatter_add, which is materially faster than
        # sparse COO matmul on GPU for dense per-orbital feature tensors.
        gathered = all_scalars[mw_orb_idx] * mw_values.unsqueeze(-1)
        atom_feats = torch.zeros(
            n_atoms_total,
            all_scalars.shape[-1],
            dtype=all_scalars.dtype,
            device=all_scalars.device,
        )
        atom_feats.index_add_(0, mw_atom_idx, gathered)
        if self.atom_head_multi:
            hidden = self.atom_energy_trunk(atom_feats)
            atom_z = data["atom_numbers"].clamp(max=self.atom_head_out_weight.shape[0] - 1)
            w = self.atom_head_out_weight[atom_z]
            b = self.atom_head_out_bias[atom_z]
            atom_correction = (hidden * w).sum(dim=-1) + b
        else:
            atom_correction = self.atom_energy_head(atom_feats).squeeze(-1)
        batch_atom = data["batch_atom"]
        correction_e = scatter_sum(
            atom_correction, batch_atom, dim=0, dim_size=num_graphs
        )

        # Multi-fidelity cheap-level head: per-atom learned surrogate of the
        # cheap baseline fidelity, anchored to the baseline via base_energy in
        # the MF loss. base_e0 is a per-element offset carried only in
        # base_energy so the head learns just the interaction part. Under mfdh
        # base_inter ALSO feeds the total energy (total = E0 + delta + base);
        # under mfah it does not, leaving the high-fidelity path untouched.
        mf_mode = self.fidelity_mode()
        mf_additive = mf_mode == "mfdh"
        base_inter = base_energy = None
        if mf_mode in ("mfdh", "mfah"):
            head = self.mf_base_head if mf_additive else self.mf_low_head
            base_atom_inter = head(atom_feats).squeeze(-1)
            base_inter = scatter_sum(
                base_atom_inter, batch_atom, dim=0, dim_size=num_graphs)
            atom_z_b = data["atom_numbers"].clamp(max=self.mf_base_atom_e0.shape[0] - 1)
            base_e0 = scatter_sum(
                self.mf_base_atom_e0[atom_z_b], batch_atom, dim=0, dim_size=num_graphs)
            base_energy = base_e0 + base_inter

        total_energy = e0 + inter_e + correction_e
        if mf_additive:
            total_energy = total_energy + base_inter

        lr_energy = None
        # getattr for the same reason as in atom_model: these checkpoints are
        # pickled modules, and one written before a flag existed restores
        # without it.
        if getattr(self, "use_long_range", False):
            final_feats = node_feats_list[-1]
            dipole_dir = (data.get("orbital_polarization", None)
                          if getattr(self, "use_multipole_lr", False) else None)
            lr_energy = self.long_range(
                final_feats, orbital_positions, batch_orb, num_graphs,
                cell=cell, has_pbc=has_pbc, dipole_dir=dipole_dir,
            )
            total_energy = total_energy + lr_energy

        zbl_energy = None
        if self.use_pair_repulsion:
            zbl_energy = self.pair_repulsion_fn(
                atom_positions=atom_positions,
                atom_numbers=data["atom_numbers"],
                batch_atom=data["batch_atom"],
                num_graphs=num_graphs,
                cell=cell,
                has_pbc=has_pbc,
            )
            total_energy = total_energy + zbl_energy

        # Forces = -dE/dR: include position-dependent contributions
        # (interaction, correction, long-range, ZBL pair repulsion).
        # E0 is excluded because it only depends on species (not positions).
        forces_energy = inter_e + correction_e
        if mf_additive:
            forces_energy = forces_energy + base_inter
        if lr_energy is not None:
            forces_energy = forces_energy + lr_energy
        if zbl_energy is not None:
            forces_energy = forces_energy + zbl_energy

        if mf_mode in ("mfdh", "mfah"):
            # forces_energy is the high-fidelity interaction energy (including
            # the base head under mfdh, excluding it under mfah); its gradient
            # is the total force (trained vs absolute reference). base_inter is
            # differentiated separately (anchor to the baseline). Both share
            # the graph, so we grad manually with retain_graph. Stress/virials
            # are not supported on the MF path (molecular use case).
            forces = base_forces = None
            if compute_force:
                t_grad = torch.autograd.grad(
                    outputs=[forces_energy.sum()], inputs=[atom_positions],
                    create_graph=training, retain_graph=True, allow_unused=True)[0]
                forces = torch.zeros_like(atom_positions) if t_grad is None else -t_grad
                b_grad = torch.autograd.grad(
                    outputs=[base_inter.sum()], inputs=[atom_positions],
                    create_graph=training, retain_graph=training, allow_unused=True)[0]
                base_forces = torch.zeros_like(atom_positions) if b_grad is None else -b_grad
            return {
                "energy": total_energy,
                "forces": forces,
                "virials": None,
                "stress": None,
                "node_energy": node_energy,
                "interaction_energy": inter_e,
                "e0_energy": e0,
                "lr_energy": lr_energy,
                "zbl_energy": zbl_energy,
                "base_energy": base_energy,
                "base_forces": base_forces,
            }

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=forces_energy,
            positions=atom_positions,
            displacement=displacement,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "e0_energy": e0,
            "lr_energy": lr_energy,
            "zbl_energy": zbl_energy,
        }
