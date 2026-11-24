# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
# Portions adapted from MACE remain under the MIT License; see NOTICE.
"""AtomOrbACE: atom-centered MACE on gauge-invariant electronic-structure features.

Design (vs. the orbital-space OrbACE v1)
----------------------------------------
* Nodes are ATOMS (positions = nuclear coordinates), so forces flow through the
  actual atomic positions via autograd -- no orbital-centroid indirection.
* Node/edge features are SMOOTH, localization-gauge-invariant descriptors built
  from a cheap extended-Hückel (TB) Hamiltonian, computed from geometry
  in-forward (see data/tb_features.py). Nothing is detached from R and no
  self-consistency loop intervenes, so reverse-mode autograd returns the exact
  gradient and the model is strictly CONSERVATIVE, with no SCF. This is unlike
  the orbital-space model, whose energy jumps discontinuously as orbitals
  re-localize.
* Equivariance comes purely from geometry (spherical harmonics on atom-atom
  edges), exactly like vanilla MACE. Features are rotation-invariant scalars.

Reuses the MACE interaction/product/readout blocks and OrbACE's ZBL pair term.
The backbone follows the layout of MACE's own model class
(https://github.com/ACEsuit/mace, MIT License, Copyright (c) 2022
ACEsuit/mace); see NOTICE.
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

if hasattr(torch, "serialization"):
    torch.serialization.add_safe_globals([slice])

from e3nn import o3
from mace.modules.blocks import (
    EquivariantProductBasisBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.utils import get_outputs
from mace.tools.scatter import scatter_sum

from .models import (
    ZBLAtomPair,
    GradDetachedScaleShift,
    _canonical_fidelity,
    model_fidelity_mode,
)


class FeatureStandardizer(nn.Module):
    """Dataset-level per-feature standardization: (x - mean) / std.

    Unlike per-sample LayerNorm, this preserves the ABSOLUTE magnitude of each
    physical feature across atoms/edges -- essential for the energy-partition
    features (Tr(P_AA F_AA), Tr(W_AA), ...) that encode the total electronic
    energy. mean/std are fixed buffers computed once from the training set.
    """

    def __init__(self, n_features: int, mean=None, std=None):
        super().__init__()
        m = torch.zeros(n_features) if mean is None else torch.as_tensor(mean, dtype=torch.get_default_dtype())
        s = torch.ones(n_features) if std is None else torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("mean", m)
        self.register_buffer("std", s.clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)


class AtomFeatureEmbedding(nn.Module):
    """Embed atomic number + gauge-invariant node scalars into node features."""

    def __init__(self, num_species: int, n_node_scalars: int, irreps_out: o3.Irreps,
                 feature_mean=None, feature_std=None):
        super().__init__()
        self.num_species = num_species
        self.species_embedding = o3.Linear(
            irreps_in=o3.Irreps(f"{num_species}x0e"), irreps_out=irreps_out
        )
        num_features = irreps_out.count(o3.Irrep(0, 1))
        self.standardize = FeatureStandardizer(n_node_scalars, feature_mean, feature_std)
        self.feature_proj = nn.Sequential(
            nn.Linear(n_node_scalars, num_features * 2),
            nn.SiLU(),
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
        )

    def forward(self, species_idx: torch.Tensor, node_scalars: torch.Tensor):
        one_hot = torch.nn.functional.one_hot(
            species_idx, self.num_species
        ).to(node_scalars.dtype)
        node_feats = self.species_embedding(one_hot)
        node_feats = node_feats + self.feature_proj(self.standardize(node_scalars))
        return one_hot, node_feats


class MFScalarHead(nn.Module):
    """MLP readout for the multi-fidelity (mfdh / mfah) heads.

    Reads the concatenation of the final scalar node features and the RAW
    (standardized) gauge-invariant QC node scalars, so the quantum-chemistry
    descriptors feed the head directly (skip connection) in addition to
    flowing through the MACE backbone. Emits one scalar (per-atom energy)
    contribution.

    ``zero_init`` starts the head at exactly zero so a fresh MF model
    reproduces its non-MF counterpart before any training signal arrives. Only
    heads that feed the total energy need it, i.e. mfdh's additive QC-delta
    refinement; mfah's low head sits outside the total and is left as is.
    """

    def __init__(self, feat_dim: int, n_qc_scalars: int, hidden: int = 64,
                 n_layers: int = 2, zero_init: bool = False):
        super().__init__()
        in_dim = feat_dim + n_qc_scalars
        layers: list = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.SiLU())
            prev = hidden
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)
        if zero_init:
            with torch.no_grad():
                self.mlp[-1].weight.zero_()
                self.mlp[-1].bias.zero_()

    def forward(self, scalar_feats: torch.Tensor, qc_scalars_std: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([scalar_feats, qc_scalars_std], dim=-1)).squeeze(-1)


class InvariantEdgeEmbedding(nn.Module):
    """Project gauge-invariant atom-pair features into the radial feature space.

    Zero-initialized residual so training starts from a pure-geometry MACE edge
    embedding and learns the chemistry correction from data.
    """

    def __init__(self, radial_dim: int, n_edge_features: int,
                 feature_mean=None, feature_std=None):
        super().__init__()
        self.standardize = FeatureStandardizer(n_edge_features, feature_mean, feature_std)
        self.proj = nn.Sequential(
            nn.Linear(n_edge_features, radial_dim),
            nn.SiLU(),
            nn.Linear(radial_dim, radial_dim),
        )
        with torch.no_grad():
            self.proj[-1].weight.zero_()
            self.proj[-1].bias.zero_()

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        return self.proj(self.standardize(edge_features))


class AtomOrbACE(nn.Module):
    def __init__(
        self,
        r_max: float = 5.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 2,
        num_interactions: int = 3,
        num_species: int = 100,
        n_node_scalars: int = 12,
        n_edge_features: int = 6,
        hidden_irreps: Union[str, o3.Irreps] = "64x0e + 64x1o + 64x2e",
        MLP_irreps: Union[str, o3.Irreps] = "16x0e",
        correlation: Union[int, List[int]] = 3,
        gate: Optional[Callable] = torch.nn.functional.silu,
        radial_MLP: Optional[List[int]] = None,
        avg_num_neighbors: float = 10.0,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
        atom_e0: Optional[np.ndarray] = None,
        node_feature_mean: Optional[np.ndarray] = None,
        node_feature_std: Optional[np.ndarray] = None,
        edge_feature_mean: Optional[np.ndarray] = None,
        edge_feature_std: Optional[np.ndarray] = None,
        use_edge_features: bool = True,
        pair_repulsion: bool = True,
        pair_repulsion_p: int = 6,
        use_long_range: bool = False,
        lr_alpha: float = 0.2,
        lr_cutoff: float = 15.0,
        baseline_model: str = "none",
        baseline_basis: str = None,
        baseline_xc: str = None,
        delta_method: str = "regular",
        base_atom_e0: Optional[np.ndarray] = None,
        mf_head_hidden: int = 64,
    ):
        super().__init__()
        self.baseline_model = baseline_model
        self.baseline_basis = baseline_basis
        self.baseline_xc = baseline_xc
        self.num_species = num_species
        self.use_edge_features = use_edge_features
        self.delta_method = _canonical_fidelity(delta_method)
        self.n_node_scalars = n_node_scalars

        if isinstance(hidden_irreps, str):
            hidden_irreps = o3.Irreps(hidden_irreps)
        if isinstance(MLP_irreps, str):
            MLP_irreps = o3.Irreps(MLP_irreps)
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float32))

        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        self.num_features = num_features
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_scalar_irreps = o3.Irreps([(num_features, (0, 1))])

        # Per-element E0 (isolated-atom energy correction), like MACE.
        if atom_e0 is not None:
            self.register_buffer("atom_e0", torch.tensor(atom_e0, dtype=torch.get_default_dtype()))
        else:
            self.register_buffer("atom_e0", torch.zeros(num_species, dtype=torch.get_default_dtype()))

        self.atom_embedding = AtomFeatureEmbedding(
            num_species=num_species,
            n_node_scalars=n_node_scalars,
            irreps_out=node_feats_scalar_irreps,
            feature_mean=node_feature_mean,
            feature_std=node_feature_std,
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max, num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        radial_out_dim = self.radial_embedding.out_dim
        edge_feats_irreps = o3.Irreps(f"{radial_out_dim}x0e")

        if use_edge_features:
            self.edge_embedding = InvariantEdgeEmbedding(
                radial_out_dim, n_edge_features,
                feature_mean=edge_feature_mean, feature_std=edge_feature_std)

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
            input_irreps = node_feats_scalar_irreps if i == 0 else hidden_irreps
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
            self.products.append(EquivariantProductBasisBlock(
                node_feats_irreps=inter.target_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i],
                num_elements=num_species,
                use_sc=True,
                use_agnostic_product=True,
                use_reduced_cg=True,
            ))
            if is_last:
                self.readouts.append(NonLinearReadoutBlock(
                    irreps_in=hidden_irreps_out, MLP_irreps=MLP_irreps, gate=gate))
            else:
                self.readouts.append(LinearReadoutBlock(irreps_in=hidden_irreps_out))

        self.scale_shift = GradDetachedScaleShift(
            scale=atomic_inter_scale, shift=atomic_inter_shift)

        self.use_pair_repulsion = pair_repulsion
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLAtomPair(p=pair_repulsion_p)

        # Ewald-damped electrostatics between ATOMIC sites, beyond the r_max
        # message-passing cutoff. The block is site-agnostic, so this is the same
        # one the orbital model uses, with nuclei in place of orbital centroids.
        #
        # Monopole only: use_multipoles needs a per-site direction to carry the
        # dipole, which the orbital model takes from the frozen HF orbital
        # polarization. An atomic site has no such intrinsic vector, so there is
        # nothing to pass and the charge-dipole and dipole-dipole terms are off.
        #
        # Off by default, matching the orbital model. The charge readout is
        # zero-initialized, so switching it on adds a term that starts at exactly
        # zero and grows from data, and a checkpoint trained without it loads
        # unchanged.
        self.use_long_range = use_long_range
        if use_long_range:
            from .blocks import LongRangeBlock
            self.long_range = LongRangeBlock(
                hidden_dim=num_features,
                alpha=lr_alpha,
                lr_cutoff=lr_cutoff,
                use_multipoles=False,
            )

        # ---- Multi-fidelity heads ------------------------------------------
        # Every head reads the final scalar node features CONCATENATED with the
        # raw (standardized) gauge-invariant Hückel node scalars.
        #
        # mfdh (additive):
        #   base_head     : learned surrogate of the cheap fidelity (xTB / HF),
        #                   replacing the per-frame physics baseline at
        #                   inference. It is ADDED into the total energy.
        #   qc_delta_head : additive refinement of the high-fidelity readout.
        #                   The "qc" in these names is kept because it appears
        #                   in every MF checkpoint's state_dict.
        # mfah (parallel):
        #   low_head      : predicts the cheap level's ABSOLUTE energy. It never
        #                   enters the total, so the high-fidelity path is left
        #                   bit-identical to a direct model and the cheap level
        #                   is purely an auxiliary task, discardable after
        #                   training.
        if self.delta_method == "mfdh":
            self.mf_base_head = MFScalarHead(
                num_features, n_node_scalars, hidden=mf_head_hidden, zero_init=False)
            self.mf_qc_delta_head = MFScalarHead(
                num_features, n_node_scalars, hidden=mf_head_hidden, zero_init=True)
        elif self.delta_method == "mfah":
            self.mf_low_head = MFScalarHead(
                num_features, n_node_scalars, hidden=mf_head_hidden, zero_init=False)
        if self.delta_method in ("mfdh", "mfah"):
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

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.scale_shift.parameters():
            p.requires_grad = True
        for r in self.readouts:
            for p in r.parameters():
                p.requires_grad = True
        # Keep the MF heads trainable when finetuning a frozen backbone.
        for name in ("mf_base_head", "mf_qc_delta_head", "mf_low_head"):
            head = getattr(self, name, None)
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = True

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        return self._forward_impl(data, training, compute_force, compute_virials, compute_stress)

    @torch.enable_grad()
    def _forward_impl(self, data, training, compute_force, compute_virials, compute_stress):
        num_graphs = data["num_graphs"]
        has_pbc = data.get("has_pbc", False)
        cell = data.get("cell", None)

        atom_positions = data["atom_positions"]
        if compute_force or compute_virials or compute_stress:
            atom_positions = atom_positions.clone().requires_grad_(True)

        displacement = None
        if (compute_virials or compute_stress) and cell is not None and has_pbc:
            displacement = torch.zeros(
                (num_graphs, 3, 3), dtype=atom_positions.dtype,
                device=atom_positions.device, requires_grad=True)

        batch_atom = data["batch_atom"]
        edge_index = data["edge_index"]
        species_idx = data["atom_numbers"].clamp(min=0, max=self.num_species - 1)

        pos = atom_positions
        if displacement is not None:
            pos = pos + torch.einsum("bi,bij->bj", pos, displacement[batch_atom])
        vectors = pos[edge_index[1]] - pos[edge_index[0]]

        unit_shifts = data.get("unit_shifts", None)
        if unit_shifts is not None and cell is not None and has_pbc:
            shift_vectors = torch.einsum(
                "ei,gij->ej", unit_shifts, cell[batch_atom[edge_index[0]]])
            vectors = vectors + shift_vectors

        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)

        # Differentiable extended-Hückel features built from atom_positions
        # in-forward. Autograd through these yields exact conservative forces
        # with no SCF.
        from ..data.tb_features import TB_PACK_KEYS, tb_features
        # Present only when the collate built it in a worker; deriving it
        # here instead would cost a device sync on every batch.
        pack = ({k: data[k] for k in TB_PACK_KEYS}
                if all(k in data for k in TB_PACK_KEYS) else None)
        # Ignored when a pack is supplied, since its occupations already
        # account for charge; this is what keeps the derive-in-forward
        # callers (eval, ASE) from silently treating an ion as neutral.
        node_scalars_in, edge_features_in = tb_features(
            atom_positions, data["atom_numbers"], edge_index, batch_atom,
            num_graphs, pack=pack, charges=data.get("charge"))

        node_attrs, node_feats = self.atom_embedding(species_idx, node_scalars_in)

        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, _cutoff = self.radial_embedding(
            lengths, node_attrs, edge_index,
            torch.zeros(1, dtype=torch.int64, device=edge_index.device))

        if self.use_edge_features and edge_features_in is not None:
            cutoff_env = self.radial_embedding.cutoff_fn(lengths)
            edge_feats = edge_feats + self.edge_embedding(edge_features_in) * cutoff_env

        node_heads = torch.zeros(node_feats.shape[0], dtype=torch.int64, device=node_feats.device)
        node_feats_list = []
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(
                node_attrs=node_attrs, node_feats=node_feats,
                edge_attrs=edge_attrs, edge_feats=edge_feats,
                edge_index=edge_index, cutoff=_cutoff)
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            node_feats_list.append(node_feats)

        node_es_list = []
        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(readout(node_feats_list[feat_idx], node_heads).squeeze(-1))

        node_inter_es = torch.sum(torch.stack(node_es_list, 0), 0)
        node_inter_es = self.scale_shift(node_inter_es)

        # Multi-fidelity path. Both modes predict the cheap fidelity with a head
        # reading the final scalar node features concatenated with the raw
        # (standardized) Hückel node scalars, and in both the high-fidelity
        # output is trained against the ABSOLUTE reference E/F -- never against
        # the (badly inflated) residual forces. They differ in whether that
        # cheap head is part of the high-fidelity sum:
        #   mfdh -- ADDITIVE. total = (readout + delta) + base, so the base head
        #           is load-bearing and must be evaluated at inference.
        #   mfah -- PARALLEL. the low head is a separate absolute prediction and
        #           node_inter_es is untouched, so the high-fidelity path stays
        #           identical to a direct model's and the low head influences it
        #           only through the shared backbone (an implicit delta).
        mf_mode = self.fidelity_mode()
        base_inter = base_energy = base_node_inter = None
        if mf_mode in ("mfdh", "mfah"):
            scalar_feats = node_feats_list[-1][:, :self.num_features]
            qc_std = self.atom_embedding.standardize(node_scalars_in)
            if mf_mode == "mfdh":
                delta_node = node_inter_es + self.mf_qc_delta_head(scalar_feats, qc_std)
                base_node_inter = self.mf_base_head(scalar_feats, qc_std)
                node_inter_es = delta_node + base_node_inter    # TOTAL interaction
            else:
                base_node_inter = self.mf_low_head(scalar_feats, qc_std)
            base_inter = scatter_sum(base_node_inter, batch_atom, dim=0, dim_size=num_graphs)
            base_e0 = scatter_sum(self.mf_base_atom_e0[species_idx], batch_atom,
                                  dim=0, dim_size=num_graphs)
            base_energy = base_e0 + base_inter

        inter_e = scatter_sum(node_inter_es, batch_atom, dim=0, dim_size=num_graphs)

        node_e0 = self.atom_e0[species_idx]
        e0 = scatter_sum(node_e0, batch_atom, dim=0, dim_size=num_graphs)

        total_energy = e0 + inter_e

        zbl_energy = None
        if self.use_pair_repulsion:
            zbl_energy = self.pair_repulsion_fn(
                atom_positions=atom_positions, atom_numbers=data["atom_numbers"],
                batch_atom=batch_atom, num_graphs=num_graphs, cell=cell, has_pbc=has_pbc)
            total_energy = total_energy + zbl_energy

        lr_energy = None
        # getattr, not attribute access: best_model.pt is a pickled module, so a
        # checkpoint written before long range was added restores an instance
        # whose __dict__ has no such flag, and evaluating it against the current
        # class then dies on AttributeError. That silently cost the
        # uracil/mf_hf arm of the rMD17 atom campaign.
        if getattr(self, "use_long_range", False):
            # node_feats_list[-1] holds the full irreps; the charge readout wants
            # the invariant part, which is the leading num_features columns.
            lr_energy = self.long_range(
                node_feats_list[-1][:, :self.num_features],
                atom_positions, batch_atom, num_graphs,
                cell=cell, has_pbc=has_pbc, dipole_dir=None)
            total_energy = total_energy + lr_energy

        forces_energy = inter_e
        if zbl_energy is not None:
            forces_energy = forces_energy + zbl_energy
        # Electrostatics belong to the model's own prediction, so they enter the
        # force energy but never base_inter, which must stay a clean surrogate of
        # the physics baseline for the multi-fidelity anchor below.
        if lr_energy is not None:
            forces_energy = forces_energy + lr_energy

        if mf_mode in ("mfdh", "mfah"):
            # forces_energy is the high-fidelity interaction energy (for mfdh
            # that includes the base head; for mfah it does not), so its
            # gradient is the total force -- trained against the absolute
            # reference forces. The cheap-level force is differentiated
            # separately (anchor to the baseline). Both share the node-feature
            # graph, so we grad manually with retain_graph=True (get_outputs
            # frees the graph after the first grad in eval mode). Stress/virials
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
                "energy": total_energy,        # absolute high-fidelity prediction
                "forces": forces,              # total force
                "virials": None,
                "stress": None,
                "interaction_energy": inter_e,
                "e0_energy": e0,
                "zbl_energy": zbl_energy,
                "lr_energy": lr_energy,
                "base_energy": base_energy,    # anchor target: baseline energy
                "base_forces": base_forces,    # anchor target: baseline forces
            }

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=forces_energy, positions=atom_positions, displacement=displacement,
            cell=cell, training=training, compute_force=compute_force,
            compute_virials=compute_virials, compute_stress=compute_stress)

        return {
            "energy": total_energy,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "interaction_energy": inter_e,
            "e0_energy": e0,
            "zbl_energy": zbl_energy,
            "lr_energy": lr_energy,
        }
