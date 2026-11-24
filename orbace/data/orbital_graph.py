# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""OrbitalData container and graph construction for OrbACE.

Supports both isolated molecules and periodic systems (PBC).
Uses vectorized distance computation for efficient neighbor-list building.
"""

from dataclasses import dataclass, fields
from typing import Dict, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


@dataclass
class OrbitalData:
    """Container for a single orbital graph.

    Orbital-level (n_orb nodes):
        orbital_positions: (n_orb, 3) centroids in Angstrom
        orbital_features:  (n_orb, n_feat) scalar features
        orbital_species:   (n_orb,) integer species index
        ao_shell_encoding: (n_orb, 15) binary AO shell presence
        coeff_norms_per_l: (n_orb, n_l) MO coefficient norms per ang. mom.
        orbital_quadrupole: (n_orb, 3, 3) traceless l=2 shape tensor (Bohr^2)
        orbital_polarization: (n_orb, 3) l=1 centroid-minus-nucleus vector (Bohr)

    Atom-level:
        atom_positions: (n_atoms, 3) nuclear positions in Angstrom
        atom_numbers:   (n_atoms,) atomic numbers

    Orbital-to-atom mapping:
        mulliken_weights: (n_orb, n_atoms) Mulliken population weights

    Graph edges:
        edge_index: (2, n_edges) source/target orbital indices
        edge_vectors: (n_edges, 3) displacement vectors
        unit_shifts: (n_edges, 3) periodic image shifts (zeros for isolated)
        edge_fock: (n_edges,) Fock matrix element F_ij
        edge_coulomb: (n_edges,) Coulomb integral (ii|jj)
        edge_exchange: (n_edges,) Exchange integral (ij|ji)
        edge_hcore: (n_edges,) Core Hamiltonian h_ij
        edge_energy_gap: (n_edges,) |eps_i - eps_j|
        edge_overlap: (n_edges,) MO overlap S_ij (localized non-orthogonality)

    Periodic:
        cell: (3, 3) lattice vectors or None
        pbc: (3,) periodic boundary flags

    Labels:
        energy: scalar total energy (eV)
        forces: (n_atoms, 3) atomic forces (eV/Ang)
        stress: (3, 3) stress tensor
    """

    orbital_positions: torch.Tensor
    orbital_features: torch.Tensor
    orbital_species: torch.Tensor
    ao_shell_encoding: torch.Tensor
    coeff_norms_per_l: torch.Tensor
    atom_positions: torch.Tensor
    atom_numbers: torch.Tensor
    mulliken_weights: torch.Tensor
    dominant_atom_idx: torch.Tensor
    edge_index: torch.Tensor
    edge_vectors: torch.Tensor
    orbital_quadrupole: Optional[torch.Tensor] = None
    orbital_polarization: Optional[torch.Tensor] = None
    edge_fock: Optional[torch.Tensor] = None
    edge_coulomb: Optional[torch.Tensor] = None
    edge_exchange: Optional[torch.Tensor] = None
    edge_hcore: Optional[torch.Tensor] = None
    edge_energy_gap: Optional[torch.Tensor] = None
    edge_overlap: Optional[torch.Tensor] = None
    unit_shifts: Optional[torch.Tensor] = None
    cell: Optional[torch.Tensor] = None
    pbc: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None
    forces: Optional[torch.Tensor] = None
    stress: Optional[torch.Tensor] = None
    baseline_energy: Optional[torch.Tensor] = None
    baseline_forces: Optional[torch.Tensor] = None
    n_orbitals: Optional[torch.Tensor] = None
    n_atoms: Optional[torch.Tensor] = None
    batch_orb: Optional[torch.Tensor] = None
    batch_atom: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "OrbitalData":
        kwargs = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                kwargs[f.name] = val.to(device)
            else:
                kwargs[f.name] = val
        return OrbitalData(**kwargs)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) is not None}


def _build_neighbor_list_isolated(centroids: NDArray, r_cut: float, self_interaction: bool):
    """Vectorized neighbor list for isolated (non-periodic) systems."""
    dists = cdist(centroids, centroids)
    mask = dists < r_cut
    if not self_interaction:
        np.fill_diagonal(mask, False)
    senders, receivers = np.where(mask)
    vectors = centroids[receivers] - centroids[senders]
    shifts = np.zeros((len(senders), 3), dtype=np.float32)
    return senders, receivers, vectors, shifts


def _build_neighbor_list_pbc(centroids: NDArray, cell: NDArray, pbc: NDArray,
                              r_cut: float, self_interaction: bool):
    """Neighbor list for periodic systems via image enumeration."""
    n_orb = len(centroids)
    lengths = np.linalg.norm(cell, axis=1)
    n_images = np.ceil(r_cut / lengths).astype(int)
    n_images = np.where(pbc, n_images, 0)

    senders, receivers, vectors, shifts_list = [], [], [], []
    for nx in range(-n_images[0], n_images[0] + 1):
        for ny in range(-n_images[1], n_images[1] + 1):
            for nz in range(-n_images[2], n_images[2] + 1):
                shift = np.array([nx, ny, nz], dtype=np.float32)
                offset = shift @ cell
                shifted = centroids + offset
                dists = cdist(centroids, shifted)
                mask = dists < r_cut
                if nx == 0 and ny == 0 and nz == 0 and not self_interaction:
                    np.fill_diagonal(mask, False)
                s, r = np.where(mask)
                if len(s) > 0:
                    senders.append(s)
                    receivers.append(r)
                    vectors.append(shifted[r] - centroids[s])
                    shifts_list.append(np.tile(shift, (len(s), 1)))

    if senders:
        return (np.concatenate(senders), np.concatenate(receivers),
                np.concatenate(vectors).astype(np.float32),
                np.concatenate(shifts_list).astype(np.float32))
    return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))


def _prune_orbital_window(
    orbital_data: Dict[str, NDArray], window: int
) -> Dict[str, NDArray]:
    """Keep only orbitals in [HOMO-window, LUMO+window] around the Fermi level.

    Core orbitals on heavy atoms (e.g. 1s of C/N/O) are energetically decoupled
    from bonding chemistry and contribute little to the learned correction;
    their contribution is already captured by the atomic-E0 baseline (or the
    delta-learning baseline). Pruning them cuts graph size 3-5x on organic
    molecules with essentially no accuracy cost.
    """
    if window is None or window <= 0:
        return orbital_data
    occ = np.asarray(orbital_data["orbital_features"])[:, 1]
    n = occ.shape[0]
    # HOMO: highest index with occ > 0.5; LUMO: lowest index with occ <= 0.5.
    occupied = np.where(occ > 0.5)[0]
    virtual = np.where(occ <= 0.5)[0]
    if occupied.size == 0 or virtual.size == 0:
        return orbital_data
    homo = int(occupied.max())
    lumo = int(virtual.min())
    lo = max(0, homo - window + 1)
    hi = min(n, lumo + window)
    keep = np.arange(lo, hi, dtype=np.int64)
    if keep.size == n:
        return orbital_data

    orb_keys_1d = {
        "orbital_centroids", "orbital_features", "orbital_species",
        "ao_shell_encoding", "coeff_norms_per_l", "mo_energy",
    }
    orb_keys_rows = {
        "mulliken_weights", "orbital_quadrupole", "orbital_polarization",
    }
    orb_keys_2d = {
        "fock_matrix_mo", "coulomb_mo", "exchange_mo", "hcore_mo", "overlap_mo",
    }

    pruned = dict(orbital_data)
    for k in orb_keys_1d:
        if k in pruned and pruned[k] is not None:
            pruned[k] = np.asarray(pruned[k])[keep]
    for k in orb_keys_rows:
        if k in pruned and pruned[k] is not None:
            pruned[k] = np.asarray(pruned[k])[keep]
    for k in orb_keys_2d:
        if k in pruned and pruned[k] is not None:
            arr = np.asarray(pruned[k])
            pruned[k] = arr[np.ix_(keep, keep)]
    return pruned


def build_orbital_graph(
    orbital_data: Dict[str, NDArray],
    r_cut: float = 6.0,
    energy: float = None,
    forces: NDArray = None,
    stress: NDArray = None,
    cell: NDArray = None,
    pbc: NDArray = None,
    self_interaction: bool = False,
    baseline_energy: Optional[float] = None,
    baseline_forces: NDArray = None,
    orbital_window: Optional[int] = None,
) -> OrbitalData:
    """Build an orbital graph from extracted orbital data.

    Args:
        orbital_data: dict from extract_all_orbital_data()
        r_cut: cutoff radius in Angstrom for orbital-orbital edges
        energy: reference total energy (eV, after E0 subtraction)
        forces: reference atomic forces (eV/Ang), shape (n_atoms, 3)
        stress: reference stress tensor, shape (3, 3)
        cell: lattice vectors (3, 3) in Angstrom, or None for isolated
        pbc: periodic boundary conditions (3,) bool, or None
        self_interaction: whether to include self-loops
    """
    if orbital_window is not None and orbital_window > 0:
        orbital_data = _prune_orbital_window(orbital_data, orbital_window)
    centroids = orbital_data["orbital_centroids"]
    n_orb = centroids.shape[0]
    n_atoms = orbital_data["atom_positions"].shape[0]

    has_pbc = cell is not None and pbc is not None and np.any(pbc)
    if has_pbc:
        senders, receivers, edge_vecs, shifts = _build_neighbor_list_pbc(
            centroids, cell, pbc, r_cut, self_interaction)
    else:
        senders, receivers, edge_vecs, shifts = _build_neighbor_list_isolated(
            centroids, r_cut, self_interaction)

    edge_index = np.stack([senders, receivers], axis=0).astype(np.int64) if len(senders) > 0 \
        else np.zeros((2, 0), dtype=np.int64)

    fock_mo = orbital_data.get("fock_matrix_mo", None)
    coulomb_mo = orbital_data.get("coulomb_mo", None)
    exchange_mo = orbital_data.get("exchange_mo", None)
    hcore_mo = orbital_data.get("hcore_mo", None)
    overlap_mo = orbital_data.get("overlap_mo", None)
    mo_energy = orbital_data.get("mo_energy", None)

    edge_fock_t = None
    edge_coulomb_t = None
    edge_exchange_t = None
    edge_hcore_t = None
    edge_energy_gap_t = None
    edge_overlap_t = None
    if len(senders) > 0:
        if fock_mo is not None:
            edge_fock_t = torch.tensor(fock_mo[senders, receivers], dtype=torch.float32)
        if coulomb_mo is not None:
            edge_coulomb_t = torch.tensor(coulomb_mo[senders, receivers], dtype=torch.float32)
        if exchange_mo is not None:
            edge_exchange_t = torch.tensor(exchange_mo[senders, receivers], dtype=torch.float32)
        if hcore_mo is not None:
            edge_hcore_t = torch.tensor(hcore_mo[senders, receivers], dtype=torch.float32)
        if overlap_mo is not None:
            edge_overlap_t = torch.tensor(overlap_mo[senders, receivers], dtype=torch.float32)
        if mo_energy is not None:
            gap = np.abs(mo_energy[senders] - mo_energy[receivers]).astype(np.float32)
            edge_energy_gap_t = torch.tensor(gap, dtype=torch.float32)

    coeff_norms = orbital_data.get("coeff_norms_per_l", None)
    if coeff_norms is None:
        coeff_norms = np.zeros((n_orb, 5), dtype=np.float32)

    quadrupole = orbital_data.get("orbital_quadrupole", None)
    if quadrupole is None:
        quadrupole = np.zeros((n_orb, 3, 3), dtype=np.float32)
    polarization = orbital_data.get("orbital_polarization", None)
    if polarization is None:
        polarization = np.zeros((n_orb, 3), dtype=np.float32)

    # Dominant atom: argmax of |Mulliken weight| per orbital. Anchors the
    # orbital to a specific nucleus so orbital_positions become degenerate-free
    # when orbitals live on different atoms.
    mw_abs = np.abs(orbital_data["mulliken_weights"])
    if mw_abs.shape[1] == 0:
        dominant_atom_idx = np.zeros(n_orb, dtype=np.int64)
    else:
        dominant_atom_idx = np.argmax(mw_abs, axis=1).astype(np.int64)

    energy_t = torch.tensor(energy, dtype=torch.float32) if energy is not None else None
    forces_t = torch.tensor(forces, dtype=torch.float32) if forces is not None else None
    stress_t = torch.tensor(stress, dtype=torch.float32) if stress is not None else None
    cell_t = torch.tensor(cell, dtype=torch.float32) if cell is not None else None
    pbc_t = torch.tensor(pbc, dtype=torch.bool) if pbc is not None else None
    baseline_t = (
        torch.tensor(float(baseline_energy), dtype=torch.float32)
        if baseline_energy is not None else None
    )
    baseline_f_t = (
        torch.tensor(baseline_forces, dtype=torch.float32)
        if baseline_forces is not None else None
    )

    return OrbitalData(
        orbital_positions=torch.tensor(centroids, dtype=torch.float32),
        orbital_features=torch.tensor(orbital_data["orbital_features"], dtype=torch.float32),
        orbital_species=torch.tensor(orbital_data["orbital_species"], dtype=torch.int64),
        ao_shell_encoding=torch.tensor(orbital_data["ao_shell_encoding"], dtype=torch.float32),
        coeff_norms_per_l=torch.tensor(coeff_norms, dtype=torch.float32),
        atom_positions=torch.tensor(orbital_data["atom_positions"], dtype=torch.float32),
        atom_numbers=torch.tensor(orbital_data["atom_numbers"], dtype=torch.int64),
        mulliken_weights=torch.tensor(orbital_data["mulliken_weights"], dtype=torch.float32),
        dominant_atom_idx=torch.tensor(dominant_atom_idx, dtype=torch.int64),
        orbital_quadrupole=torch.tensor(quadrupole, dtype=torch.float32),
        orbital_polarization=torch.tensor(polarization, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.int64),
        edge_vectors=torch.tensor(edge_vecs, dtype=torch.float32),
        edge_fock=edge_fock_t,
        edge_coulomb=edge_coulomb_t,
        edge_exchange=edge_exchange_t,
        edge_hcore=edge_hcore_t,
        edge_energy_gap=edge_energy_gap_t,
        edge_overlap=edge_overlap_t,
        unit_shifts=torch.tensor(shifts, dtype=torch.float32),
        cell=cell_t,
        pbc=pbc_t,
        energy=energy_t,
        forces=forces_t,
        stress=stress_t,
        baseline_energy=baseline_t,
        baseline_forces=baseline_f_t,
        n_orbitals=torch.tensor(n_orb, dtype=torch.int64),
        n_atoms=torch.tensor(n_atoms, dtype=torch.int64),
    )
