# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""PyTorch Dataset and collate function for batched orbital graphs."""

import os
from dataclasses import fields
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .fidelity import list_npz, pack_frame_rows, pack_frame_scalars
from .orbital_graph import OrbitalData


class OrbitalDataset(Dataset):
    """Dataset of OrbitalData objects, loaded from a directory of .npz files
    or from a list of OrbitalData in memory."""

    def __init__(
        self,
        data_list: Optional[List[OrbitalData]] = None,
        data_dir: Optional[str] = None,
        r_cut: float = 6.0,
        orbital_window: Optional[int] = None,
        mf: bool = False,
        lazy: bool = False,
        max_frames: int = 0,
    ):
        # mf=True (multi-fidelity): reconstruct ABSOLUTE targets from the stored
        # residual + baseline so the network's total (delta + base) is trained
        # against the well-behaved absolute E/F. baseline_energy/baseline_forces
        # are kept for the base-head anchor.
        # lazy=True: keep only the file list and build each graph on demand in
        # __getitem__ (flat memory -- required for very large eval/test sets).
        self.mf = mf
        self.lazy = lazy
        self._r_cut = r_cut
        self._orbital_window = orbital_window
        if data_list is not None:
            self.data = data_list
            self._files = None
        elif data_dir is not None:
            # A comma-separated list is accepted, which is how a cheap prep and
            # a smaller expensive one get trained together without physically
            # merging directories that may hold a million files.
            files = list_npz(data_dir)
            # Truncate before building so a subsampled run never pays to
            # construct the graphs it is about to throw away.
            if max_frames:
                files = files[:max_frames]
            if lazy:
                self.data = None
                self._files = files
            else:
                self.data = [self._build_one(p) for p in files]
                self._files = None
        else:
            raise ValueError("Provide data_list or data_dir.")

    def _build_one(self, path: str) -> OrbitalData:
        from .orbital_graph import build_orbital_graph
        npz = np.load(path, allow_pickle=True)
        skip_keys = {"energy", "forces", "stress", "cell", "pbc",
                     "baseline_energy", "baseline_forces"}
        orb_data = {k: npz[k] for k in npz.files if k not in skip_keys}
        energy = float(npz["energy"]) if "energy" in npz.files else None
        forces = npz["forces"] if "forces" in npz.files else None
        stress = npz["stress"] if "stress" in npz.files else None
        cell = npz["cell"] if "cell" in npz.files else None
        pbc = npz["pbc"] if "pbc" in npz.files else None
        baseline = float(npz["baseline_energy"]) if "baseline_energy" in npz.files else None
        baseline_f = npz["baseline_forces"] if "baseline_forces" in npz.files else None
        d = build_orbital_graph(
            orb_data, r_cut=self._r_cut, energy=energy, forces=forces,
            stress=stress, cell=cell, pbc=pbc, baseline_energy=baseline,
            baseline_forces=baseline_f,
            orbital_window=self._orbital_window,
        )
        if self.mf and getattr(d, "baseline_energy", None) is not None:
            # residual (+ baseline) -> absolute reference target. Frames
            # prepared without a baseline already store absolute targets.
            if getattr(d, "energy", None) is not None:
                d.energy = d.energy + d.baseline_energy
            if getattr(d, "forces", None) is not None and \
                    getattr(d, "baseline_forces", None) is not None:
                d.forces = d.forces + d.baseline_forces
        return d

    def __len__(self) -> int:
        return len(self._files) if self.lazy else len(self.data)

    def __getitem__(self, idx: int) -> OrbitalData:
        if self.lazy:
            return self._build_one(self._files[idx])
        return self.data[idx]

    def labeled_frames(self) -> List[bool]:
        """Which frames carry an expensive-level energy, per frame.

        Under a multi-fidelity split only a subset does. In lazy mode this
        reads the npz member list, not the arrays, so it is a directory-sized
        scan rather than a load.
        """
        if not self.lazy:
            return [d.energy is not None for d in self.data]
        out = []
        for p in self._files:
            with np.load(p, allow_pickle=True) as npz:
                out.append("energy" in npz.files)
        return out


def collate_orbital_data(batch: List[OrbitalData]) -> Dict[str, torch.Tensor]:
    """Collate a list of OrbitalData into a batched dict.

    Handles PBC (cell, unit_shifts, stress) alongside the orbital graph.
    """
    orb_offset = 0
    atom_offset = 0

    lists = {
        "orbital_positions": [], "orbital_features": [], "orbital_species": [],
        "ao_shell_encoding": [], "coeff_norms_per_l": [],
        "orbital_quadrupole": [], "orbital_polarization": [],
        "atom_positions": [], "atom_numbers": [],
        "edge_index": [], "edge_vectors": [], "unit_shifts": [],
        "edge_fock": [], "edge_coulomb": [], "edge_exchange": [],
        "edge_hcore": [], "edge_energy_gap": [], "edge_overlap": [],
        "energy": [], "forces": [], "stress": [],
        "baseline_energy": [], "baseline_forces": [],
        "batch_orb": [], "batch_atom": [],
        "mw_orb_idx": [], "mw_atom_idx": [], "mw_values": [],
        "dominant_atom_idx": [],
        "cell": [],
    }
    n_orbitals_list = []
    n_atoms_list = []
    has_pbc = False

    for graph_idx, data in enumerate(batch):
        n_orb = data.orbital_positions.shape[0]
        n_at = data.atom_positions.shape[0]

        lists["orbital_positions"].append(data.orbital_positions)
        lists["orbital_features"].append(data.orbital_features)
        lists["orbital_species"].append(data.orbital_species)
        lists["ao_shell_encoding"].append(data.ao_shell_encoding)
        lists["coeff_norms_per_l"].append(data.coeff_norms_per_l)
        if data.orbital_quadrupole is not None:
            lists["orbital_quadrupole"].append(data.orbital_quadrupole)
        if data.orbital_polarization is not None:
            lists["orbital_polarization"].append(data.orbital_polarization)
        lists["atom_positions"].append(data.atom_positions)
        lists["atom_numbers"].append(data.atom_numbers)

        ei = data.edge_index.clone()
        ei += orb_offset
        lists["edge_index"].append(ei)
        lists["edge_vectors"].append(data.edge_vectors)

        if data.edge_fock is not None:
            lists["edge_fock"].append(data.edge_fock)
        if data.edge_coulomb is not None:
            lists["edge_coulomb"].append(data.edge_coulomb)
        if data.edge_exchange is not None:
            lists["edge_exchange"].append(data.edge_exchange)
        if data.edge_hcore is not None:
            lists["edge_hcore"].append(data.edge_hcore)
        if data.edge_energy_gap is not None:
            lists["edge_energy_gap"].append(data.edge_energy_gap)
        if data.edge_overlap is not None:
            lists["edge_overlap"].append(data.edge_overlap)
        if data.unit_shifts is not None:
            lists["unit_shifts"].append(data.unit_shifts)
        else:
            lists["unit_shifts"].append(torch.zeros(data.edge_index.shape[1], 3))

        if data.cell is not None:
            lists["cell"].append(data.cell)
            has_pbc = True
        else:
            lists["cell"].append(torch.zeros(3, 3))

        mw = data.mulliken_weights
        orb_idx, atom_idx = torch.where(mw.abs() > 1e-8)
        lists["mw_orb_idx"].append(orb_idx + orb_offset)
        lists["mw_atom_idx"].append(atom_idx + atom_offset)
        lists["mw_values"].append(mw[orb_idx, atom_idx])

        if getattr(data, "dominant_atom_idx", None) is not None:
            lists["dominant_atom_idx"].append(data.dominant_atom_idx + atom_offset)
        else:
            lists["dominant_atom_idx"].append(
                torch.zeros(n_orb, dtype=torch.int64) + atom_offset
            )

        lists["batch_orb"].append(torch.full((n_orb,), graph_idx, dtype=torch.int64))
        lists["batch_atom"].append(torch.full((n_at,), graph_idx, dtype=torch.int64))

        # Appended even when absent: a multi-fidelity set labels only some
        # frames at the expensive level, and the packers below need the gaps in
        # order to zero-fill them and report which frames they were.
        lists["energy"].append(data.energy)
        lists["forces"].append(data.forces)
        lists["baseline_energy"].append(getattr(data, "baseline_energy", None))
        lists["baseline_forces"].append(getattr(data, "baseline_forces", None))
        if data.stress is not None:
            lists["stress"].append(data.stress.unsqueeze(0) if data.stress.dim() == 2 else data.stress)

        n_orbitals_list.append(n_orb)
        n_atoms_list.append(n_at)
        orb_offset += n_orb
        atom_offset += n_at

    result = {
        "orbital_positions": torch.cat(lists["orbital_positions"], dim=0),
        "orbital_features": torch.cat(lists["orbital_features"], dim=0),
        "orbital_species": torch.cat(lists["orbital_species"], dim=0),
        "ao_shell_encoding": torch.cat(lists["ao_shell_encoding"], dim=0),
        "coeff_norms_per_l": torch.cat(lists["coeff_norms_per_l"], dim=0),
        "atom_positions": torch.cat(lists["atom_positions"], dim=0),
        "atom_numbers": torch.cat(lists["atom_numbers"], dim=0),
        "edge_index": torch.cat(lists["edge_index"], dim=1),
        "edge_vectors": torch.cat(lists["edge_vectors"], dim=0),
        "unit_shifts": torch.cat(lists["unit_shifts"], dim=0),
        "cell": torch.stack(lists["cell"], dim=0),
        "batch_orb": torch.cat(lists["batch_orb"], dim=0),
        "batch_atom": torch.cat(lists["batch_atom"], dim=0),
        "n_orbitals": torch.tensor(n_orbitals_list, dtype=torch.int64),
        "n_atoms": torch.tensor(n_atoms_list, dtype=torch.int64),
        "num_graphs": len(batch),
        "mw_orb_idx": torch.cat(lists["mw_orb_idx"], dim=0),
        "mw_atom_idx": torch.cat(lists["mw_atom_idx"], dim=0),
        "mw_values": torch.cat(lists["mw_values"], dim=0),
        "dominant_atom_idx": torch.cat(lists["dominant_atom_idx"], dim=0),
        "total_n_orb": orb_offset,
        "total_n_atoms": atom_offset,
        "has_pbc": has_pbc,
    }

    for key, packed, mask in (
        ("energy", *pack_frame_scalars(lists["energy"])),
        ("forces", *pack_frame_rows(lists["forces"], n_atoms_list)),
        ("baseline_energy", *pack_frame_scalars(lists["baseline_energy"])),
        ("baseline_forces", *pack_frame_rows(lists["baseline_forces"], n_atoms_list)),
    ):
        if packed is not None:
            result[key] = packed
        # Only emitted when the batch is partly labeled, so a fully labeled
        # batch carries exactly the keys it always has.
        if mask is not None:
            result[f"{key}_mask"] = mask
    if lists["stress"]:
        result["stress"] = torch.cat(lists["stress"], dim=0)
    if lists["edge_fock"]:
        result["edge_fock"] = torch.cat(lists["edge_fock"], dim=0)
    if lists["edge_coulomb"]:
        result["edge_coulomb"] = torch.cat(lists["edge_coulomb"], dim=0)
    if lists["edge_exchange"]:
        result["edge_exchange"] = torch.cat(lists["edge_exchange"], dim=0)
    if lists["edge_hcore"]:
        result["edge_hcore"] = torch.cat(lists["edge_hcore"], dim=0)
    if lists["edge_energy_gap"]:
        result["edge_energy_gap"] = torch.cat(lists["edge_energy_gap"], dim=0)
    if lists["edge_overlap"]:
        result["edge_overlap"] = torch.cat(lists["edge_overlap"], dim=0)
    if lists["orbital_quadrupole"]:
        result["orbital_quadrupole"] = torch.cat(lists["orbital_quadrupole"], dim=0)
    if lists["orbital_polarization"]:
        result["orbital_polarization"] = torch.cat(lists["orbital_polarization"], dim=0)
    return result
