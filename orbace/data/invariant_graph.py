# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Atom-graph container, builder, dataset and collate for the invariant model.

Mirrors orbital_graph.py / dataset.py but for the gauge-invariant, atom-centered
featurization (see tb_features.py). Nodes are ATOMS; features are smooth,
localization-gauge-invariant density-matrix descriptors. Equivariance comes from
the geometry (spherical harmonics on atom-atom edges), so node/edge features are
rotation-invariant scalars.
"""

import os
from dataclasses import dataclass, fields
from typing import Dict, List, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset, Sampler

from .fidelity import list_npz, pack_frame_rows, pack_frame_scalars
from .tb_features import PAIR_KEYS_DEFAULT as PAIR_KEYS


@dataclass
class InvariantData:
    """Single atom graph with gauge-invariant features.

    atom_positions   (n_atoms, 3)  nuclear positions [Angstrom]
    atom_numbers      (n_atoms,)    atomic numbers Z
    node_scalars      (n_atoms, D)  gauge-invariant per-atom features
    edge_index        (2, n_edges)  sender/receiver atom indices
    edge_features     (n_edges, 6)  gauge-invariant atom-pair features
    energy            scalar        target energy [eV] (E0/baseline subtracted)
    forces            (n_atoms, 3)  target forces [eV/Ang] (baseline subtracted)
    charge            scalar        total charge; None means neutral
    """

    atom_positions: torch.Tensor
    atom_numbers: torch.Tensor
    node_scalars: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    unit_shifts: Optional[torch.Tensor] = None
    cell: Optional[torch.Tensor] = None
    pbc: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None
    forces: Optional[torch.Tensor] = None
    baseline_energy: Optional[torch.Tensor] = None
    baseline_forces: Optional[torch.Tensor] = None
    n_atoms: Optional[torch.Tensor] = None
    charge: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "InvariantData":
        kwargs = {}
        for f in fields(self):
            val = getattr(self, f.name)
            kwargs[f.name] = val.to(device) if isinstance(val, torch.Tensor) else val
        return InvariantData(**kwargs)


def _neighbor_list_isolated(pos: NDArray, r_cut: float):
    dists = cdist(pos, pos)
    mask = dists < r_cut
    np.fill_diagonal(mask, False)
    senders, receivers = np.where(mask)
    shifts = np.zeros((len(senders), 3), dtype=np.float32)
    return senders, receivers, shifts


def _neighbor_list_pbc(pos: NDArray, cell: NDArray, pbc: NDArray, r_cut: float):
    lengths = np.linalg.norm(cell, axis=1)
    n_images = np.where(pbc, np.ceil(r_cut / np.clip(lengths, 1e-6, None)).astype(int), 0)
    senders, receivers, shifts_list = [], [], []
    for nx in range(-n_images[0], n_images[0] + 1):
        for ny in range(-n_images[1], n_images[1] + 1):
            for nz in range(-n_images[2], n_images[2] + 1):
                shift = np.array([nx, ny, nz], dtype=np.float32)
                shifted = pos + shift @ cell
                mask = cdist(pos, shifted) < r_cut
                if nx == ny == nz == 0:
                    np.fill_diagonal(mask, False)
                s, r = np.where(mask)
                if len(s):
                    senders.append(s)
                    receivers.append(r)
                    shifts_list.append(np.tile(shift, (len(s), 1)))
    if senders:
        return (np.concatenate(senders), np.concatenate(receivers),
                np.concatenate(shifts_list).astype(np.float32))
    return (np.array([], np.int64), np.array([], np.int64),
            np.zeros((0, 3), np.float32))


def build_invariant_graph(
    feats: Dict[str, NDArray],
    r_cut: float = 5.0,
    energy: Optional[float] = None,
    forces: Optional[NDArray] = None,
    cell: Optional[NDArray] = None,
    pbc: Optional[NDArray] = None,
    baseline_energy: Optional[float] = None,
    baseline_forces: Optional[NDArray] = None,
) -> InvariantData:
    """Build an atom graph from a feature dict (a loaded npz, or geometry and
    targets assembled in-memory)."""
    pos = np.asarray(feats["atom_positions"], dtype=np.float64)
    n_atoms = pos.shape[0]

    has_pbc = cell is not None and pbc is not None and np.any(pbc)
    if has_pbc:
        s, r, shifts = _neighbor_list_pbc(pos, cell, pbc, r_cut)
    else:
        s, r, shifts = _neighbor_list_isolated(pos, r_cut)

    edge_index = (np.stack([s, r], axis=0).astype(np.int64)
                  if len(s) else np.zeros((2, 0), dtype=np.int64))

    # QC node/edge features are OPTIONAL: the conservative (TB) backend recomputes
    # its own extended-Hückel features from geometry in-forward and ignores these,
    # so a "no-SCF" TB prep can omit them entirely. Fall back to placeholder zeros.
    have_pairs = len(s) and all(k in feats and feats[k] is not None for k in PAIR_KEYS)
    if have_pairs:
        cols = [np.asarray(feats[k])[s, r] for k in PAIR_KEYS]
        edge_features = np.stack(cols, axis=1).astype(np.float32)
    else:
        edge_features = np.zeros((len(s), len(PAIR_KEYS)), dtype=np.float32)

    node_scalars = feats.get("node_scalars")
    if node_scalars is None:
        node_scalars = np.zeros((n_atoms, 1), dtype=np.float32)

    def _t(x, dt):
        return None if x is None else torch.tensor(x, dtype=dt)

    return InvariantData(
        atom_positions=torch.tensor(pos, dtype=torch.float32),
        atom_numbers=torch.tensor(np.asarray(feats["atom_numbers"]), dtype=torch.int64),
        node_scalars=torch.tensor(np.asarray(node_scalars), dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.int64),
        edge_features=torch.tensor(edge_features, dtype=torch.float32),
        unit_shifts=torch.tensor(shifts, dtype=torch.float32),
        cell=_t(cell, torch.float32),
        pbc=_t(pbc, torch.bool),
        energy=_t(None if energy is None else float(energy), torch.float32),
        forces=_t(forces, torch.float32),
        baseline_energy=_t(None if baseline_energy is None else float(baseline_energy), torch.float32),
        baseline_forces=_t(baseline_forces, torch.float32),
        n_atoms=torch.tensor(n_atoms, dtype=torch.int64),
    )


class InvariantDataset(Dataset):
    """Dataset of InvariantData, from a directory of .npz or an in-memory list."""

    def __init__(self, data_list=None, data_dir=None, r_cut: float = 5.0,
                 basis: str = "sto-3g", mf: bool = False, lazy: bool = False,
                 max_frames: int = 0):
        self.basis = basis
        # mf=True (multi-fidelity): reconstruct ABSOLUTE targets from the stored
        # residual + baseline so the network's total (base + delta) is trained
        # against the well-behaved absolute E/F (never the inflated residual).
        # baseline_energy / baseline_forces are kept for the base-head anchor.
        # lazy=True: keep only the file list and build each graph on demand in
        # __getitem__ (flat memory -- required for very large eval/test sets).
        self.mf = mf
        self.lazy = lazy
        self._r_cut = r_cut
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

    def _build_one(self, path: str) -> InvariantData:
        npz = np.load(path, allow_pickle=True)
        # node_scalars / PAIR_KEYS are optional (absent in no-SCF TB preps).
        want = ("atom_positions", "atom_numbers", "node_scalars", *PAIR_KEYS)
        feats = {k: npz[k] for k in want if k in npz.files}
        d = build_invariant_graph(
            feats, r_cut=self._r_cut,
            energy=float(npz["energy"]) if "energy" in npz.files else None,
            forces=npz["forces"] if "forces" in npz.files else None,
            cell=npz["cell"] if "cell" in npz.files else None,
            pbc=npz["pbc"] if "pbc" in npz.files else None,
            baseline_energy=float(npz["baseline_energy"]) if "baseline_energy" in npz.files else None,
            baseline_forces=npz["baseline_forces"] if "baseline_forces" in npz.files else None,
        )
        if "charge" in npz.files:
            d.charge = torch.tensor(int(npz["charge"]), dtype=torch.int64)
        if self.mf and d.baseline_energy is not None:
            # residual (+ baseline) -> absolute reference target. Frames
            # prepared without a baseline (e.g. a plain validation set)
            # already store absolute targets, so they are left untouched.
            if d.energy is not None:
                d.energy = d.energy + d.baseline_energy
            if d.forces is not None and d.baseline_forces is not None:
                d.forces = d.forces + d.baseline_forces
        return d

    def _load_from_dir(self, data_dir: str, r_cut: float) -> List[InvariantData]:
        files = sorted(os.path.join(data_dir, f)
                       for f in os.listdir(data_dir) if f.endswith(".npz"))
        return [self._build_one(p) for p in files]

    def __len__(self):
        return len(self._files) if self.lazy else len(self.data)

    def __getitem__(self, idx):
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

    def tb_ao_counts(self) -> List[int]:
        """Per-frame AO count, for the bucketed sampler.

        In lazy mode this reads only the ``atom_numbers`` array from each file.
        npz members decompress on access, so it costs one small read per frame
        rather than building the graphs -- which matters for the 442k-frame test
        set, where building them is the thing lazy mode exists to avoid. That
        scan is still one stat+read per frame (~12 min cold for 442k), largely
        repaid by the page cache it warms for the pass that follows.
        """
        from .tb_features import tb_num_ao
        if not self.lazy:
            return [tb_num_ao(d.atom_numbers.tolist()) for d in self.data]
        counts = []
        for p in self._files:
            with np.load(p, allow_pickle=True) as npz:
                counts.append(tb_num_ao(npz["atom_numbers"].tolist()))
        return counts


class BucketedBatchSampler(Sampler):
    """Batches molecules of similar size together, for the TB feature backend.

    The batched TB eigensolve pads every molecule up to the batch's largest AO
    count, so a batch costs ``batch_size * nao_max^3`` rather than
    ``sum(nao_i^3)``. ANI-1x spans nao 6..105, which makes random batches waste
    4-5x the necessary FLOPs; grouping by size removes essentially all of it and
    roughly doubles the featurizer throughput.

    Randomness is kept where it matters. Each epoch the dataset is shuffled, cut
    into pools of ``batch_size * pool_batches`` frames, and only sorted by size
    *within* a pool; the resulting batches are then emitted in random order. So
    batch membership and batch order both change every epoch, and only the
    size-heterogeneity inside a single batch is given up. That last point is a
    real change to the optimization, which is why this is opt-in and applied only
    to the backend that benefits.

    ``pool_batches`` sets that trade-off. Measured on the 10k-frame ANI-1x
    training set at batch 16 (padding waste, then mean within-batch nao spread):
    random 3.38x / 9.3, pool 8 -> 1.47x / 2.6, pool 16 -> 1.27x / 1.5,
    pool 50 -> 1.08x / 0.6. The default of 8 deliberately leaves some FLOPs on
    the table to keep batches heterogeneous, since the effect of homogeneous
    batches on convergence has not been measured over a full run.
    """

    def __init__(self, sizes: List[int], batch_size: int, shuffle: bool = True,
                 drop_last: bool = False, pool_batches: int = 8,
                 seed: int = 0):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.sizes = list(sizes)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        # A larger pool sorts more globally (less padding) at the cost of less
        # variation in batch membership between epochs.
        self.pool = max(self.batch_size, self.batch_size * int(pool_batches))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> List[List[int]]:
        n = len(self.sizes)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(n, generator=g).tolist()
        else:
            # Deterministic: a full sort is optimal and order is irrelevant when
            # no gradients are taken.
            g = None
            order = sorted(range(n), key=lambda i: self.sizes[i])

        batches = []
        for start in range(0, n, self.pool):
            pool = order[start:start + self.pool]
            if self.shuffle:
                pool = sorted(pool, key=lambda i: self.sizes[i])
            for b in range(0, len(pool), self.batch_size):
                batch = pool[b:b + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1          # so a plain training loop still varies by epoch
        return iter(batches)

    def __len__(self):
        n = len(self.sizes)
        if self.drop_last:
            per_pool = self.pool // self.batch_size
            full, rem = divmod(n, self.pool)
            return full * per_pool + rem // self.batch_size
        total = 0
        for start in range(0, n, self.pool):
            m = len(range(start, min(start + self.pool, n)))
            total += -(-m // self.batch_size)
        return total


def _graph_charge(d) -> int:
    """Total charge of one graph; 0 when the frame does not carry one.

    Datasets prepared before charge was threaded through have no such field,
    and those are all neutral, so absence and zero mean the same thing here.
    """
    q = getattr(d, "charge", None)
    if q is None:
        return 0
    return int(q.item()) if torch.is_tensor(q) else int(q)


def collate_invariant_data(batch: List[InvariantData],
                           tb_pack: bool = False) -> Dict[str, torch.Tensor]:
    """Collate a list of graphs into one batch dict.

    ``tb_pack`` adds the padded extended-Hückel AO bookkeeping (``tb_*`` keys)
    that the TB feature backend needs. It depends only on the element sequences,
    so building it here means it happens in the DataLoader worker rather than
    inside the training step, where deriving it costs a device sync per batch.
    Off by default: the QC backend never reads it.
    """
    atom_offset = 0
    pos, Z, node, ei, ef, shifts = [], [], [], [], [], []
    energy, forces, baseline_e, baseline_f = [], [], [], []
    batch_atom, cell = [], []
    n_atoms_list = []
    has_pbc = False

    for gi, d in enumerate(batch):
        n_at = d.atom_positions.shape[0]
        pos.append(d.atom_positions)
        Z.append(d.atom_numbers)
        node.append(d.node_scalars)
        ei.append(d.edge_index + atom_offset)
        ef.append(d.edge_features)
        if d.unit_shifts is not None:
            shifts.append(d.unit_shifts)
        else:
            shifts.append(torch.zeros(d.edge_index.shape[1], 3))
        if d.cell is not None:
            cell.append(d.cell)
            has_pbc = True
        else:
            cell.append(torch.zeros(3, 3))
        batch_atom.append(torch.full((n_at,), gi, dtype=torch.int64))
        # Appended even when absent: a multi-fidelity set labels only some
        # frames at the expensive level, and the packers below need the gaps in
        # order to zero-fill them and report which frames they were.
        energy.append(d.energy)
        forces.append(d.forces)
        baseline_e.append(d.baseline_energy)
        baseline_f.append(d.baseline_forces)
        n_atoms_list.append(n_at)
        atom_offset += n_at

    out = {
        "atom_positions": torch.cat(pos, 0),
        "atom_numbers": torch.cat(Z, 0),
        "node_scalars": torch.cat(node, 0),
        "edge_index": torch.cat(ei, 1),
        "edge_features": torch.cat(ef, 0),
        "unit_shifts": torch.cat(shifts, 0),
        "cell": torch.stack(cell, 0),
        "batch_atom": torch.cat(batch_atom, 0),
        "n_atoms": torch.tensor(n_atoms_list, dtype=torch.int64),
        "num_graphs": len(batch),
        "total_n_atoms": atom_offset,
        "has_pbc": has_pbc,
    }
    for key, packed, mask in (
        ("energy", *pack_frame_scalars(energy)),
        ("forces", *pack_frame_rows(forces, n_atoms_list)),
        ("baseline_energy", *pack_frame_scalars(baseline_e)),
        ("baseline_forces", *pack_frame_rows(baseline_f, n_atoms_list)),
    ):
        if packed is not None:
            out[key] = packed
        # Only emitted when the batch is partly labeled, so a fully labeled
        # batch carries exactly the keys it always has.
        if mask is not None:
            out[f"{key}_mask"] = mask
    charges = [_graph_charge(d) for d in batch]
    if any(charges):
        # Emitted only when something is actually charged, so batches from the
        # neutral datasets that predate this keep exactly the keys they had.
        # Consumers that derive TB bookkeeping themselves (eval, the ASE
        # calculator) read this; without it they would featurize an ion as
        # neutral and never say so.
        out["charge"] = torch.tensor(charges, dtype=torch.int64)
    if tb_pack:
        from .tb_features import build_tb_pack
        out.update(build_tb_pack(
            [tuple(int(z) for z in d.atom_numbers.tolist()) for d in batch],
            charges))
    return out
