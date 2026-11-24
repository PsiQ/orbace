# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Utility functions for OrbACE training."""

import copy
import logging
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_used_parameters(build_fn, n_present: int) -> Optional[int]:
    """Parameters belonging to the species that actually occur in the data.

    ``build_fn(num_species)`` must return the model with every species-
    dependent block sized to ``num_species`` slots. Each such block (one-hot
    embeddings, element-dependent self-connections and product weights,
    per-species heads) holds one equal slice per slot, so the total is affine
    in the slot count, and the slices of slots that never occur multiply a zero
    one-hot entry and can neither contribute to a prediction nor be trained.
    Returns None if the count is not affine, in which case only the allocated
    total is meaningful.

    The throwaway builds run under a forked RNG so the caller's seeded
    initialization stream is left untouched.
    """
    np_state, py_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        counts = [sum(p.numel() for p in build_fn(n).parameters()) for n in (1, 2, 3)]
    np.random.set_state(np_state)
    random.setstate(py_state)
    per_slot = counts[1] - counts[0]
    if counts[2] - counts[1] != per_slot:
        return None
    return counts[0] + per_slot * (n_present - 1)


def compute_orbital_e0s(
    dataset,
    num_species: int = 200,
    data_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage E0 regression for near-zero residual.

    Stage 1 — atom-count regression on reconstructed total energies
    (like MACE).  Requires ``data_dir`` so we can load ``e0s.json``
    and add back the atomic E0 that was subtracted during data prep.

    Stage 2 — per-orbital-species constant regression on the stage-1
    residual.  One constant per species (no orbital-energy slope).

    Returns:
        atom_e0_correction: (max_Z+1,) per-element delta = E0_fitted - E0_json
        e0_base:  (num_species,) orbital base energy
        e0_slope: (num_species,) zeros (kept for API compat)
    """
    import json, os

    e0_json: Dict[int, float] = {}
    if data_dir is not None:
        e0_path = os.path.join(data_dir, "e0s.json")
        if os.path.exists(e0_path):
            with open(e0_path) as f:
                raw = json.load(f)
            e0_json = {int(k): float(v) for k, v in raw.items()}
            logger.info(f"  Loaded atomic E0s from {e0_path} ({len(e0_json)} elements)")

    count_features = []
    atom_counts_list = []
    stored_energies = []
    max_Z = 0
    max_species = 0

    for i in range(len(dataset)):
        d = dataset[i]
        z_max = d.atom_numbers.max().item()
        if z_max > max_Z:
            max_Z = z_max

    n_Z = max_Z + 1

    for i in range(len(dataset)):
        d = dataset[i]
        if d.energy is None:
            continue
        sp = d.orbital_species.numpy()
        if sp.size:
            max_species = max(max_species, int(sp.max()))

        counts = np.zeros(num_species)
        for j, s in enumerate(sp):
            if s < num_species:
                counts[s] += 1
        count_features.append(counts)

        a_nums = d.atom_numbers.numpy()
        a_counts = np.zeros(n_Z)
        for z in a_nums:
            a_counts[z] += 1
        atom_counts_list.append(a_counts)
        stored_energies.append(d.energy.item())

    # The encoding (Z*15 + l*3 + occ) can exceed num_species for heavy
    # elements. Indices beyond the table are clamped in the model (no crash)
    # but distinct species then collapse into one bucket, degrading accuracy.
    # Warn so the user can raise --num_species to >= max_species + 1.
    if max_species >= num_species:
        logger.warning(
            "  Orbital species index reaches %d but num_species=%d. Species "
            ">= num_species are clamped (collapsed) in the model. Re-run with "
            "--num_species %d to represent every species distinctly.",
            max_species, num_species, max_species + 1,
        )

    if not stored_energies:
        return np.zeros(n_Z), np.zeros(num_species), np.zeros(num_species)

    b_stored = np.array(stored_energies)
    A_atom = np.array(atom_counts_list)

    # --- Stage 1: atom-count regression on total energies ----------------
    atom_e0_correction = np.zeros(n_Z)
    if e0_json:
        e0_add_back = np.zeros(len(b_stored))
        for i, a_counts in enumerate(atom_counts_list):
            for z, cnt in enumerate(a_counts):
                if cnt > 0 and z in e0_json:
                    e0_add_back[i] += cnt * e0_json[z]
        b_total = b_stored + e0_add_back

        used_z = np.where(A_atom.sum(axis=0) > 0)[0]
        if len(used_z) > 0:
            A_z = A_atom[:, used_z]
            try:
                z_coeffs, _, _, _ = np.linalg.lstsq(A_z, b_total, rcond=None)
            except np.linalg.LinAlgError:
                z_coeffs = np.zeros(len(used_z))

            atom_e0_fitted = np.zeros(n_Z)
            for idx, z in enumerate(used_z):
                atom_e0_fitted[z] = z_coeffs[idx]

            res1 = b_total - A_atom @ atom_e0_fitted
            logger.info(
                f"  Stage 1 (atom-count on total E): {len(used_z)} elements, "
                f"residual MAE={np.mean(np.abs(res1)):.4f} eV, "
                f"RMSE={np.sqrt(np.mean(res1**2)):.4f} eV"
            )

            for z in range(n_Z):
                atom_e0_correction[z] = atom_e0_fitted[z] - e0_json.get(z, 0.0)

            b_stage2 = res1
        else:
            b_stage2 = b_stored
    else:
        logger.info("  No e0s.json found; skipping stage-1 atom-count regression")
        b_stage2 = b_stored

    # Stage 2 (per-orbital-species constant) is disabled — the NN learns
    # the full residual after atom-count correction.  Return zeros for
    # backward compatibility with callers that still unpack 3 values.
    e0_base = np.zeros(num_species)
    e0_slope = np.zeros(num_species)
    logger.info("  Stage 2 skipped (orbital E0 disabled); NN learns full residual")

    return atom_e0_correction, e0_base, e0_slope


def compute_statistics(
    dataset,
    e0_base: Optional[np.ndarray] = None,
    atom_e0_correction: Optional[np.ndarray] = None,
    num_species: int = 200,
) -> Tuple[float, float, float]:
    """Compute training statistics: avg_num_neighbors, mean, rms.

    Subtracts the atom-level E0 correction (stage 1) and optionally
    the orbital-level E0 (stage 2) before computing per-orbital mean
    interaction energy and force RMS.

    Returns:
        avg_num_neighbors, mean, rms
    """
    atom_energy_list = []
    forces_list = []
    num_neighbors_list = []

    for i in range(len(dataset)):
        d = dataset[i]
        n_orb = d.orbital_positions.shape[0]

        if d.edge_index.shape[1] > 0:
            _, counts = torch.unique(d.edge_index[1], return_counts=True)
            num_neighbors_list.append(counts.float())

        if d.energy is not None:
            e0_sum = 0.0
            if e0_base is not None:
                sp = d.orbital_species.numpy()
                for j, s in enumerate(sp):
                    if s < num_species:
                        e0_sum += e0_base[s]
            atom_corr = 0.0
            if atom_e0_correction is not None:
                for z in d.atom_numbers.numpy():
                    if z < len(atom_e0_correction):
                        atom_corr += atom_e0_correction[z]
            e_inter = d.energy.item() - e0_sum - atom_corr
            atom_energy_list.append(e_inter / n_orb)

        if d.forces is not None:
            forces_list.append(d.forces.numpy().ravel())

    avg_num_neighbors = 1.0
    if num_neighbors_list:
        avg_num_neighbors = float(torch.cat(num_neighbors_list).mean().item())

    mean = float(np.mean(atom_energy_list)) if atom_energy_list else 0.0

    if forces_list:
        forces_flat = np.concatenate(forces_list)
        rms = float(np.sqrt(np.mean(forces_flat ** 2)))
    else:
        rms = float(np.std(atom_energy_list)) if atom_energy_list else 1.0

    if rms < 1e-10:
        rms = 1.0

    return avg_num_neighbors, mean, rms


class EMAModel:
    """Exponential Moving Average of model parameters.

    Delays averaging until after ``warmup_steps`` updates so the shadow
    is not contaminated by random-init parameters.  On the first post-warmup
    call the shadow is hard-reset to the current model weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999,
                 warmup_steps: int = 0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self._step = 0
        self._initialised = warmup_steps == 0
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._step += 1
        if not self._initialised:
            if self._step <= self.warmup_steps:
                return
            for name, p in model.named_parameters():
                self.shadow[name].copy_(p.data)
            self._initialised = True
            return
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: dict):
        for name, p in model.named_parameters():
            p.data.copy_(backup[name])

    def get_backup(self, model: nn.Module) -> dict:
        return {name: p.data.clone() for name, p in model.named_parameters()}


class SWAModel:
    """Stochastic Weight Averaging."""

    def __init__(self, model: nn.Module):
        self.avg_params = {name: p.clone().detach() for name, p in model.named_parameters()}
        self.n_averaged = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.n_averaged += 1
        for name, p in model.named_parameters():
            self.avg_params[name] += (p.data - self.avg_params[name]) / self.n_averaged

    def apply(self, model: nn.Module):
        for name, p in model.named_parameters():
            p.data.copy_(self.avg_params[name])


def compute_energy_statistics(dataset) -> Tuple[float, float]:
    """Compute per-orbital shift and RMS-force scale.

    Used by train.py (--method orbital) when energy_scale/shift are at defaults.
    Returns (shift, scale) where:
      shift = mean per-orbital energy
      scale = RMS of all force components
    """
    atom_energy_list = []
    forces_list = []

    for i in range(len(dataset)):
        d = dataset[i]
        n_orb = d.orbital_positions.shape[0]
        if d.energy is not None and n_orb > 0:
            atom_energy_list.append(d.energy.item() / n_orb)
        if d.forces is not None:
            forces_list.append(d.forces.numpy().ravel())

    shift = float(np.mean(atom_energy_list)) if atom_energy_list else 0.0

    if forces_list:
        forces_flat = np.concatenate(forces_list)
        scale = float(np.sqrt(np.mean(forces_flat ** 2)))
    else:
        scale = float(np.std(atom_energy_list)) if atom_energy_list else 1.0

    if scale < 1e-10:
        scale = 1.0

    return shift, scale


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
