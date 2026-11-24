#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""End-to-end test: generate extxyz -> prepare data -> train -> predict.

Simulates the real user workflow:
  1. Generate a small extxyz file (water molecules, energies in eV, forces in eV/Å)
  2. Prepare the data: compute E0s and orbital features
  3. Train the model
  4. Load and predict

Usage:
    python tests/test_train_e2e.py
"""

import os
import sys
import warnings
import shutil
import json
import logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="  %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import pytest
import torch

if hasattr(torch, "serialization"):
    torch.serialization.add_safe_globals([slice])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader

from orbace.data.dataset import OrbitalDataset, collate_orbital_data
from orbace.data.extxyz import read_extxyz as read_extxyz_shared
from orbace.modules.models import OrbACE
from orbace.tools.train import Trainer, WeightedHuberEnergyForcesStressLoss
from orbace.tools.utils import (
    set_seeds, count_parameters, compute_orbital_e0s, compute_statistics,
)
from orbace.tools.isolated_atoms import (
    subtract_e0,
    HARTREE_TO_EV,
)
from orbace.data.scf_backend import gpu_available

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_output")
BOHR_TO_ANG = 0.529177210903
# CPU PySCF for QC by default; PyTorch training uses CUDA in main() when available.
# ORBACE_BENCHMARK_SCF_DEVICE=cuda → gpu4pyscf for HF when the stack imports.
# Resolve SCF_DEVICE in main() so importing this module does not probe gpu4pyscf.
_bench_scf = os.environ.get("ORBACE_BENCHMARK_SCF_DEVICE", "cpu").strip().lower()
BENCHMARK_REQUESTED_GPU_SCF = _bench_scf in ("cuda", "gpu")
SCF_DEVICE = "cpu"


def generate_extxyz(path: str, n_samples: int = 6):
    """Generate a small extxyz file with perturbed water molecules.

    Energies computed via HF (converted to eV), forces via HF gradient (eV/Å).
    This mimics what a user would provide.
    """
    from pyscf import gto
    from orbace.data.scf_backend import run_hf

    with open(path, "w") as f:
        for i in range(n_samples):
            np.random.seed(42 + i)
            noise = np.random.randn(3, 3) * 0.05
            coords_ang = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.757, 0.587],
                [0.0, -0.757, 0.587],
            ]) + noise

            mol = gto.M(
                atom=f"O {coords_ang[0,0]} {coords_ang[0,1]} {coords_ang[0,2]}; "
                     f"H {coords_ang[1,0]} {coords_ang[1,1]} {coords_ang[1,2]}; "
                     f"H {coords_ang[2,0]} {coords_ang[2,1]} {coords_ang[2,2]}",
                basis="sto-3g", unit="Angstrom",
            )
            mf = run_hf(mol, device=SCF_DEVICE)

            energy_eV = mf.e_tot * HARTREE_TO_EV
            forces_ha_bohr = -mf.nuc_grad_method().kernel()
            forces_eV_ang = forces_ha_bohr * (HARTREE_TO_EV / BOHR_TO_ANG)

            f.write(f"3\n")
            f.write(f'energy={energy_eV:.10f} pbc="F F F"\n')
            symbols = ["O", "H", "H"]
            for j in range(3):
                f.write(f"{symbols[j]} "
                        f"{coords_ang[j,0]:.8f} {coords_ang[j,1]:.8f} {coords_ang[j,2]:.8f} "
                        f"{forces_eV_ang[j,0]:.8f} {forces_eV_ang[j,1]:.8f} {forces_eV_ang[j,2]:.8f}\n")

    return n_samples


def run_prepare_data(extxyz_path: str, data_dir: str):
    """Run the prepare_data pipeline (in-process)."""
    from orbace.data.orbital_features import extract_all_orbital_data
    from orbace.data.preprocessing import _build_mol, _run_hf, _localize
    from orbace.tools.isolated_atoms import compute_e0_from_extxyz

    frames = read_extxyz_shared(extxyz_path)
    print(f"  -> Read {len(frames)} frames from extxyz")

    # Compute E0s
    e0s = compute_e0_from_extxyz(frames, basis="sto-3g", device=SCF_DEVICE)
    with open(os.path.join(data_dir, "e0s.json"), "w") as f:
        json.dump({str(k): v for k, v in e0s.items()}, f, indent=2)

    for Z, e0 in sorted(e0s.items()):
        print(f"  -> E0(Z={Z}): {e0:.6f} eV")

    n_success = 0
    for idx, frame in enumerate(frames):
        mol = _build_mol(
            frame["symbols"], frame["coords"],
            basis="sto-3g", charge=0, spin=0, unit="Angstrom",
        )
        mf = _run_hf(mol, device=SCF_DEVICE)
        mf = _localize(mf, "boys")
        orb_data = extract_all_orbital_data(mol, mf, "boys")

        save_dict = {**orb_data}

        # Energy in eV, E0-subtracted
        if frame["energy"] is not None:
            atom_numbers = np.array(
                [mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int64
            )
            save_dict["energy"] = np.float32(
                subtract_e0(frame["energy"], atom_numbers, e0s)
            )

        # Forces in eV/Ang
        if frame["forces"] is not None:
            save_dict["forces"] = frame["forces"].astype(np.float32)

        np.savez(os.path.join(data_dir, f"frame_{idx:04d}.npz"), **save_dict)
        n_success += 1

    print(f"  -> Processed {n_success} frames, labels in eV / eV/Å (E0-subtracted)")
    return e0s


def main():
    global SCF_DEVICE

    set_seeds(42)
    if BENCHMARK_REQUESTED_GPU_SCF and torch.cuda.is_available():
        SCF_DEVICE = "cuda" if gpu_available() else "cpu"
    else:
        SCF_DEVICE = "cpu"

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    data_dir = os.path.join(OUTPUT_DIR, "data")
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_backend = "CUDA (PyTorch)" if torch.cuda.is_available() else "CPU (PyTorch)"
    if SCF_DEVICE.startswith("cuda"):
        scf_label = f"gpu4pyscf (GPU); training on {train_backend}"
    elif torch.cuda.is_available():
        scf_label = f"PySCF (CPU for HF); training on {train_backend}"
    else:
        scf_label = "PySCF (CPU)"
    print("=" * 60)
    print("OrbACE End-to-End Test (eV/Å, E0 subtraction)")
    print(f"  SCF backend: {scf_label}")
    print("=" * 60)

    # 1. Generate extxyz
    extxyz_path = os.path.join(OUTPUT_DIR, "water.xyz")
    print("\n[1/6] Generating extxyz (10 perturbed water molecules)...")
    n = generate_extxyz(extxyz_path, n_samples=10)
    print(f"  -> Wrote {n} frames to {extxyz_path}")
    print(f"  -> Units: Å, eV, eV/Å")

    # 2. Prepare data (HF + localization + E0 subtraction)
    print("\n[2/6] Preparing orbital data (HF + Boys + E0 subtraction)...")
    e0s = run_prepare_data(extxyz_path, data_dir)

    # 3. Load dataset and compute statistics
    print("\n[3/6] Loading dataset and computing energy statistics...")
    dataset = OrbitalDataset(data_dir=data_dir, r_cut=6.0)
    print(f"  -> {len(dataset)} samples loaded")

    n_val = max(2, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_data = [dataset[i] for i in range(n_train)]
    val_data = [dataset[i] for i in range(n_train, len(dataset))]

    train_set = OrbitalDataset(data_list=train_data)
    val_set = OrbitalDataset(data_list=val_data)

    # Show some sample energies
    for i in range(min(3, len(dataset))):
        d = dataset[i]
        print(f"  -> Sample {i}: E={d.energy.item():.4f} eV (E0-sub), "
              f"F_max={d.forces.abs().max().item():.4f} eV/Å")

    # Full-batch for stability on tiny dataset
    train_loader = DataLoader(
        train_set, batch_size=n_train, shuffle=True, collate_fn=collate_orbital_data,
    )
    val_loader = DataLoader(
        val_set, batch_size=n_val, shuffle=False, collate_fn=collate_orbital_data,
    )

    NUM_SPECIES = 200
    print("\n  Computing orbital E0s (two-stage regression)...")
    atom_e0_corr, _e0_base, _e0_slope = compute_orbital_e0s(
        train_set, num_species=NUM_SPECIES, data_dir=data_dir,
    )

    avg_neighbors, mean, rms = compute_statistics(
        train_set,
        atom_e0_correction=atom_e0_corr, num_species=NUM_SPECIES,
    )
    print(f"  -> Avg neighbors (orbital graph): {avg_neighbors:.1f}")
    print(f"  -> Per-orbital interaction mean:   {mean:.6f} eV")
    print(f"  -> Force RMS (scale):              {rms:.6f} eV/Å")

    # 4. Build model -- small model for small data, but with L=1 + L=2
    print("\n[4/6] Building model...")
    model = OrbACE(
        r_max=6.0, num_bessel=8, max_ell=2,
        num_interactions=2, num_species=NUM_SPECIES,
        hidden_irreps="16x0e + 16x1o + 16x2e",
        MLP_irreps="32x0e",
        radial_MLP=[64, 64, 64],
        correlation=2, use_long_range=False,
        avg_num_neighbors=avg_neighbors,
        atomic_inter_scale=rms,
        atomic_inter_shift=mean,
        atom_e0_correction=atom_e0_corr,
    )
    n_params = count_parameters(model)
    print(f"  -> {n_params:,} parameters")

    # 5. Train -- AdamW with force-normalized Huber loss
    n_epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_two_start = max(1, int(n_epochs * 0.75))
    print(f"\n[5/6] Training for {n_epochs} epochs on {device}...")
    print(f"       Stage-two at epoch {stage_two_start}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_epochs - 10), eta_min=1e-6,
    )
    loss_fn = WeightedHuberEnergyForcesStressLoss(
        energy_weight=10.0, forces_weight=10.0, huber_delta=0.01,
        forces_normalizer=rms,
    )

    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        loss_fn=loss_fn, device=device, checkpoint_dir=checkpoint_dir,
        max_grad_norm=5.0, warmup_epochs=10, save_cpu=True,
        stage_two_start=stage_two_start,
        stage_two_energy_weight=50.0,
        stage_two_forces_weight=100.0,
        stage_two_lr=None,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        log_interval=50,
        patience=200,
    )

    # 6. Load and predict
    print(f"\n[6/6] Loading best_model.pt and predicting...")
    saved_files = sorted(os.listdir(checkpoint_dir))
    for f_name in saved_files:
        size = os.path.getsize(os.path.join(checkpoint_dir, f_name))
        print(f"  -> {f_name:25s}  ({size / 1024:.1f} KB)")

    loaded_model = torch.load(
        os.path.join(checkpoint_dir, "best_model.pt"),
        map_location="cpu", weights_only=False,
    )
    loaded_model.eval()

    print("\n  Predictions (eV / eV/Å):")
    print(f"  {'Sample':<8} {'E_ref':>12} {'E_pred':>12} {'E_err':>10} {'F_MAE':>10}")
    print(f"  {'-'*52}")

    e_errors = []
    f_errors = []
    for i in range(len(dataset)):
        d = dataset[i]
        batch = collate_orbital_data([d])
        with torch.no_grad():
            out = loaded_model(batch, compute_force=True)

        e_ref = d.energy.item()
        e_pred = out["energy"].item()
        e_err = abs(e_pred - e_ref)
        e_errors.append(e_err)

        f_mae = (out["forces"] - d.forces).abs().mean().item() if d.forces is not None else 0.0
        f_errors.append(f_mae)

        print(f"  {i:<8} {e_ref:>12.4f} {e_pred:>12.4f} {e_err:>10.4f} {f_mae:>10.4f}")

    print(f"\n  Mean |E_err|: {np.mean(e_errors):.4f} eV")
    print(f"  Mean F_MAE:   {np.mean(f_errors):.4f} eV/Å")

    # Save E0s alongside model for deployment
    deploy_info = {
        "e0s": {str(k): v for k, v in e0s.items()},
        "atomic_inter_scale": rms,
        "atomic_inter_shift": mean,
        "avg_num_neighbors": avg_neighbors,
        "units": {"energy": "eV", "forces": "eV/Ang"},
        "basis": "sto-3g",
        "localizer": "boys",
        "features": {
            "fock_edge_features": True,
            "coulomb_edge_features": True,
            "exchange_edge_features": True,
            "hcore_edge_features": True,
            "energy_gap_edge_features": True,
            "coeff_norms_per_l": True,
            "orbital_shape_invariants": True,
        },
    }
    with open(os.path.join(checkpoint_dir, "deploy_info.json"), "w") as f:
        json.dump(deploy_info, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"End-to-end test PASSED")
    print(f"All outputs: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"=" * 60)


@pytest.mark.slow
def test_main():
    """pytest entry point.

    Unlike the other scripts here main() returns None and signals failure by
    raising, so there is no return code to assert on. Marked slow because it
    runs a real prepare -> train -> eval cycle (~2 min) and rewrites
    test_output/; deselect with -m 'not slow'.
    """
    main()


if __name__ == "__main__":
    main()
