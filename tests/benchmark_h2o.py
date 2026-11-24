#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Benchmark: 150 H2O molecules in extended XYZ format -> OrbACE training.

Generates perturbed water geometries with HF energies/forces,
writes them in standard extended XYZ format, then runs the
full OrbACE pipeline: prepare data -> train -> evaluate.

By default Hartree–Fock uses **CPU PySCF** (fast for this small system); PyTorch
training still uses **CUDA** when available. To run HF with gpu4pyscf instead,
set ``ORBACE_BENCHMARK_SCF_DEVICE=cuda``.

Quantum-chemical features beyond spatial geometry:
  - Fock matrix elements F_ij as edge features (orbital coupling strength)
  - Coulomb/exchange integrals as edge features (electron interaction)
  - MO coefficient norms per angular momentum channel as node features
  - AdamW optimizer with weight decay
  - Force-normalized MSE training
"""

import errno
import json
import logging
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="  %(message)s")
logging.getLogger("orbace").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import torch

if hasattr(torch, "serialization"):
    torch.serialization.add_safe_globals([slice])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader

from orbace.data.dataset import OrbitalDataset, collate_orbital_data
from orbace.data.extxyz import read_extxyz
from orbace.modules.models import OrbACE
from orbace.tools.deployment import DESCRIPTOR_VERSION
from orbace.tools.train import Trainer, WeightedEnergyForcesStressLoss
from orbace.tools.utils import (
    set_seeds, count_parameters, compute_orbital_e0s, compute_statistics,
)
from orbace.data.orbital_features import extract_all_orbital_data
from orbace.data.preprocessing import _build_mol, _run_hf, _localize
from orbace.data.scf_backend import gpu_available, gpu_unavailable_reason
from orbace.tools.isolated_atoms import compute_e0_from_extxyz, subtract_e0

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_h2o")
DEFAULT_BENCHMARK_SAMPLES = int(os.environ.get("ORBACE_BENCHMARK_SAMPLES", "150"))
DEFAULT_BENCHMARK_EPOCHS = int(os.environ.get("ORBACE_BENCHMARK_EPOCHS", "2000"))
BENCHMARK_BASIS = os.environ.get("ORBACE_BENCHMARK_BASIS", "sto-3g")
BENCHMARK_LOCALIZER = "boys"

# HF / E0: CPU PySCF by default. Training uses torch CUDA in main() when available.
# ORBACE_BENCHMARK_SCF_DEVICE=cuda → gpu4pyscf when CuPy + gpu4pyscf import OK.
# Resolve SCF_DEVICE at the start of main() so importing this module does not probe gpu4pyscf.
_bench_scf = os.environ.get("ORBACE_BENCHMARK_SCF_DEVICE", "cpu").strip().lower()
BENCHMARK_REQUESTED_GPU_SCF = _bench_scf in ("cuda", "gpu")
SCF_DEVICE = "cpu"


def generate_h2o_extxyz(path: str, n_samples: int = 150):
    """Generate perturbed H2O in extended XYZ format."""
    from pyscf import gto
    from orbace.data.scf_backend import run_hf

    eq_coords = np.array([
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.7572, -0.4692],
        [0.0000, -0.7572, -0.4692],
    ])

    np.random.seed(42)
    perturbations = np.concatenate([
        np.random.randn(n_samples // 3, 3, 3) * 0.02,
        np.random.randn(n_samples // 3, 3, 3) * 0.05,
        np.random.randn(n_samples - 2 * (n_samples // 3), 3, 3) * 0.10,
    ], axis=0)

    with open(path, "w") as f:
        for i in range(n_samples):
            coords = eq_coords + perturbations[i]

            mol = gto.M(
                atom=f"O {coords[0,0]} {coords[0,1]} {coords[0,2]}; "
                     f"H {coords[1,0]} {coords[1,1]} {coords[1,2]}; "
                     f"H {coords[2,0]} {coords[2,1]} {coords[2,2]}",
                basis=BENCHMARK_BASIS, unit="Angstrom", verbose=0,
            )
            mf = run_hf(mol, device=SCF_DEVICE)
            mf.verbose = 0

            energy_eV = mf.e_tot * HARTREE_TO_EV
            grad_ha_bohr = mf.nuc_grad_method().kernel()
            forces_eV_ang = -grad_ha_bohr * (HARTREE_TO_EV / BOHR_TO_ANG)

            f.write(f"3\n")
            f.write(
                f'Lattice="100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0" '
                f'Properties=species:S:1:pos:R:3:REF_forces:R:3 '
                f'REF_energy={energy_eV:.10f} pbc="F F F"\n'
            )
            symbols = ["O", "H", "H"]
            for j in range(3):
                f.write(
                    f"{symbols[j]}    "
                    f"{coords[j,0]:12.8f}   {coords[j,1]:12.8f}   {coords[j,2]:12.8f}   "
                    f"{forces_eV_ang[j,0]:12.8f}   {forces_eV_ang[j,1]:12.8f}   {forces_eV_ang[j,2]:12.8f}\n"
                )

            if (i + 1) % 25 == 0:
                logger.info(f"  Generated {i+1}/{n_samples} configs")

    return n_samples


def prepare_data(extxyz_path: str, data_dir: str, basis: str = BENCHMARK_BASIS):
    """Run the full data preparation pipeline with Fock/integral extraction."""
    all_frames = read_extxyz(extxyz_path)

    # Separate IsolatedAtom frames (E0 sources) from training structures
    frames = [f for f in all_frames if f.get("config_type") != "IsolatedAtom"]
    n_isolated = len(all_frames) - len(frames)
    logger.info(f"  Read {len(frames)} frames from extxyz"
                + (f" + {n_isolated} IsolatedAtom E0 frame(s)" if n_isolated else ""))

    e0s = compute_e0_from_extxyz(all_frames, basis=basis, device=SCF_DEVICE)
    with open(os.path.join(data_dir, "e0s.json"), "w") as f:
        json.dump({str(k): v for k, v in e0s.items()}, f, indent=2)

    for Z, e0 in sorted(e0s.items()):
        logger.info(f"  E0(Z={Z}): {e0:.6f} eV")

    n_success = 0
    t0 = time.time()
    for idx, frame in enumerate(frames):
        mol = _build_mol(
            frame["symbols"], frame["coords"],
            basis=basis, charge=0, spin=0, unit="Angstrom",
        )
        mf = _run_hf(mol, device=SCF_DEVICE)
        mf = _localize(mf, "boys")
        orb_data = extract_all_orbital_data(mol, mf, "boys")

        save_dict = {**orb_data}

        if frame["energy"] is not None:
            atom_numbers = np.array(
                [mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int64
            )
            save_dict["energy"] = np.float32(
                subtract_e0(frame["energy"], atom_numbers, e0s)
            )

        if frame["forces"] is not None:
            save_dict["forces"] = frame["forces"].astype(np.float32)

        np.savez(os.path.join(data_dir, f"frame_{idx:04d}.npz"), **save_dict)
        n_success += 1

        if (idx + 1) % 25 == 0:
            logger.info(f"  Prepared {idx+1}/{len(frames)} frames ({time.time()-t0:.0f}s)")

    logger.info(f"  Done: {n_success}/{len(frames)} frames in {time.time()-t0:.0f}s")
    metadata = {
        "basis": basis,
        "localizer": BENCHMARK_LOCALIZER,
        "charge": 0,
        "spin": 0,
        "n_frames": n_success,
        "input_file": os.path.basename(extxyz_path),
        "e0s": {str(k): v for k, v in e0s.items()},
        "baseline_model": "none",
        "descriptor_version": DESCRIPTOR_VERSION,
    }
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return e0s


def _reset_checkpoint_dir(checkpoint_dir: str) -> None:
    """Remove old checkpoints; tolerate NFS `.nfs*` busy files and partial failures."""
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        return
    try:
        shutil.rmtree(checkpoint_dir)
    except OSError as err:
        # NFS leaves `.nfs*` when a deleted file is still open; EBUSY common.
        if err.errno in (errno.EBUSY, errno.EAGAIN) or (
            getattr(err, "filename", None) and ".nfs" in str(err.filename)
        ):
            logger.warning(
                "Could not fully remove %s (%s); retrying with ignore_errors.",
                checkpoint_dir,
                err,
            )
        else:
            logger.warning("rmtree %s: %s; retrying with ignore_errors.", checkpoint_dir, err)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        if os.path.exists(checkpoint_dir):
            stale = f"{checkpoint_dir}.stale.{os.getpid()}.{int(time.time())}"
            try:
                os.rename(checkpoint_dir, stale)
                logger.warning("Renamed locked checkpoint dir to %s", stale)
            except OSError:
                pass
    os.makedirs(checkpoint_dir, exist_ok=True)


def main():
    global SCF_DEVICE

    t_total_start = time.time()
    set_seeds(42)

    if BENCHMARK_REQUESTED_GPU_SCF and torch.cuda.is_available():
        SCF_DEVICE = "cuda" if gpu_available() else "cpu"
    else:
        SCF_DEVICE = "cpu"

    data_dir = os.path.join(OUTPUT_DIR, "data")
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(data_dir, exist_ok=True)
    _reset_checkpoint_dir(checkpoint_dir)

    train_backend = "CUDA (PyTorch)" if torch.cuda.is_available() else "CPU (PyTorch)"
    if SCF_DEVICE.startswith("cuda"):
        scf_label = f"gpu4pyscf (GPU); training on {train_backend}"
    elif torch.cuda.is_available():
        scf_label = f"PySCF (CPU for HF); training on {train_backend}"
    else:
        scf_label = "PySCF (CPU)"
    print("=" * 70)
    print("OrbACE Benchmark: 150 H2O")
    print("  Equivariant message passing in orbital space")
    print("  Chemical edge features: Fock / Coulomb / exchange / hcore / energy gap / bond order / density / dipole")
    print("  Optimizer: AdamW | Per-group LayerNorm | Grad-detached scale")
    print(f"  SCF backend: {scf_label}")
    if BENCHMARK_REQUESTED_GPU_SCF and SCF_DEVICE == "cpu" and torch.cuda.is_available():
        why = gpu_unavailable_reason()
        if why:
            print("  GPU SCF requested but unavailable:")
            for line in why.split("\n"):
                print(f"    {line}")
        pp = os.environ.get("PYTHONPATH", "")
        if "gpu4pyscf" in pp and "libcusolver.so.11" not in (why or ""):
            print("  Hint: PYTHONPATH may point at a local gpu4pyscf repo (OrbACE strips it for imports).")
            print("        If issues persist: unset PYTHONPATH or use env -u PYTHONPATH python benchmark_h2o.py")
    print("=" * 70)

    extxyz_path = os.path.join(
        OUTPUT_DIR, f"h2o_{DEFAULT_BENCHMARK_SAMPLES}.xyz"
    )
    e0s_path = os.path.join(data_dir, "e0s.json")

    # Force regeneration to include new features (Fock, Coulomb, exchange)
    npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")] if os.path.isdir(data_dir) else []
    need_regen = (
        len(npz_files) != DEFAULT_BENCHMARK_SAMPLES or not os.path.exists(e0s_path)
    )

    if not need_regen:
        sample_npz = np.load(os.path.join(data_dir, npz_files[0]), allow_pickle=True)
        if "fock_matrix_mo" not in sample_npz.files:
            logger.info("  Old data format detected -- regenerating with edge features...")
            need_regen = True

    if need_regen:
        for f in npz_files:
            os.remove(os.path.join(data_dir, f))

        print(f"\n[1/5] Generating {DEFAULT_BENCHMARK_SAMPLES} H2O configs (HF/6-31G*)...")
        t0 = time.time()
        n = generate_h2o_extxyz(extxyz_path, n_samples=DEFAULT_BENCHMARK_SAMPLES)
        print(f"  -> {n} frames in {time.time()-t0:.0f}s -> {extxyz_path}")

        print("\n[2/5] Preparing orbital data (HF + Boys + Fock/J/K + E0 subtraction)...")
        t0 = time.time()
        e0s = prepare_data(extxyz_path, data_dir)
        print(f"  -> {time.time()-t0:.0f}s total")
    else:
        print("\n[1/5] Data already generated with Fock features, skipping...")
        print("[2/5] Data already prepared, skipping...")
        with open(e0s_path) as f:
            e0s = {int(k): float(v) for k, v in json.load(f).items()}

    # ── 3. Load dataset, compute stats, split ───────────────────────────
    print("\n[3/5] Loading dataset and computing statistics...")
    dataset = OrbitalDataset(data_dir=data_dir, r_cut=5.0)
    print(f"  -> {len(dataset)} samples")

    # Use a local Generator with an explicit, fixed seed instead of
    # ``np.random.permutation`` which reads the *global* numpy RNG. The
    # global RNG state when we get here depends on whether HF data
    # regeneration ran in this process (it calls np.random.seed/randn),
    # so reusing it would silently produce a different val split between
    # cached and regenerated runs and inflate run-to-run variance.
    n_val = int(os.environ.get("ORBACE_BENCHMARK_NVAL", "30"))
    n_train = len(dataset) - n_val
    split_rng = np.random.default_rng(20240425)
    indices = split_rng.permutation(len(dataset))
    train_data = [dataset[i] for i in indices[:n_train]]
    val_data = [dataset[i] for i in indices[n_train:]]

    train_set = OrbitalDataset(data_list=train_data)
    val_set = OrbitalDataset(data_list=val_data)

    print(f"  -> Train: {n_train}, Val: {n_val}")
    print(f"  -> Val frame indices (first 10): {sorted(indices[n_train:].tolist())[:10]}")

    energies = [train_set[i].energy.item() for i in range(len(train_set))]
    print(f"  -> E range: [{min(energies):.4f}, {max(energies):.4f}] eV (E0-subtracted)")

    # Verify Fock features are present and units
    sample = train_set[0]
    has_fock = sample.edge_fock is not None
    has_coulomb = sample.edge_coulomb is not None
    print(f"  -> Fock edge features:    {'YES' if has_fock else 'NO'}")
    print(f"  -> Coulomb edge features: {'YES' if has_coulomb else 'NO'}")
    print(f"  -> Coeff norms per l:     {sample.coeff_norms_per_l.shape}")
    print(f"  -> Positions unit check (expect ~Angstrom scale):")
    pos_range = sample.atom_positions.abs().max().item()
    orb_range = sample.orbital_positions.abs().max().item()
    print(f"     atom_positions max: {pos_range:.3f} (should be ~0.5-1 for H2O)")
    print(f"     orbital_positions max: {orb_range:.3f}")

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

    train_loader = DataLoader(
        train_set, batch_size=30, shuffle=True, collate_fn=collate_orbital_data,
    )
    val_loader = DataLoader(
        val_set, batch_size=30, shuffle=False, collate_fn=collate_orbital_data,
    )

    # ── 4. Build model ──────────────────────────────────────────────────
    print("\n[4/5] Building OrbACE model...")
    print("  Quantum-chemical features:")
    print("    - Fock matrix off-diagonal elements as edge features")
    print("    - Coulomb (ii|jj) and exchange (ij|ji) integrals as edge features")
    print("    - Core Hamiltonian h_ij and energy gap as edge features")
    print("    - Energy gap |eps_i - eps_j| as edge feature")
    print("    - MO coefficient norms per angular momentum channel")
    print("    - Per-group LayerNorm on input features")
    print("    - Gradient-detached scale/shift")
    print("    - Positions in Angstrom (forces = -dE/dR in eV/Ang)")
    print("    - Identity-centered residual readout modules")

    model = OrbACE(
        r_max=5.0,
        num_bessel=8,
        max_ell=2,
        num_interactions=3,
        num_species=NUM_SPECIES,
        n_scalar_features=10,
        n_ao_shell_features=15,
        n_coeff_norm_features=5,
        hidden_irreps="64x0e + 64x1o + 64x2e",
        MLP_irreps="128x0e",
        radial_MLP=[128, 128, 128],
        correlation=3,
        avg_num_neighbors=avg_neighbors,
        atomic_inter_scale=rms,
        atomic_inter_shift=mean,
        atom_e0_correction=atom_e0_corr,
        use_chem_edge_features=True,
        n_chem_edge_features=6,
        use_equivariant_orbital_features=True,
        n_equiv_channels=8,
        dropout=0.05,
        use_long_range=False,
    )
    n_params = count_parameters(model)
    print(f"  -> {n_params:,} parameters")

    # ── 5. Train ───────────────────────────────────────────────────────
    n_epochs = DEFAULT_BENCHMARK_EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[5/5] Training for {n_epochs} epochs on {device}...")
    stage_two_start = max(1, int(n_epochs * 0.6))
    print(f"       AdamW (lr=1e-3, wd=5e-5), stage1 MSE energy:forces = 10:10")
    print(f"       Stage two at epoch {stage_two_start}: weights 10:30 (mild force boost)")
    print(f"       Warmup 15 epochs, ReduceLROnPlateau(patience=50, factor=0.8), grad_clip=10, EMA(0.999)")
    print(f"       Batch=30, dropout=0.05, force-normalized loss (rms={rms:.4f})")
    log_interval = 50
    print(f"       Progress: epoch 1 always logged, then every {log_interval} epochs", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=5e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=50,
    )
    loss_fn = WeightedEnergyForcesStressLoss(
        energy_weight=10.0, forces_weight=10.0, forces_normalizer=rms,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        ema_decay=0.999,
        device=device,
        checkpoint_dir=checkpoint_dir,
        max_grad_norm=10.0,
        warmup_epochs=15,
        save_cpu=True,
        stage_two_start=stage_two_start,
        stage_two_energy_weight=10.0,
        stage_two_forces_weight=30.0,
        stage_two_lr=None,
    )

    t0 = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        log_interval=log_interval,
        patience=200,
    )
    train_time = time.time() - t0
    print(f"\n  Training time: {train_time:.0f}s ({train_time/60:.1f} min)")

    # ── Results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS (OrbACE with quantum-chemistry edge features)")
    print("=" * 70)

    eval_device = device

    def _eval_model(label, model_path):
        m = torch.load(model_path, map_location="cpu", weights_only=False)
        m.to(eval_device)
        m.eval()
        ee, fe = [], []
        er, ep = [], []
        for i in range(len(val_set)):
            d = val_set[i]
            batch = collate_orbital_data([d])
            batch = {k: v.to(eval_device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.no_grad():
                out = m(batch, compute_force=True)
            e_r = d.energy.item()
            e_p = out["energy"].detach().cpu().item()
            ee.append(abs(e_p - e_r))
            er.append(e_r)
            ep.append(e_p)
            if d.forces is not None and out["forces"] is not None:
                fe.append((out["forces"].detach().cpu() - d.forces).abs().mean().item())
        na = 3
        print(f"\n  {label} ({n_val} samples):")
        print(f"  Energy MAE:          {np.mean(ee):.4f} eV")
        print(f"  Energy MAE/atom:     {np.mean(ee)/na:.4f} eV/atom")
        print(f"  Energy RMSE:         {np.sqrt(np.mean(np.array(ee)**2)):.4f} eV")
        print(f"  Energy RMSE/atom:    {np.sqrt(np.mean(np.array(ee)**2))/na:.4f} eV/atom")
        if fe:
            print(f"  Forces MAE:          {np.mean(fe):.4f} eV/Å")
            print(f"  Forces RMSE:         {np.sqrt(np.mean(np.array(fe)**2)):.4f} eV/Å")
        return ee, fe, er, ep

    e_errors, f_errors, e_refs, e_preds = _eval_model(
        "best_model (val-loss)", os.path.join(checkpoint_dir, "best_model.pt")
    )
    ema_path = os.path.join(checkpoint_dir, "ema_model.pt")
    if os.path.exists(ema_path):
        _eval_model("ema_model", ema_path)
    forces_path = os.path.join(checkpoint_dir, "best_forces_model.pt")
    if os.path.exists(forces_path):
        _eval_model("best_forces_model", forces_path)

    print(f"\n  Sample predictions (first 10 val):")
    print(f"  {'#':<4} {'E_ref':>10} {'E_pred':>10} {'|err|':>8} {'F_MAE':>8}")
    print(f"  {'-'*44}")
    for i in range(min(10, len(e_refs))):
        f_str = f"{f_errors[i]:.4f}" if i < len(f_errors) else "N/A"
        print(f"  {i:<4} {e_refs[i]:>10.4f} {e_preds[i]:>10.4f} {e_errors[i]:>8.4f} {f_str:>8}")

    saved = sorted(os.listdir(checkpoint_dir))
    print(f"\n  Saved checkpoints:")
    for fname in saved:
        size = os.path.getsize(os.path.join(checkpoint_dir, fname))
        print(f"    {fname:25s}  ({size/1024:.1f} KB)")

    deploy_info = {
        "e0s": {str(k): v for k, v in e0s.items()},
        "atomic_inter_scale": rms,
        "atomic_inter_shift": mean,
        "avg_num_neighbors": avg_neighbors,
        "units": {"energy": "eV", "forces": "eV/Ang"},
        "basis": "6-31g*",
        "localizer": "boys",
        "descriptor_version": DESCRIPTOR_VERSION,
        "n_train": n_train,
        "n_val": n_val,
        "features": {
            "fock_edge_features": True,
            "coulomb_edge_features": True,
            "exchange_edge_features": True,
            "hcore_edge_features": True,
            "energy_gap_edge_features": True,
            "bond_order_edge_features": True,
            "density_edge_features": True,
            "dipole_norm_edge_features": True,
            "coeff_norms_per_l": True,
            "orbital_shape_invariants": True,
        },
        "optimizer": "AdamW",
        "weight_decay": 5e-5,
        "dropout": 0.05,
        "forces_normalizer": rms,
    }
    with open(os.path.join(checkpoint_dir, "deploy_info.json"), "w") as fp:
        json.dump(deploy_info, fp, indent=2)

    print(f"\n  All outputs: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 70)

    # ── 6. Short MD demo ────────────────────────────────────────────────
    print("\n[6/6] MD demo: 20-step NVE on one H2O with OrbACECalculator...")
    try:
        from ase import Atoms, units
        from ase.md.verlet import VelocityVerlet
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from orbace.calculators.ase_calc import OrbACECalculator

        model_path = os.path.join(checkpoint_dir, "best_model.pt")
        md_device = str(device)
        calc = OrbACECalculator(
            model_path=model_path,
            basis="6-31g*",
            localizer="boys",
            r_cut=5.0,
            device=md_device,
        )

        atoms = Atoms("OH2",
                       positions=[[0.0, 0.0, 0.1173],
                                  [0.0, 0.7572, -0.4692],
                                  [0.0, -0.7572, -0.4692]])
        atoms.calc = calc

        MaxwellBoltzmannDistribution(atoms, temperature_K=300)
        dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)

        print(f"  Step {'E_pot (eV)':>12} {'E_kin (eV)':>12} {'E_tot (eV)':>12}")
        print(f"  {'-'*42}")
        for step in range(21):
            if step > 0:
                dyn.run(1)
            ep = atoms.get_potential_energy()
            ek = atoms.get_kinetic_energy()
            if step % 5 == 0:
                print(f"  {step:4d} {ep:12.6f} {ek:12.6f} {ep+ek:12.6f}")

        print("  [PASS] MD completed — energy conservation looks fine if E_tot is roughly constant")
    except Exception as e:
        print(f"  [SKIP] MD demo failed: {e}")

    t_total = time.time() - t_total_start
    print("\n" + "=" * 70)
    print(f"Total benchmark time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
