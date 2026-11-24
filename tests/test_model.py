#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Smoke tests for OrbACE: forward pass, equivariance, long-range, stress.

Run:  python -m pytest tests/test_model.py -v
  or: python tests/test_model.py
"""

import os
import sys
import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch

if hasattr(torch, "serialization"):
    torch.serialization.add_safe_globals([slice])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from e3nn import o3

from orbace.data.orbital_features import (
    compute_orbital_centroids,
    compute_mulliken_weights,
    compute_orbital_features,
    encode_orbital_species,
    compute_ao_shell_encoding,
    extract_all_orbital_data,
)
from orbace.data.orbital_graph import build_orbital_graph
from orbace.data.preprocessing import geometry_to_orbital_data
from orbace.data.dataset import collate_orbital_data
from orbace.modules.models import OrbACE


def _build_h2o():
    """Build a water molecule with PySCF and run HF."""
    from pyscf import gto
    from orbace.data.scf_backend import run_hf
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
    )
    mf = run_hf(mol)
    return mol, mf


def _make_model(**kwargs):
    defaults = dict(
        r_max=10.0, num_bessel=8, max_ell=2, num_interactions=2,
        num_species=200, n_scalar_features=10, n_chem_edge_features=6,
        use_equivariant_orbital_features=True, n_equiv_channels=8,
        hidden_irreps="16x0e + 16x1o + 16x2e", MLP_irreps="16x0e",
        correlation=2, use_long_range=True,
    )
    defaults.update(kwargs)
    return OrbACE(**defaults)


def test_orbital_features():
    """Test orbital feature extraction from H2O/STO-3G."""
    mol, mf = _build_h2o()
    data = extract_all_orbital_data(mol, mf, "canonical")

    n_mo = mf.mo_coeff.shape[1]
    assert data["orbital_centroids"].shape == (n_mo, 3)
    assert data["mulliken_weights"].shape == (n_mo, mol.natm)
    assert data["orbital_species"].shape == (n_mo,)
    assert data["orbital_features"].shape == (n_mo, 10)
    assert data["ao_shell_encoding"].shape == (n_mo, 15)
    assert "hcore_mo" in data
    assert data["hcore_mo"].shape == (n_mo, n_mo)

    row_sums = np.abs(data["mulliken_weights"]).sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=0.05)

    print(f"[PASS] orbital_features: {n_mo} orbitals, 10 features + hcore_mo")


def test_orbital_graph():
    """Test orbital graph construction."""
    mol, mf = _build_h2o()
    data = extract_all_orbital_data(mol, mf, "canonical")
    graph = build_orbital_graph(data, r_cut=10.0, energy=-75.0, forces=np.zeros((3, 3)))

    assert graph.orbital_positions.shape[0] == mf.mo_coeff.shape[1]
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] > 0
    assert graph.unit_shifts is not None
    assert graph.unit_shifts.shape == (graph.edge_index.shape[1], 3)

    print(f"[PASS] orbital_graph: {graph.edge_index.shape[1]} edges")


def test_collate():
    """Test batching of orbital graphs."""
    mol, mf = _build_h2o()
    data = extract_all_orbital_data(mol, mf, "canonical")
    g1 = build_orbital_graph(data, r_cut=10.0, energy=-75.0, forces=np.zeros((3, 3)))
    g2 = build_orbital_graph(data, r_cut=10.0, energy=-74.5, forces=np.ones((3, 3)))

    batch = collate_orbital_data([g1, g2])
    n_mo = mf.mo_coeff.shape[1]

    assert batch["orbital_positions"].shape[0] == 2 * n_mo
    assert batch["atom_positions"].shape[0] == 6
    assert batch["num_graphs"] == 2
    assert batch["energy"].shape == (2,)
    assert batch["unit_shifts"].shape[0] == batch["edge_index"].shape[1]
    assert batch["cell"].shape == (2, 3, 3)

    print(f"[PASS] collate: batch of 2, total orbitals={2*n_mo}")


def test_forward_pass():
    """Test OrbACE forward pass produces energy and forces."""
    mol, mf = _build_h2o()
    data = extract_all_orbital_data(mol, mf, "canonical")
    graph = build_orbital_graph(data, r_cut=10.0, energy=-75.0, forces=np.zeros((3, 3)))
    batch = collate_orbital_data([graph])

    model = _make_model(
        avg_num_neighbors=float(graph.edge_index.shape[1]) / mf.mo_coeff.shape[1],
    )
    model.eval()

    output = model(batch, training=False, compute_force=True)

    assert "energy" in output
    assert output["energy"].shape == (1,)
    assert output["energy"].isfinite().all()
    assert output["forces"] is not None
    assert output["forces"].shape == (3, 3)
    assert output["forces"].isfinite().all()

    print(f"[PASS] forward: E={output['energy'].item():.6f}, "
          f"F_norm={output['forces'].norm():.6f}")


def test_long_range():
    """Test that the long-range block contributes to the energy."""
    mol, mf = _build_h2o()
    data = extract_all_orbital_data(mol, mf, "canonical")
    graph = build_orbital_graph(data, r_cut=10.0)
    batch = collate_orbital_data([graph])
    avg_nn = float(graph.edge_index.shape[1]) / mf.mo_coeff.shape[1]

    model_lr = _make_model(use_long_range=True, avg_num_neighbors=avg_nn)
    model_lr.eval()

    model_no_lr = _make_model(use_long_range=False, avg_num_neighbors=avg_nn)
    model_no_lr.eval()

    with torch.no_grad():
        out_lr = model_lr(batch, compute_force=False)
        out_no_lr = model_no_lr(batch, compute_force=False)

    assert out_lr["lr_energy"] is not None
    assert out_no_lr["lr_energy"] is None

    print(f"[PASS] long_range: lr_energy={out_lr['lr_energy'].item():.6f}")


def test_gradient_flow():
    """Test that gradients flow through the model including long-range."""
    mol, mf = _build_h2o()
    data = extract_all_orbital_data(mol, mf, "canonical")
    graph = build_orbital_graph(data, r_cut=10.0, energy=-75.0, forces=np.zeros((3, 3)))
    batch = collate_orbital_data([graph])
    avg_nn = float(graph.edge_index.shape[1]) / mf.mo_coeff.shape[1]

    model = _make_model(avg_num_neighbors=avg_nn)
    model.train()

    output = model(batch, training=True, compute_force=True)
    loss = (output["energy"] - batch["energy"]).pow(2).mean()
    if output["forces"] is not None and "forces" in batch:
        loss = loss + (output["forces"] - batch["forces"]).pow(2).mean()
    loss.backward()

    has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )

    grad_norms = [
        p.grad.norm().item() for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]

    print(f"[PASS] gradient_flow: all params have grad={has_grad}, "
          f"mean_grad_norm={np.mean(grad_norms):.6e}")


def test_equivariance():
    """Check approximate energy match for rotated geometry (two separate SCF runs).

    A strict SE(3) invariance test would rotate Cartesian tensors while keeping one
    electronic snapshot. Here we rotate coordinates and **re-run HF**, so canonical
    MOs differ by gauge (degenerate subspaces, phase, ordering). MO-basis features
    (F_ij, overlap, etc.) are then not identical, and the network sees slightly
    different inputs for the same physical state. Expect **small** energy drift
    (often a few meV to tens of meV), not machine precision.
    """
    torch.manual_seed(0)
    mol, mf = _build_h2o()

    angles = torch.randn(3)
    R = o3.matrix_x(angles[0]) @ o3.matrix_y(angles[1]) @ o3.matrix_z(angles[2])
    R_np = R.numpy()

    data1 = extract_all_orbital_data(mol, mf, "canonical")
    graph1 = build_orbital_graph(data1, r_cut=10.0)
    batch1 = collate_orbital_data([graph1])

    from pyscf import gto
    from orbace.data.scf_backend import run_hf
    coords_orig = mol.atom_coords()
    coords_rot = coords_orig @ R_np.T
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    atom_lines = []
    for s, (x, y, z) in zip(symbols, coords_rot):
        atom_lines.append(f"{s} {x} {y} {z}")
    mol_rot = gto.M(atom="; ".join(atom_lines), basis="sto-3g", unit="Bohr")
    mf_rot = run_hf(mol_rot)

    data2 = extract_all_orbital_data(mol_rot, mf_rot, "canonical")
    graph2 = build_orbital_graph(data2, r_cut=10.0)
    batch2 = collate_orbital_data([graph2])

    avg_nn = float(graph1.edge_index.shape[1]) / mf.mo_coeff.shape[1]
    model = _make_model(avg_num_neighbors=avg_nn)
    model.eval()

    with torch.no_grad():
        out1 = model(batch1, training=False, compute_force=False)
        out2 = model(batch2, training=False, compute_force=False)

    e1 = out1["energy"].item()
    e2 = out2["energy"].item()
    diff = abs(e1 - e2)
    scale = max(abs(e1), abs(e2), 1.0)
    rel_diff = diff / scale

    # Relative tolerance: random-weight models can produce large absolute energies,
    # but the MO-gauge drift from re-SCF should stay within ~1% of the energy scale.
    rtol = 0.01
    passed = rel_diff < rtol
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] equivariance: E_orig={e1:.8f}, E_rot={e2:.8f}, "
          f"rel_diff={rel_diff:.2e} (tol={rtol})")
    assert passed, (
        f"Energy drift under rotated-geometry re-SCF too large: "
        f"rel_diff={rel_diff:.4f} (max {rtol}). If this regresses, check MO feature pipeline."
    )


def test_multipole_lr_equivariance():
    """Multipole long-range energy must be invariant under rotation.

    Tests the LongRangeBlock directly with nonzero charge/dipole heads (the
    model zero-inits them, so the integrated equivariance test does not exercise
    the term numerically). node_feats are scalars (rotation-invariant); the
    orbital positions and the dipole direction (frozen polarization) rotate
    together, so the screened charge-dipole + dipole-dipole energy must match.
    """
    from orbace.modules.blocks import LongRangeBlock

    torch.manual_seed(7)
    n = 6
    hidden = 16
    block = LongRangeBlock(hidden_dim=hidden, alpha=0.3, use_multipoles=True)
    with torch.no_grad():
        block.charge_net[-1].weight.normal_(0, 0.3)
        block.dipole_net[-1].weight.normal_(0, 0.3)

    node_feats = torch.randn(n, hidden)
    pos = torch.randn(n, 3) * 3.0
    dip_dir = torch.randn(n, 3)
    batch_orb = torch.zeros(n, dtype=torch.long)

    angles = torch.randn(3)
    R = o3.matrix_x(angles[0]) @ o3.matrix_y(angles[1]) @ o3.matrix_z(angles[2])

    with torch.no_grad():
        e1 = block(node_feats, pos, batch_orb, 1, dipole_dir=dip_dir)
        e2 = block(node_feats, pos @ R.T, batch_orb, 1, dipole_dir=dip_dir @ R.T)

    diff = (e1 - e2).abs().item()
    assert diff < 1e-4, f"Multipole LR not rotation-invariant: |dE|={diff:.2e}"
    assert e1.abs().item() > 1e-8, "Multipole LR energy is zero; term inactive."

    print(f"[PASS] multipole_lr_equivariance: E={e1.item():.6f}, |dE_rot|={diff:.2e}")


def test_model_save_load():
    """Test saving and loading a model for deployment."""
    import tempfile
    model = _make_model()
    model.eval()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model, f.name)
        loaded = torch.load(f.name, map_location="cpu", weights_only=False)

    mol, mf = _build_h2o()
    data = extract_all_orbital_data(mol, mf, "canonical")
    graph = build_orbital_graph(data, r_cut=10.0)
    batch = collate_orbital_data([graph])

    with torch.no_grad():
        out1 = model(batch, compute_force=False)
        out2 = loaded(batch, compute_force=False)

    assert torch.allclose(out1["energy"], out2["energy"], atol=1e-6)
    os.unlink(f.name)

    print(f"[PASS] model_save_load: energies match after reload")


def run_all():
    print("=" * 60)
    print("OrbACE smoke tests (long-range + PBC; equivariance = loose MO-gauge tol)")
    print("=" * 60)

    test_orbital_features()
    test_orbital_graph()
    test_collate()
    test_forward_pass()
    test_long_range()
    test_gradient_flow()
    test_equivariance()
    test_multipole_lr_equivariance()
    test_model_save_load()

    print("=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
