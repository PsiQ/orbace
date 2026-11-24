# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Validate diff_integrals against PySCF (values) and finite diff (gradients).

Run either way:  python tests/test_diff_integrals.py   |   pytest tests/test_diff_integrals.py
"""
import os
import sys

import numpy as np
import torch
from pyscf import gto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbace.data.diff_integrals import GaussianBasis  # noqa: E402

BOHR = 0.52917721090380
TOL = 1e-8


def build(basis="sto-3g"):
    # H2O geometry in Angstrom
    sym = ["O", "H", "H"]
    pos_ang = np.array([
        [0.0, 0.0, 0.119262],
        [0.0, 0.763239, -0.477047],
        [0.0, -0.763239, -0.477047],
    ])
    mol = gto.M(atom=[(s, p) for s, p in zip(sym, pos_ang)], basis=basis,
                unit="Angstrom", cart=True)
    Z = np.array([8, 1, 1])
    pos_bohr = torch.tensor(pos_ang / BOHR, dtype=torch.float64)
    return mol, Z, pos_bohr


def test_values_match_pyscf():
    mol, Z, pos_bohr = build()
    gb = GaussianBasis(Z, "sto-3g")
    assert gb.n_ao == mol.nao_nr()

    S = gb.overlap(pos_bohr).detach().numpy()
    dip, r2 = gb.moments(pos_bohr)
    errs = {
        "overlap": np.max(np.abs(S - mol.intor("int1e_ovlp"))),
        "dipole": np.max(np.abs(dip.detach().numpy() - mol.intor("int1e_r", comp=3))),
        "r2": np.max(np.abs(r2.detach().numpy() - mol.intor("int1e_r2"))),
    }
    print("max |mine - pyscf|: " + ", ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    for name, err in errs.items():
        assert err < TOL, f"{name}: {err:.3e}"


def test_gradient_matches_finite_difference():
    _, Z, pos_bohr = build()
    gb = GaussianBasis(Z, "sto-3g")

    def scalar(pos):
        S = gb.overlap(pos)
        dip, r2 = gb.moments(pos)
        return S.sum() + dip.sum() + r2.sum()

    pos = pos_bohr.clone().requires_grad_(True)
    scalar(pos).backward()
    g_auto = pos.grad.detach().numpy()

    h = 1e-5
    base = pos_bohr.numpy()
    g_fd = np.zeros_like(g_auto)
    for a in range(base.shape[0]):
        for k in range(3):
            pp = base.copy(); pp[a, k] += h
            pm = base.copy(); pm[a, k] -= h
            g_fd[a, k] = (scalar(torch.tensor(pp)).item()
                          - scalar(torch.tensor(pm)).item()) / (2 * h)
    err = np.max(np.abs(g_auto - g_fd))
    print(f"max |autograd - finite difference| = {err:.2e}")
    assert err < 1e-7, err


if __name__ == "__main__":
    test_values_match_pyscf()
    test_gradient_matches_finite_difference()
    print("diff_integrals checks passed")
