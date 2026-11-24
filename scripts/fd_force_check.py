# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Finite-difference force-consistency check for an OrbACE-style calculator.

For a *conservative* potential, the autograd force must equal the central
finite-difference of the energy actually evaluated along the trajectory:

    F_autograd[a,k]  ?=  -(E(R + h e_{a,k}) - E(R - h e_{a,k})) / (2h)

The current OrbACE recomputes its PySCF features at every geometry but feeds
them to the network as detached constants. Its autograd force therefore only
follows the orbital-centroid geometry path and drops the dominant
feature-response term -- so this check is expected to FAIL (large mismatch)
for the legacy model and PASS for the differentiable-featurizer rework.

Usage:
    python scripts/fd_force_check.py            # untrained legacy model on H2O
    python scripts/fd_force_check.py --h 1e-3
"""

import argparse

import numpy as np
from ase import Atoms


def water(d=0.9584, ang=104.45):
    import math
    a = math.radians(ang / 2.0)
    return Atoms(
        "OHH",
        positions=[
            (0.0, 0.0, 0.0),
            (d * math.sin(a), d * math.cos(a), 0.0),
            (-d * math.sin(a), d * math.cos(a), 0.0),
        ],
    )


def build_legacy_calc(basis, localizer, r_cut):
    """Untrained legacy OrbACE wrapped in its ASE calculator."""
    import torch
    from orbace.modules.models import OrbACE
    from orbace.calculators.ase_calc import OrbACECalculator

    torch.manual_seed(0)
    model = OrbACE(
        r_max=r_cut,
        hidden_irreps="8x0e",
        MLP_irreps="4x0e",
        num_interactions=1,
        correlation=2,
        num_species=128,                 # min to cover O (species index ~125)
        use_long_range=False,
        pair_repulsion=False,            # isolate the orbital-feature energy path
        use_equivariant_orbital_features=True,
        use_chem_edge_features=True,
    )
    model.eval()
    calc = OrbACECalculator(
        model=model,
        basis=basis,
        localizer=localizer,
        r_cut=r_cut,
        device="cpu",
        cache_orbitals=False,            # force feature recompute at every geometry
        baseline_model="none",
    )
    return calc


def fd_forces(calc, atoms, h):
    """Central finite-difference forces (eV/Ang) of the evaluated energy."""
    base = atoms.copy()
    p0 = base.get_positions()
    n = len(base)
    F = np.zeros((n, 3))
    for a in range(n):
        for k in range(3):
            ap = base.copy()
            pp = p0.copy(); pp[a, k] += h; ap.set_positions(pp); ap.calc = clone_calc(calc)
            em = ap.get_potential_energy()
            am = base.copy()
            pm = p0.copy(); pm[a, k] -= h; am.set_positions(pm); am.calc = clone_calc(calc)
            en = am.get_potential_energy()
            F[a, k] = -(em - en) / (2.0 * h)
    return F


def clone_calc(calc):
    """Fresh calculator sharing the same model (avoid result caching)."""
    from orbace.calculators.ase_calc import OrbACECalculator
    return OrbACECalculator(
        model=calc.model,
        basis=calc.basis,
        localizer=calc.localizer,
        r_cut=calc.r_cut,
        device=str(calc.device),
        cache_orbitals=False,
        baseline_model=calc.baseline_model,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--localizer", default="boys")
    ap.add_argument("--r_cut", type=float, default=5.0)
    ap.add_argument("--h", type=float, default=1e-3)
    ap.add_argument("--mode", default="legacy", choices=["legacy"])
    args = ap.parse_args()

    atoms = water()
    calc = build_legacy_calc(args.basis, args.localizer, args.r_cut)

    atoms.calc = clone_calc(calc)
    F_auto = atoms.get_forces()
    F_fd = fd_forces(calc, atoms, args.h)

    diff = F_auto - F_fd
    max_abs = np.max(np.abs(diff))
    rms = np.sqrt(np.mean(diff ** 2))
    fd_scale = np.max(np.abs(F_fd)) + 1e-12

    np.set_printoptions(precision=4, suppress=True)
    print(f"mode={args.mode} basis={args.basis} h={args.h}")
    print("F_autograd (eV/A):\n", F_auto)
    print("F_finite_diff (eV/A):\n", F_fd)
    print("F_auto - F_fd (eV/A):\n", diff)
    print(f"\nmax|F_true| (FD)        = {fd_scale:.4f} eV/A")
    print(f"max|F_auto - F_fd|      = {max_abs:.4f} eV/A")
    print(f"rms(F_auto - F_fd)      = {rms:.4f} eV/A")
    print(f"relative max error      = {max_abs / fd_scale:.2%}")
    ok = max_abs < 1e-2
    print(f"\nFORCE-CONSISTENCY: {'PASS' if ok else 'FAIL'} (threshold 1e-2 eV/A)")


if __name__ == "__main__":
    main()
