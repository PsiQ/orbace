# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Absolute correctness of the TB featurizer gradient, via finite differences.

This is the check that matters for conservative forces: the atom_cons family
differentiates these features to get forces, so a wrong feature gradient is a
wrong force no matter how well the network trains.

The molecules are chosen by symmetry, not variety. A symmetric molecule has
*exactly* degenerate occupied levels (CH4's t2 triplet, NH3's e doublet), which
is precisely where an eigenvector-based backward needs a 1/(e_i - e_j)
regularization and where that regularization silently biases the result -- the
energy-weighted density matrix W contributes a term whose true value survives
the degeneracy but which broadening sends to zero. Before the spectral
(Daleckii-Krein) backward, CH4 here was off by 25% while H2O looked fine, so a
test on asymmetric molecules alone would have passed and proved nothing.

Run directly:  python tests/test_tb_gradient_exact.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbace.data.tb_features import (  # noqa: E402
    _ELEMENTS, _molecule_features, _molecule_matrices, _sym_orthogonalize,
)

torch.manual_seed(0)

MOLECULES = {
    "H2O": ([8, 1, 1],
            [[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]]),
    "CH4 (Td)": ([6, 1, 1, 1, 1],
                 [[0.0, 0.0, 0.0], [0.629, 0.629, 0.629],
                  [-0.629, -0.629, 0.629], [-0.629, 0.629, -0.629],
                  [0.629, -0.629, -0.629]]),
    "NH3 (C3v)": ([7, 1, 1, 1],
                  [[0.0, 0.0, 0.120], [0.0, 0.939, -0.280],
                   [0.813, -0.469, -0.280], [-0.813, -0.469, -0.280]]),
    "C2H4O": ([6, 6, 8, 1, 1, 1, 1],
              [[0.0, 0.0, 0.0], [1.50, 0.05, 0.10], [2.10, 1.20, -0.15],
               [-0.50, 0.95, 0.20], [-0.45, -0.90, 0.55], [-0.30, -0.15, -1.00],
               [1.95, -0.85, 0.35]]),
}

# Central differences in float64 truncate at O(h^2 * |f'''|); with h=1e-5 the
# achievable agreement is ~1e-9, so anything below 1e-7 is FD noise, not error.
FD_STEP = 1e-5
FD_TOL = 1e-7


def occupied_gap(coords, zs):
    """Smallest spacing between occupied levels -- the quantity that breaks it."""
    with torch.no_grad():
        _P, _W, S, H, _F, _aa, _al = _molecule_matrices(coords, zs)
        X = _sym_orthogonalize(S)
        e = torch.linalg.eigvalsh(X @ H @ X)
    n_occ = int(round(sum(_ELEMENTS[int(z)]["valence_Z"] for z in zs))) // 2
    eo = e[:n_occ]
    gaps = (eo[1:] - eo[:-1]).abs()
    return (gaps.min().item() if gaps.numel() else float("inf"),
            (e[n_occ] - e[n_occ - 1]).item())


def projected_loss(coords, zs, wn, wp):
    """Random projection of all features, so no column can cancel out."""
    node, pair = _molecule_features(coords, zs)
    return (node * wn).sum() + (pair * wp).sum()


def check(name, zs, xyz):
    coords = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
    node, pair = _molecule_features(coords, zs)
    wn, wp = torch.randn_like(node), torch.randn_like(pair)

    g_auto, = torch.autograd.grad(projected_loss(coords, zs, wn, wp), coords)

    base = coords.detach()
    g_fd = torch.zeros_like(g_auto)
    for a in range(base.shape[0]):
        for k in range(3):
            for sign in (+1, -1):
                shifted = base.clone()
                shifted[a, k] += sign * FD_STEP
                with torch.no_grad():
                    g_fd[a, k] += (sign
                                   * projected_loss(shifted, zs, wn, wp)
                                   / (2 * FD_STEP))

    err = (g_auto - g_fd).abs().max().item()
    scale = max(g_fd.abs().max().item(), 1.0)
    min_gap, homo_lumo = occupied_gap(base, zs)
    ok = err < FD_TOL * scale
    print(f"{name:<10s} min occ gap={min_gap:.2e}  HOMO-LUMO={homo_lumo:.3f}  "
          f"max|auto-FD|={err:.3e}  (scale {scale:.3e})  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def main():
    print("TB feature gradient vs central finite differences (float64)\n")
    ok = [check(n, zs, xyz) for n, (zs, xyz) in MOLECULES.items()]
    print("\nPASS" if all(ok) else "\nFAIL")
    return 0 if all(ok) else 1


def test_main():
    """pytest entry point; main() returns 0 on success, 1 on failure."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
