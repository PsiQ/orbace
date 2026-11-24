# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Differentiable one-electron Gaussian integrals in PyTorch.

Implements overlap, dipole (first moment), and second-moment integrals over
contracted Cartesian Gaussian-type orbitals (GTOs), with the atom positions as
*differentiable* inputs. This is the geometry-aware core that lets OrbACE build
orbital features inside the autograd graph, so that forces (-dE/dR) are the
exact gradient of the evaluated energy.

Method
------
Overlap primitives use the McMurchie-Davidson Hermite expansion: the product of
two Cartesian Gaussians is expanded in Hermite Gaussians with coefficients
``E(i, j, t)`` obtained by a stable recursion. Only ``t = 0`` is needed for the
overlap. Higher moments are obtained without any new machinery by noting

    (x - B_x) * G_b(l) = G_b(l + 1)

so that, about the origin,

    <a| x   |b(l)> = <a|b(l+1)> + B_x <a|b(l)>
    <a| x^2 |b(l)> = <a|b(l+2)> + 2 B_x <a|b(l+1)> + B_x^2 <a|b(l)>

i.e. dipole and second-moment integrals are linear combinations of *overlap*
integrals with the ket angular momentum incremented. This keeps the whole
engine resting on a single, easily validated primitive (overlap).

Conventions
-----------
* Lengths in Bohr (atomic units), consistent with PySCF integral routines.
* Cartesian GTO ordering for a shell of angular momentum L follows the order
  produced by :func:`cart_components` (matches PySCF ``mol.cart = True``:
  for p -> (x, y, z); for d -> (xx, xy, xz, yy, yz, zz)).
* Each contracted AO is normalized to unit self-overlap, matching PySCF's
  normalized AO basis (diagonal of the overlap matrix is 1).
"""

from functools import lru_cache
from math import pi
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "GaussianBasis",
    "cart_components",
    "overlap_matrix",
    "moment_matrices",
]


def cart_components(L: int) -> List[Tuple[int, int, int]]:
    """Cartesian components (lx, ly, lz) for angular momentum ``L``.

    Order matches PySCF cartesian convention: lexicographic with x outermost,
    e.g. d -> (2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2).
    """
    comps = []
    for lx in range(L, -1, -1):
        for ly in range(L - lx, -1, -1):
            lz = L - lx - ly
            comps.append((lx, ly, lz))
    return comps


def _double_factorial(n: int) -> float:
    if n <= 0:
        return 1.0
    out = 1.0
    while n > 0:
        out *= n
        n -= 2
    return out


@lru_cache(maxsize=None)
def _prim_norm(L: int, lx: int, ly: int, lz: int, alpha: float) -> float:
    """Normalization of a primitive Cartesian GTO with exponent ``alpha``.

    N such that the self-overlap of x^lx y^ly z^lz exp(-alpha r^2) is 1.
    """
    num = (2.0 * alpha / pi) ** 1.5 * (4.0 * alpha) ** L
    den = (
        _double_factorial(2 * lx - 1)
        * _double_factorial(2 * ly - 1)
        * _double_factorial(2 * lz - 1)
    )
    return float(np.sqrt(num / den))


def _hermite_E(i: int, j: int, t: int, Qx: Tensor, a: float, b: float,
               memo: dict) -> Tensor:
    """McMurchie-Davidson Hermite expansion coefficient E(i, j, t).

    ``Qx`` is the (differentiable) Cartesian separation A_x - B_x. ``a``, ``b``
    are the primitive exponents. Returns a 0-d tensor (broadcasts with Qx).
    """
    if t < 0 or t > (i + j):
        return torch.zeros_like(Qx)
    key = (i, j, t)
    cached = memo.get(key)
    if cached is not None:
        return cached
    p = a + b
    q = a * b / p
    if i == 0 and j == 0 and t == 0:
        val = torch.exp(-q * Qx * Qx)
    elif j == 0:
        val = (
            (1.0 / (2.0 * p)) * _hermite_E(i - 1, j, t - 1, Qx, a, b, memo)
            - (q * Qx / a) * _hermite_E(i - 1, j, t, Qx, a, b, memo)
            + (t + 1) * _hermite_E(i - 1, j, t + 1, Qx, a, b, memo)
        )
    else:
        val = (
            (1.0 / (2.0 * p)) * _hermite_E(i, j - 1, t - 1, Qx, a, b, memo)
            + (q * Qx / b) * _hermite_E(i, j - 1, t, Qx, a, b, memo)
            + (t + 1) * _hermite_E(i, j - 1, t + 1, Qx, a, b, memo)
        )
    memo[key] = val
    return val


def _prim_overlap_1d(i: int, j: int, Qx: Tensor, a: float, b: float) -> Tensor:
    """1-D primitive overlap factor (without the (pi/p)^{1/2} prefactor's
    cross-dim grouping): returns E(i, j, 0)."""
    return _hermite_E(i, j, 0, Qx, a, b, {})


def _prim_overlap_3d(
    la: Tuple[int, int, int],
    lb: Tuple[int, int, int],
    a: float,
    b: float,
    Q: Tensor,
) -> Tensor:
    """Overlap of two primitive Cartesian Gaussians (un-normalized).

    ``Q = A - B`` is a length-3 differentiable tensor.
    """
    p = a + b
    pre = (pi / p) ** 1.5
    ex = _prim_overlap_1d(la[0], lb[0], Q[0], a, b)
    ey = _prim_overlap_1d(la[1], lb[1], Q[1], a, b)
    ez = _prim_overlap_1d(la[2], lb[2], Q[2], a, b)
    return pre * ex * ey * ez


class GaussianBasis:
    """A contracted Cartesian GTO basis built from a PySCF basis spec.

    The basis layout (exponents, contraction coefficients, angular momentum,
    atom assignment) is fixed at construction from atomic numbers; only the
    atom *positions* are supplied (differentiably) at integral-evaluation time.
    """

    def __init__(self, atom_numbers: np.ndarray, basis: str = "sto-3g"):
        from pyscf import gto

        self.atom_numbers = np.asarray(atom_numbers, dtype=int)
        self.basis_name = basis

        # Per-AO-component metadata (one entry per Cartesian basis function):
        #   atom_idx, (lx,ly,lz), exponents[np], coeffs[np] (already including
        #   primitive normalization and the contraction normalization).
        self.ao_atom: List[int] = []
        self.ao_lmn: List[Tuple[int, int, int]] = []
        self.ao_exps: List[np.ndarray] = []
        self.ao_coeffs: List[np.ndarray] = []
        self.ao_L: List[int] = []

        for atom_idx, Z in enumerate(self.atom_numbers):
            sym = gto.charge_symbol(int(Z)) if hasattr(gto, "charge_symbol") else _Z_to_sym(int(Z))
            b = gto.basis.load(basis, sym)
            for shell in b:
                L = shell[0]
                prims = np.array(shell[1:], dtype=float)  # rows: [exp, c1, c2,...]
                exps = prims[:, 0]
                # Use the first contraction column (minimal bases have one).
                coeffs_raw = prims[:, 1]
                for (lx, ly, lz) in cart_components(L):
                    pnorm = np.array(
                        [_prim_norm(L, lx, ly, lz, float(e)) for e in exps]
                    )
                    c = coeffs_raw * pnorm
                    self.ao_atom.append(atom_idx)
                    self.ao_lmn.append((lx, ly, lz))
                    self.ao_exps.append(exps)
                    self.ao_coeffs.append(c)
                    self.ao_L.append(L)

        self.n_ao = len(self.ao_atom)
        self._self_norm = self._compute_self_norms()
        self._build_primitive_table()

    def _build_primitive_table(self) -> None:
        """Flatten contracted AOs into a primitive table for vectorized eval.

        Each primitive m carries: host atom, Cartesian powers (lx,ly,lz),
        exponent, and contraction coefficient (including primitive + contracted
        normalization). ``prim_to_ao`` maps primitives to their contracted AO so
        that ``S_ao = A^T (S_prim) A`` with A the one-hot primitive->AO map.
        """
        atom_idx, lx, ly, lz, exps, coefs, ao_of = [], [], [], [], [], [], []
        for k in range(self.n_ao):
            l = self.ao_lmn[k]
            cN = self.ao_coeffs[k] * self._self_norm[k]
            for e, c in zip(self.ao_exps[k], cN):
                atom_idx.append(self.ao_atom[k])
                lx.append(l[0]); ly.append(l[1]); lz.append(l[2])
                exps.append(float(e)); coefs.append(float(c)); ao_of.append(k)
        self.prim_atom = np.asarray(atom_idx, dtype=np.int64)
        self.prim_l = np.stack([lx, ly, lz], axis=1).astype(np.int64)  # (M,3)
        self.prim_exp = np.asarray(exps, dtype=np.float64)
        self.prim_coef = np.asarray(coefs, dtype=np.float64)
        self.prim_to_ao = np.asarray(ao_of, dtype=np.int64)
        self.n_prim = len(exps)
        self.Imax = int(self.prim_l.max()) if self.n_prim else 0
        self.Jmax = self.Imax + 2  # room for second-moment ket shift (+2)

    def _compute_self_norms(self) -> np.ndarray:
        """Normalize each contracted AO to unit self-overlap (PySCF convention)."""
        norms = np.ones(self.n_ao)
        zeroQ = torch.zeros(3, dtype=torch.float64)
        for k in range(self.n_ao):
            lmn = self.ao_lmn[k]
            exps = self.ao_exps[k]
            coeffs = self.ao_coeffs[k]
            s = 0.0
            for ai, ca in zip(exps, coeffs):
                for bi, cb in zip(exps, coeffs):
                    s += float(ca * cb * _prim_overlap_3d(lmn, lmn, float(ai), float(bi), zeroQ))
            norms[k] = 1.0 / np.sqrt(s)
        return norms

    def _overlap_block(
        self,
        positions: Tensor,
        lmn_shift_b: Tuple[int, int, int] = (0, 0, 0),
    ) -> Tensor:
        """Assemble the (n_ao, n_ao) overlap matrix, optionally with the ket's
        Cartesian angular momentum shifted by ``lmn_shift_b`` (used to build
        moment integrals). Differentiable w.r.t. ``positions``.
        """
        n = self.n_ao
        dtype = positions.dtype
        S = positions.new_zeros((n, n))
        rows = []
        for i in range(n):
            ai_idx = self.ao_atom[i]
            la = self.ao_lmn[i]
            exps_a = self.ao_exps[i]
            coeffs_a = self.ao_coeffs[i] * self._self_norm[i]
            cols = []
            for j in range(n):
                bi_idx = self.ao_atom[j]
                lb0 = self.ao_lmn[j]
                lb = (lb0[0] + lmn_shift_b[0], lb0[1] + lmn_shift_b[1],
                      lb0[2] + lmn_shift_b[2])
                exps_b = self.ao_exps[j]
                coeffs_b = self.ao_coeffs[j] * self._self_norm[j]
                Q = positions[ai_idx] - positions[bi_idx]
                val = positions.new_zeros(())
                for ca, ea in zip(coeffs_a, exps_a):
                    for cb, eb in zip(coeffs_b, exps_b):
                        val = val + (ca * cb) * _prim_overlap_3d(
                            la, lb, float(ea), float(eb), Q
                        )
                cols.append(val)
            rows.append(torch.stack(cols))
        S = torch.stack(rows)
        return S.to(dtype)

    # ── Vectorized path ──────────────────────────────────────────────────
    def _E0_table(self, Q: Tensor, alpha: Tensor) -> Tensor:
        """Batched Hermite E(i, j, 0) for all primitive pairs and dims.

        Q:     (M, M, 3) differentiable Cartesian separations A - B (Bohr).
        alpha: (M,) primitive exponents.
        Returns E0 of shape (Imax+1, Jmax+1, M, M, 3).
        """
        a = alpha.view(-1, 1, 1)      # bra exponent  (M,1,1)
        b = alpha.view(1, -1, 1)      # ket exponent  (1,M,1)
        p = a + b                     # (M,M,1)
        q = a * b / p
        inv2p = 1.0 / (2.0 * p)
        qQa = q * Q / a
        qQb = q * Q / b
        Imax, Jmax = self.Imax, self.Jmax
        memo = {}

        def E(i, j, t):
            if t < 0 or t > i + j:
                return torch.zeros_like(Q)
            key = (i, j, t)
            v = memo.get(key)
            if v is not None:
                return v
            if i == 0 and j == 0 and t == 0:
                v = torch.exp(-q * Q * Q)
            elif j == 0:
                v = (inv2p * E(i - 1, 0, t - 1)
                     - qQa * E(i - 1, 0, t)
                     + (t + 1) * E(i - 1, 0, t + 1))
            else:
                v = (inv2p * E(i, j - 1, t - 1)
                     + qQb * E(i, j - 1, t)
                     + (t + 1) * E(i, j - 1, t + 1))
            memo[key] = v
            return v

        rows = []
        for i in range(Imax + 1):
            cols = [E(i, j, 0) for j in range(Jmax + 1)]
            rows.append(torch.stack(cols, dim=0))
        return torch.stack(rows, dim=0)  # (Imax+1, Jmax+1, M, M, 3)

    def _overlap_ao(self, positions: Tensor, E0: Tensor, alpha: Tensor,
                    Amap: Tensor, shift=(0, 0, 0)) -> Tensor:
        """AO overlap matrix from a prebuilt E0 table, with optional ket shift."""
        M = self.n_prim
        prim_l = torch.as_tensor(self.prim_l, device=positions.device)  # (M,3)
        Pg, Qg = torch.meshgrid(torch.arange(M, device=positions.device),
                                torch.arange(M, device=positions.device),
                                indexing="ij")
        a = alpha.view(-1, 1)
        bb = alpha.view(1, -1)
        p = a + bb
        pre = (pi / p) ** 1.5                                   # (M,M)
        val = pre
        for d in range(3):
            ii = prim_l[:, d][Pg]                               # (M,M)
            jj = prim_l[:, d][Qg] + shift[d]
            val = val * E0[ii, jj, Pg, Qg, d]
        coef = torch.as_tensor(self.prim_coef, dtype=positions.dtype,
                               device=positions.device)
        Sprim = val * coef.view(-1, 1) * coef.view(1, -1)       # (M,M)
        return Amap.T @ Sprim @ Amap                            # (n_ao, n_ao)

    def _prep_fast(self, positions: Tensor):
        prim_atom = torch.as_tensor(self.prim_atom, device=positions.device)
        centers = positions[prim_atom]                          # (M,3)
        Q = centers.unsqueeze(1) - centers.unsqueeze(0)         # (M,M,3)
        alpha = torch.as_tensor(self.prim_exp, dtype=positions.dtype,
                                device=positions.device)
        E0 = self._E0_table(Q, alpha)
        Amap = positions.new_zeros((self.n_prim, self.n_ao))
        Amap[torch.arange(self.n_prim, device=positions.device),
             torch.as_tensor(self.prim_to_ao, device=positions.device)] = 1.0
        return E0, alpha, Amap

    def overlap(self, positions: Tensor) -> Tensor:
        """Overlap matrix S (n_ao, n_ao). Positions in Bohr (n_atom, 3)."""
        E0, alpha, Amap = self._prep_fast(positions)
        return self._overlap_ao(positions, E0, alpha, Amap)

    def moments(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (dipole, r2) integral matrices about the origin (vectorized).

        dipole: (3, n_ao, n_ao) with <i|x|j>, <i|y|j>, <i|z|j>.
        r2:     (n_ao, n_ao) with <i| x^2 + y^2 + z^2 |j>. Positions in Bohr.
        """
        n = self.n_ao
        E0, alpha, Amap = self._prep_fast(positions)
        S = self._overlap_ao(positions, E0, alpha, Amap)
        ket_center = positions[[self.ao_atom[j] for j in range(n)]]  # (n,3)

        e1 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        e2 = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
        S1 = [self._overlap_ao(positions, E0, alpha, Amap, s) for s in e1]
        dipole = []
        for k in range(3):
            Bk = ket_center[:, k].unsqueeze(0)
            dipole.append(S1[k] + Bk * S)
        dipole = torch.stack(dipole, dim=0)

        r2 = positions.new_zeros((n, n))
        for k in range(3):
            S2 = self._overlap_ao(positions, E0, alpha, Amap, e2[k])
            Bk = ket_center[:, k].unsqueeze(0)
            r2 = r2 + S2 + 2.0 * Bk * S1[k] + (Bk * Bk) * S
        return dipole, r2

    def moments_full(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (dipole, second_moment) integral matrices about the origin.

        dipole:        (3, n_ao, n_ao)      <i| r_a |j>
        second_moment: (3, 3, n_ao, n_ao)   <i| r_a r_b |j>  (symmetric in a,b)
        Positions in Bohr. Used to build the l=1 polarization and l=2 quadrupole
        node features.
        """
        n = self.n_ao
        E0, alpha, Amap = self._prep_fast(positions)
        S = self._overlap_ao(positions, E0, alpha, Amap)
        ket_center = positions[[self.ao_atom[j] for j in range(n)]]  # (n,3)
        B = [ket_center[:, k].unsqueeze(0) for k in range(3)]        # 1xn each

        e = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        S1 = [self._overlap_ao(positions, E0, alpha, Amap, e[k]) for k in range(3)]
        dipole = torch.stack([S1[k] + B[k] * S for k in range(3)], dim=0)

        second = positions.new_zeros((3, 3, n, n))
        for a in range(3):
            # diagonal a==a : shift 2 e_a
            sh = [0, 0, 0]; sh[a] = 2
            S2 = self._overlap_ao(positions, E0, alpha, Amap, tuple(sh))
            second[a, a] = S2 + 2.0 * B[a] * S1[a] + (B[a] * B[a]) * S
            for b in range(a + 1, 3):
                sh = [0, 0, 0]; sh[a] += 1; sh[b] += 1
                Sab = self._overlap_ao(positions, E0, alpha, Amap, tuple(sh))
                M = Sab + B[a] * S1[b] + B[b] * S1[a] + (B[a] * B[b]) * S
                second[a, b] = M
                second[b, a] = M
        return dipole, second

    def overlap_ref(self, positions: Tensor) -> Tensor:
        """Reference (slow, looped) overlap, kept for cross-validation."""
        return self._overlap_block(positions)


def _Z_to_sym(Z: int) -> str:
    from ase.data import chemical_symbols
    return chemical_symbols[Z]


def overlap_matrix(atom_numbers, positions_bohr: Tensor, basis: str = "sto-3g") -> Tensor:
    return GaussianBasis(atom_numbers, basis).overlap(positions_bohr)


def moment_matrices(atom_numbers, positions_bohr: Tensor, basis: str = "sto-3g"):
    gb = GaussianBasis(atom_numbers, basis)
    dipole, r2 = gb.moments(positions_bohr)
    return gb.overlap(positions_bohr), dipole, r2
