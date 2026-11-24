# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Differentiable tight-binding (extended-Hückel) features for a CHEAP,
CONSERVATIVE atom-centered OrbACE.

Motivation
----------
Gauge-invariant features can be built from an HF density matrix ``P`` (+
``W, S, H, F``), but computing them at prep time leaves them DETACHED w.r.t.
geometry, which makes the resulting forces non-conservative.

This module keeps the same feature definitions (12 node scalars + 6 edge
features, same column order) but computes ``P/W/S/H/F`` from a one-shot
**extended-Hückel** (non-self-consistent) tight-binding model implemented purely
in torch, as a differentiable function of the atomic coordinates:

    S(R)  : minimal-basis Gaussian overlaps (analytic, s/p only)
    H      : extended-Hückel (Wolfsberg-Helmholz)  H_uv = K S_uv (H_uu+H_vv)/2
    solve  : one generalized eigenproblem  H C = S C e   (symmetric orthog.)
    P, W   : closed-shell density / energy-weighted density from the lowest MOs
    F      : = H   (Hückel: no self-consistent Coulomb term)

Because there is NO SCF loop, forces are the exact gradient of the energy via
plain ``autograd`` through ``torch.linalg.eigh`` -- cheap and conservative by
construction. This is the atom model's only feature path.

Everything is in atomic units internally (Bohr, Hartree); positions come in
Angstrom to match the rest of the pipeline.

Supported elements: H, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl (main-group
s/p). Extendable via ``_ELEMENTS`` (add STO-3G valence exps + EHT VSIP).
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

HARTREE_TO_EV = 27.211386245988
ANG_TO_BOHR = 1.0 / 0.529177210903
N_L_CHANNELS = 5  # s, p, d, f, g  (only s,p populated here)

# Feature layout produced by tb_features(): widths size the model's embeddings.
N_NODE_SCALARS = 12
N_EDGE_FEATURES = 6  # bond, P_fro, F_fro, S_fro, H_fro, PF_coupling

# Canonical order of the 6 atom-pair (edge) feature columns.
PAIR_KEYS_DEFAULT = ("pair_bond", "pair_P_fro", "pair_F_fro",
                     "pair_S_fro", "pair_H_fro", "pair_PF_coupling")

# STO-3G primitive contraction coefficients (valence-only minimal basis).
# Coefficients assume UNNORMALIZED primitives; we normalize below. All period-2
# elements share the 2s/2p coeffs; all period-3 elements share the 3s/3p coeffs
# (only the exponents change per element). Values taken from STO-3G (EMSL/PySCF).
_C_1S = (0.15432897, 0.53532814, 0.44463454)
_C_2S = (-0.09996723, 0.39951283, 0.70011547)
_C_2P = (0.15591627, 0.60768372, 0.39195739)
_C_3S = (-0.21962037, 0.22559543, 0.90039843)
_C_3P = (0.01058760, 0.59516701, 0.46200101)


def _sp(valence_Z, exps, vsip_s, vsip_p, cs, cp):
    """Build a main-group entry with a valence s-shell and p-shell that share
    one set of STO-3G exponents (the standard sp contraction)."""
    return {"valence_Z": float(valence_Z),
            "shells": [(0, exps, cs, vsip_s), (1, exps, cp, vsip_p)]}


# Per element: list of shells (l, exponents, coeffs, VSIP[eV]).
#   l=0 -> one s AO; l=1 -> three p AOs (px,py,pz) sharing the radial part.
# VSIP = extended-Hückel on-site (valence-state ionization) energies, eV
# (Hoffmann/Alvarez parameters). Exponents are the STO-3G valence shells.
_ELEMENTS: Dict[int, Dict] = {
    1: {  # H : 1s
        "valence_Z": 1.0,
        "shells": [(0, (3.4252509, 0.6239137, 0.1688554), _C_1S, -13.6)],
    },
    # ---- period 2 : 2s 2p (share _C_2S / _C_2P) ----
    5: _sp(3.0, (2.2369561, 0.5198205, 0.1690618), -15.2, -8.5, _C_2S, _C_2P),   # B
    6: _sp(4.0, (2.9412494, 0.6834831, 0.2222899), -21.4, -11.4, _C_2S, _C_2P),  # C
    7: _sp(5.0, (3.7804559, 0.8784966, 0.2857144), -26.0, -13.4, _C_2S, _C_2P),  # N
    8: _sp(6.0, (5.0331513, 1.1695961, 0.3803890), -32.3, -14.8, _C_2S, _C_2P),  # O
    9: _sp(7.0, (6.4648032, 1.5022812, 0.4885885), -40.0, -18.1, _C_2S, _C_2P),  # F
    # ---- period 3 : 3s 3p (share _C_3S / _C_3P) ----
    11: _sp(1.0, (1.4787406, 0.4125649, 0.1614751), -5.1, -3.0, _C_3S, _C_3P),   # Na
    12: _sp(2.0, (1.3954480, 0.3893260, 0.1523800), -9.0, -4.5, _C_3S, _C_3P),   # Mg
    13: _sp(3.0, (1.3954483, 0.3893265, 0.1523798), -12.3, -6.5, _C_3S, _C_3P),  # Al
    14: _sp(4.0, (1.4787406, 0.4125649, 0.1614751), -17.3, -9.2, _C_3S, _C_3P),  # Si
    15: _sp(5.0, (1.7431032, 0.4863214, 0.1903429), -18.6, -14.0, _C_3S, _C_3P), # P
    16: _sp(6.0, (2.0291943, 0.5661401, 0.2215834), -20.0, -13.3, _C_3S, _C_3P), # S
    17: _sp(7.0, (2.1293865, 0.5940934, 0.2325241), -26.3, -14.2, _C_3S, _C_3P), # Cl
}

_K_EHT = 1.75  # Wolfsberg-Helmholz constant

# Cartesian angular momentum vectors for the AOs of each l.
#   s -> (0,0,0);  p -> px,py,pz
_L_VECS = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
}


def _prim_norm(alpha, ltot):
    """Primitive Cartesian-Gaussian normalization for |L|=ltot (s or p)."""
    base = (2.0 * alpha / np.pi) ** 0.75
    if ltot == 0:
        return base
    # |L|=1: (4 alpha)^{1/2} / sqrt((2*1-1)!!)=(4 alpha)^{1/2}
    return base * (4.0 * alpha) ** 0.5


# ---- per-element AO tables (static; built once as numpy, cached) -----------
_ELEMENT_AO: Dict[int, Dict] = {}


def _element_ao(Z: int) -> Dict:
    Z = int(Z)
    if Z in _ELEMENT_AO:
        return _ELEMENT_AO[Z]
    if Z not in _ELEMENTS:
        raise KeyError(
            f"tb_features: element Z={Z} is not parameterized. Supported: "
            f"{sorted(_ELEMENTS)}. Add STO-3G valence exps + EHT VSIP to _ELEMENTS.")
    exps, coefs, L, hdiag, ao_l = [], [], [], [], []
    for (l, e3, c3, vsip) in _ELEMENTS[Z]["shells"]:
        for Lvec in _L_VECS[l]:
            ltot = sum(Lvec)
            exps.append(list(e3))
            coefs.append([c * _prim_norm(al, ltot) for c, al in zip(c3, e3)])
            L.append(list(Lvec))
            hdiag.append(vsip / HARTREE_TO_EV)
            ao_l.append(l)
    d = dict(
        exps=np.asarray(exps, np.float64), coefs=np.asarray(coefs, np.float64),
        L=np.asarray(L, np.float64), hdiag=np.asarray(hdiag, np.float64),
        ao_l=np.asarray(ao_l, np.int64), n_ao=len(ao_l),
        valence_Z=float(_ELEMENTS[Z]["valence_Z"]))
    _ELEMENT_AO[Z] = d
    return d


def tb_num_ao(atom_numbers) -> int:
    """AOs a molecule occupies in the TB valence basis (1 for H, 4 for C/N/O...).

    Public because it is the cost driver, not just bookkeeping: the eigensolves
    are O(nao^3) and the batched path pads every molecule up to the batch's
    largest nao, so grouping molecules by this number is what keeps that padding
    from dominating (see BucketedBatchSampler in data/invariant_graph.py).
    """
    return sum(_element_ao(int(z))["n_ao"] for z in atom_numbers)


# Every quantity this module differentiates through -- S^{-1/2}, the density
# matrix P, the energy-weighted density matrix W -- is a *spectral function* of a
# symmetric matrix, i.e. of the form Y = V f(w) V^T. That structure is what makes
# an exact backward possible even when eigenvalues are degenerate.
#
# Differentiating the eigenvectors themselves would introduce a bare
# 1/(w_i - w_j), which diverges at the degeneracies a symmetric tight-binding
# Hamiltonian is full of. The usual workaround is to broaden it to
# d/(d^2 + eps). That is bounded, but it is also WRONG: for W the correctly
# symmetrized backward retains a factor d^2/(d^2 + eps) whose true value is 1,
# so broadening drives a finite term to zero whenever two occupied levels touch.
# Every symmetric molecule (CH4, NH3, benzene, ...) hits this, giving forces that
# are off by tens of percent and that shift with the LAPACK build, since the
# eigenvector basis inside a degenerate block is arbitrary.
#
# The Daleckii-Krein form avoids the problem entirely: dY = V [L o (V^T dA V)] V^T
# with the divided difference L_ij = (f(w_i)-f(w_j))/(w_i-w_j). Here the numerator
# vanishes together with the denominator, leaving a removable singularity whose
# limit is simply f'(w). No broadening, no eps, exact at degeneracies, and
# manifestly invariant to the eigenvector basis.

# Eigenvalue separation below which the divided difference is evaluated as its
# limit rather than as a ratio. Above it the plain ratio is accurate; below it
# catastrophic cancellation in f(w_i)-f(w_j) would dominate.
_DEGEN_TOL = 1e-8


def _loewner(w, fw, fpw, tol=_DEGEN_TOL):
    """Divided-difference matrix ``L_ij = (f(w_i)-f(w_j))/(w_i-w_j)``.

    ``fpw`` supplies f'(w) for the degenerate limit, which also fills the
    diagonal. The limit is taken at the pair midpoint, so it is second-order
    accurate in the gap and exact for the constant f' this module needs.

    The limit is only taken where the singularity is REMOVABLE, which requires
    the numerator to vanish with the denominator. A small denominator alone is
    not enough. Two levels straddling the Fermi level have f values that differ
    by a finite amount however close their eigenvalues are -- for the density
    matrix, f_P = 2*occ gives f(w_i)-f(w_j) = 2 across an occupied/virtual pair
    -- so L_ij = 2/(w_i-w_j) genuinely diverges there and taking the midpoint
    limit would silently return f'(w) = 0 in place of an unbounded entry.
    Testing ``df`` as well as ``dw`` keeps that case on the exact ratio, which
    is well conditioned precisely because its numerator does not cancel. The
    threshold on ``df`` scales with the local slope, since a removable pair
    has |df| <= max(|f'(w_i)|, |f'(w_j)|) * |dw| to first order. That covers
    the density matrices, where |f'| <= 2, and also S^{-1/2}, whose slope
    -s^{-3/2}/2 is not bounded by 2 for small overlap eigenvalues.
    """
    dw = w.unsqueeze(-1) - w.unsqueeze(-2)
    df = fw.unsqueeze(-1) - fw.unsqueeze(-2)
    slope = torch.maximum(fpw.abs().unsqueeze(-1), fpw.abs().unsqueeze(-2))
    removable = (dw.abs() < tol) & (df.abs() < 4.0 * tol * torch.clamp(slope, min=1.0))
    ratio = df / torch.where(removable, torch.ones_like(dw), dw)
    limit = 0.5 * (fpw.unsqueeze(-1) + fpw.unsqueeze(-2))
    return torch.where(removable, limit, ratio)


def _sym(A):
    return 0.5 * (A + A.transpose(-1, -2))


class _InvSqrtPSD(torch.autograd.Function):
    """``S -> S^{-1/2}`` (symmetric orthogonalizer) with an exact backward."""

    @staticmethod
    def forward(ctx, S, floor):
        w, V = torch.linalg.eigh(S)
        wc = w.clamp(min=floor)
        fw = wc.rsqrt()
        # A clamped eigenvalue is pinned, so it carries no sensitivity. Reaching
        # the floor at all means S is numerically singular (near-linearly
        # dependent basis), which these minimal bases do not do.
        fpw = torch.where(w > floor, -0.5 * fw / wc, torch.zeros_like(fw))
        ctx.save_for_backward(wc, V, fw, fpw)
        return V @ torch.diag_embed(fw) @ V.transpose(-1, -2)

    @staticmethod
    def backward(ctx, gX):
        wc, V, fw, fpw = ctx.saved_tensors
        L = _loewner(wc, fw, fpw)
        Vt = V.transpose(-1, -2)
        return _sym(V @ (L * (Vt @ gX @ V)) @ Vt), None


class _DensityMatrices(torch.autograd.Function):
    """Orthogonal-basis Hamiltonian -> (P, W), both with exact backwards.

    ``P = 2 V occ V^T`` and ``W = 2 V occ e V^T`` in the orthogonal basis. Their
    divided differences behave very differently at a degeneracy: P's numerator
    ``2(occ_i - occ_j)`` is identically zero there (degenerate levels share an
    occupation), while W's tends to ``2 occ``. Dropping that second term is
    exactly the error a broadened eigenvector backward makes.
    """

    @staticmethod
    def forward(ctx, Hp, occ):
        e, V = torch.linalg.eigh(Hp)
        fP = 2.0 * occ
        fW = 2.0 * occ * e
        Vt = V.transpose(-1, -2)
        P = V @ torch.diag_embed(fP) @ Vt
        W = V @ torch.diag_embed(fW) @ Vt
        ctx.save_for_backward(e, V, occ, fP, fW)
        return P, W

    @staticmethod
    def backward(ctx, gP, gW):
        e, V, occ, fP, fW = ctx.saved_tensors
        # d(2*occ)/de = 0 and d(2*occ*e)/de = 2*occ, both valid away from the
        # Fermi level. A pair straddling it is not covered by these
        # derivatives, because occupation is attached to the eigenvalue INDEX
        # rather than to its value, so f is a step there and P is discontinuous
        # in the limit of a closing gap. _loewner keeps such a pair on the
        # exact ratio rather than the degenerate limit, so the gradient it
        # returns is the correct (large) one rather than a silent zero; it is
        # still the derivative of a function that is about to become
        # discontinuous, which is a statement about the frontier gap of the
        # system, not about the differentiation.
        LP = _loewner(e, fP, torch.zeros_like(e))
        LW = _loewner(e, fW, 2.0 * occ)
        Vt = V.transpose(-1, -2)
        M = LP * (Vt @ gP @ V) + LW * (Vt @ gW @ V)
        return _sym(V @ M @ Vt), None


def _density_matrices(Hp, occ):
    return _DensityMatrices.apply(Hp, occ)


def _build_ao_table(atom_numbers: List[int]):
    """Static (geometry-independent) AO bookkeeping for one molecule.

    Returns:
      ao_atom  (nao,)   atom index of each AO
      ao_l     (nao,)   angular momentum l of each AO
      ao_L     (nao,3)  Cartesian angular vector
      exps     (nao,3)  primitive exponents (padded to 3)
      coefs    (nao,3)  contraction coeffs * primitive-norm
      hdiag    (nao,)   extended-Hückel on-site energies [Hartree]
      n_val_elec (float) total valence electrons
    """
    els = [_element_ao(z) for z in atom_numbers]
    exps = np.concatenate([e["exps"] for e in els], 0)
    coefs = np.concatenate([e["coefs"] for e in els], 0)
    ao_L = np.concatenate([e["L"] for e in els], 0)
    hdiag = np.concatenate([e["hdiag"] for e in els], 0)
    ao_l = np.concatenate([e["ao_l"] for e in els], 0)
    ao_atom = np.concatenate([np.full(e["n_ao"], a, np.int64)
                              for a, e in enumerate(els)], 0)
    n_val = float(sum(e["valence_Z"] for e in els))
    return (ao_atom, ao_l, ao_L, exps, coefs, hdiag, n_val)


# AO bookkeeping depends only on the element sequence, so for a fixed dataset it
# is identical on every epoch. Cache it per composition: ANI-1x has a few
# thousand distinct formulas against 442k frames, so this converges to a hit
# almost immediately and removes the table rebuild from the training step.
_AO_TABLE_CACHE: Dict[Tuple[int, ...], Tuple] = {}


def _closed_shell_n_occ(n_val: float, charge: int, z_key=None) -> int:
    """Doubly-occupied orbital count for a molecule of the given total charge.

    Only closed shells are representable here: the Hückel solve produces one
    set of spatial orbitals and fills each with a pair, so an odd electron
    count has no valid occupation. Rejecting it is deliberate -- flooring the
    division would quietly drop an electron and return features for the wrong
    system, which is far worse than refusing to run.
    """
    n_elec = int(round(n_val)) - int(charge)
    where = f" for Z={tuple(z_key)}" if z_key is not None else ""
    if n_elec < 0:
        raise ValueError(
            f"charge {charge:+d} strips all {int(round(n_val))} valence "
            f"electrons{where}")
    if n_elec % 2:
        raise ValueError(
            f"{n_elec} valence electrons{where} at charge {charge:+d} is an "
            "open shell; the conservative TB featurizer is closed-shell only")
    return n_elec // 2


def _ao_table_cached(z_key: Tuple[int, ...]) -> Tuple:
    t = _AO_TABLE_CACHE.get(z_key)
    if t is None:
        t = _build_ao_table(list(z_key))
        _AO_TABLE_CACHE[z_key] = t
    return t


def _overlap_matrix(coords_bohr, ao_L, exps, coefs) -> torch.Tensor:
    """Contracted-Gaussian overlap S (nao,nao), differentiable in coords.

    s/p only: the 1-D primitive overlap uses the closed form
      S1d(0,0)=1, S1d(1,0)=PA, S1d(0,1)=PB, S1d(1,1)=PA*PB + 1/(2p).
    """
    nao = coords_bohr.shape[0]
    dev, dt = coords_bohr.device, coords_bohr.dtype
    a = exps[:, None, :, None]                      # (nao,1,3,1)
    b = exps[None, :, None, :]                       # (1,nao,1,3)
    p = a + b                                        # (nao,nao,3,3)
    A = coords_bohr[:, None, None, None, :]          # (nao,1,1,1,3)
    B = coords_bohr[None, :, None, None, :]          # (1,nao,1,1,3)
    AB = A - B                                       # (nao,nao,1,1,3)
    ab2 = (AB ** 2).sum(-1)                           # (nao,nao,1,1)
    P = (a[..., None] * A + b[..., None] * B) / p[..., None]   # (nao,nao,3,3,3)
    PA = P - A
    PB = P - B
    mu = a * b / p                                   # (nao,nao,3,3)
    pref = (torch.pi / p) ** 1.5 * torch.exp(-mu * ab2)        # (nao,nao,3,3)

    Lm = ao_L[:, None, None, None, :].to(dt)         # (nao,1,1,1,3)
    Ln = ao_L[None, :, None, None, :].to(dt)         # (1,nao,1,1,3)
    inv2p = (1.0 / (2.0 * p))[..., None]             # (nao,nao,3,3,1)
    # elementwise S1d for L in {0,1}
    s1d = ((1 - Lm) * (1 - Ln)
           + Lm * (1 - Ln) * PA
           + (1 - Lm) * Ln * PB
           + Lm * Ln * (PA * PB + inv2p))             # (nao,nao,3,3,3)
    sdim = s1d.prod(dim=-1)                           # (nao,nao,3,3)
    prim = pref * sdim                                # (nao,nao,3,3)
    ci = coefs[:, None, :, None]
    cj = coefs[None, :, None, :]
    S = (ci * cj * prim).sum(dim=(-1, -2))            # (nao,nao)
    # normalize contracted functions to unit self-overlap
    d = torch.rsqrt(torch.clamp(torch.diagonal(S), min=1e-12))
    S = d[:, None] * S * d[None, :]
    return S


_S_FLOOR = 1e-8


def _sym_orthogonalize(S: torch.Tensor) -> torch.Tensor:
    """Symmetric (Loewdin) orthogonalizer S^{-1/2}; batched or not."""
    return _InvSqrtPSD.apply(S, _S_FLOOR)


def _molecule_matrices(coords_ang, atom_numbers, charge: int = 0):
    """Return P, W, S, H (all nao×nao, differentiable) + ao maps for one mol.

    The electronic solve runs in float64 regardless of the input dtype. The
    divided-difference backwards are finite at exact degeneracy in any precision,
    but float32 rounding splits exactly degenerate levels by more than
    ``_DEGEN_TOL``, so the pair is evaluated as a ratio of rounding errors (5-12%
    gradient error at the symmetric CH4, NH3 and C6H6 geometries). float64 (on
    tiny matrices) is cheap and keeps such splits below the tolerance; gradients
    cast back to the input dtype.
    """
    ao_atom, ao_l, ao_L, exps, coefs, hdiag, n_val = _build_ao_table(atom_numbers)
    dev = coords_ang.device
    dt = torch.float64                               # solve in double
    ao_atom_t = torch.as_tensor(ao_atom, dtype=torch.long, device=dev)
    ao_l_t = torch.as_tensor(ao_l, dtype=torch.long, device=dev)
    ao_L_t = torch.as_tensor(ao_L, dtype=dt, device=dev)
    exps_t = torch.as_tensor(exps, dtype=dt, device=dev)
    coefs_t = torch.as_tensor(coefs, dtype=dt, device=dev)
    hdiag_t = torch.as_tensor(hdiag, dtype=dt, device=dev)

    coords_bohr = coords_ang.to(dt) * ANG_TO_BOHR
    ao_coords = coords_bohr[ao_atom_t]               # (nao,3)

    S = _overlap_matrix(ao_coords, ao_L_t, exps_t, coefs_t)
    # extended-Hückel Hamiltonian
    hsum = 0.5 * (hdiag_t[:, None] + hdiag_t[None, :])
    H = _K_EHT * S * hsum
    H = H - torch.diag(torch.diagonal(H)) + torch.diag(hdiag_t)   # exact diag

    X = _sym_orthogonalize(S)
    Hp = X @ H @ X

    n_occ = _closed_shell_n_occ(n_val, charge, tuple(int(z) for z in atom_numbers))
    occ = (torch.arange(Hp.shape[-1], device=dev) < n_occ).to(dt)
    # P, W come back in the orthogonal basis; X is symmetric, so X (.) X maps
    # them to the AO basis.
    P_o, W_o = _density_matrices(Hp, occ)
    P = X @ P_o @ X
    W = X @ W_o @ X
    F = H
    return P, W, S, H, F, ao_atom_t, ao_l_t


def _molecule_features(coords_ang, atom_numbers, charge: int = 0):
    """12 node scalars (natm,12) and pair matrices (natm,natm,6) for one mol.

    Fully vectorized via the AO->atom indicator ``G`` (nao,natm): any per-atom
    block-sum of a matrix ``M`` is ``G.T @ M @ G`` (no Python atom loops), so the
    whole thing is a handful of BLAS matmuls -- cheap and autograd-friendly.
    """
    # electronic matrices are float64 (see _molecule_matrices); build features
    # in float64 too, then cast back to the model's dtype at the end.
    P, W, S, H, F, ao_atom, ao_l = _molecule_matrices(coords_ang, atom_numbers,
                                                      charge)
    natm = len(atom_numbers)
    dev, dt = coords_ang.device, P.dtype
    nao = P.shape[0]

    G = torch.zeros((nao, natm), dtype=dt, device=dev)
    G[torch.arange(nao, device=dev), ao_atom] = 1.0

    def blocksum(M):  # (natm,natm) sum of M over AO blocks of each atom pair
        return G.T @ M @ G

    def frob(M):      # (natm,natm) Frobenius norm of each atom-pair block
        return torch.sqrt(torch.clamp(blocksum(M * M), min=1e-18))

    PS = P @ S
    pop_ao = torch.diagonal(PS)
    pop_atom = G.T @ pop_ao
    valence_Z = torch.tensor(
        [_ELEMENTS[int(z)]["valence_Z"] for z in atom_numbers], dtype=dt, device=dev)

    PHt = blocksum(P * H.T)     # Tr(P_AB H_BA)
    PFt = blocksum(P * F.T)     # Tr(P_AB F_BA)

    node = torch.zeros((natm, 12), dtype=dt, device=dev)
    node[:, 0] = valence_Z - pop_atom               # Mulliken charge
    node[:, 1] = pop_atom
    for l in range(N_L_CHANNELS):
        maskl = (ao_l == l).to(dt)
        node[:, 2 + l] = G.T @ (pop_ao * maskl)     # per-l population
    node[:, 7] = torch.diagonal(PHt)                # Tr(P_AA H_AA)
    node[:, 8] = torch.diagonal(PFt)                # Tr(P_AA F_AA)
    node[:, 9] = G.T @ torch.diagonal(W)            # Tr(W_AA)
    node[:, 10] = torch.diagonal(frob(P))           # ||P_AA||_F
    node[:, 11] = pop_atom                          # valence pop (no core)

    pair = torch.zeros((natm, natm, 6), dtype=dt, device=dev)
    pair[:, :, 0] = blocksum(PS * PS.T)             # Mayer bond order
    pair[:, :, 1] = frob(P)
    pair[:, :, 2] = frob(F)
    pair[:, :, 3] = frob(S)
    pair[:, :, 4] = frob(H)
    pair[:, :, 5] = PFt                             # Tr(P_AB F_BA)
    out_dt = coords_ang.dtype
    return node.to(out_dt), pair.to(out_dt)


# ---------------------------------------------------------------------------
# Batched path
#
# The reference path above solves one molecule at a time, so a training step
# serialized 2*batch_size float64 eigendecompositions of ~40x40 matrices through
# a Python loop. Those are latency-bound, not compute-bound, which is why the
# conservative backend cost ~3.4x the detached one per epoch.
#
# Batching pads every molecule to the batch maximum and issues two batched
# eigh calls instead. Padding makes both matrices exactly block diagonal, with
# the padded block constant and its spectrum held clear of the real one:
#
#   S = [S_real    0    ]    H = [H_real     0     ]
#       [  0    s_pad*I ]        [  0    E_pad*I   ]
#
# so eigh returns the real eigenpairs unchanged, plus ``nao_pad`` padded levels
# that sort to the top and are never occupied. Features and gradients are
# therefore identical to the reference path to floating-point precision -- see
# tests/test_tb_batched.py, which pins both.
# ---------------------------------------------------------------------------

# Padding constants, chosen to keep the padded block's spectrum clear of the real
# one. The spectral backwards are basis-independent, so block mixing would no
# longer corrupt the gradient, but keeping the blocks separated still avoids
# manufacturing near-degeneracies that would push the divided differences onto
# their limit branch for no reason.
#
# A normalized minimal-basis overlap matrix has eigenvalues in roughly (0, 2)
# clustered near 1, so padding S with the identity would sit right on top of them.
_PAD_OVERLAP = 10.0
# Hartree, on the padded H diagonal. After orthogonalization the padded levels
# land at _PAD_HDIAG / _PAD_OVERLAP = 100 Ha, far above every real Hückel level
# (real VSIPs span roughly -1.5..-0.1 Ha) so they can never enter the occupied
# set, while keeping ``Hp`` well conditioned for float64 eigh.
_PAD_HDIAG = 1.0e3

_OVERLAP_COMPILED = None


def _maybe_compile_overlap():
    """Opt-in ``torch.compile`` of the overlap builder.

    The builder is ~30 elementwise kernels over (B,nao,nao,3,3,3) tensors. At
    batch 16 with nao~40 each one is far too small to saturate an H200, so the
    stage is launch-bound: the same step costs 17% more than the detached
    backend on CPU but 82% more on GPU. Fusing the chain is the direct fix.

    Kept opt-in (``ORBACE_TB_COMPILE=1``) because it must hold up under the
    double backward the force loss needs, and it falls back silently if the
    installed torch cannot compile it.
    """
    global _OVERLAP_COMPILED
    if _OVERLAP_COMPILED is not None:
        return _OVERLAP_COMPILED
    if os.environ.get("ORBACE_TB_COMPILE", "0") not in ("1", "true", "True"):
        _OVERLAP_COMPILED = _overlap_matrix_batched
        return _OVERLAP_COMPILED
    try:
        _OVERLAP_COMPILED = torch.compile(_overlap_matrix_batched, dynamic=True)
    except Exception:
        _OVERLAP_COMPILED = _overlap_matrix_batched
    return _OVERLAP_COMPILED


def _overlap_matrix_batched(coords_bohr, ao_L, exps, coefs) -> torch.Tensor:
    """Batched contracted-Gaussian overlap S (B,nao,nao).

    Identical closed form to ``_overlap_matrix``, with a leading batch axis.
    Padded AO slots carry ``coefs=0`` so they contribute exactly zero (and
    exactly zero gradient) while ``exps=1`` keeps every prefactor finite.
    """
    dt = coords_bohr.dtype
    a = exps[:, :, None, :, None]                      # (B,nao,1,3,1)
    b = exps[:, None, :, None, :]                      # (B,1,nao,1,3)
    p = a + b                                          # (B,nao,nao,3,3)
    A = coords_bohr[:, :, None, None, None, :]         # (B,nao,1,1,1,3)
    Bc = coords_bohr[:, None, :, None, None, :]        # (B,1,nao,1,1,3)
    AB = A - Bc
    ab2 = (AB ** 2).sum(-1)                            # (B,nao,nao,1,1)
    P = (a[..., None] * A + b[..., None] * Bc) / p[..., None]
    PA = P - A
    PB = P - Bc
    mu = a * b / p
    pref = (torch.pi / p) ** 1.5 * torch.exp(-mu * ab2)

    Lm = ao_L[:, :, None, None, None, :].to(dt)        # (B,nao,1,1,1,3)
    Ln = ao_L[:, None, :, None, None, :].to(dt)        # (B,1,nao,1,1,3)
    inv2p = (1.0 / (2.0 * p))[..., None]
    s1d = ((1 - Lm) * (1 - Ln)
           + Lm * (1 - Ln) * PA
           + (1 - Lm) * Ln * PB
           + Lm * Ln * (PA * PB + inv2p))
    prim = pref * s1d.prod(dim=-1)                     # (B,nao,nao,3,3)
    ci = coefs[:, :, None, :, None]
    cj = coefs[:, None, :, None, :]
    S = (ci * cj * prim).sum(dim=(-1, -2))             # (B,nao,nao)
    d = torch.rsqrt(torch.clamp(torch.diagonal(S, dim1=-2, dim2=-1), min=1e-12))
    return d[:, :, None] * S * d[:, None, :]


def _assemble_batch(z_lists: List[Tuple[int, ...]], natm_max: int,
                    charges: Optional[List[int]] = None):
    """Pad the per-molecule AO bookkeeping of a whole batch into numpy arrays.

    The only Python loop left in the featurizer, and it does nothing but numpy
    slice copies against the composition cache -- no SCF, no eigh, no GPU sync.

    ``charges`` is the per-molecule total charge; None means every molecule is
    neutral. Charge affects only the occupation count, not the AO tables, which
    is why the composition cache stays keyed on the elements alone.
    """
    if charges is None:
        charges = [0] * len(z_lists)
    elif len(charges) != len(z_lists):
        raise ValueError(
            f"got {len(charges)} charges for {len(z_lists)} molecules")
    tables = [_ao_table_cached(z) for z in z_lists]
    naos = [t[0].shape[0] for t in tables]
    B, nao_max = len(tables), max(naos)

    ao_atom_p = np.zeros((B, nao_max), np.int64)
    ao_l_p = np.full((B, nao_max), -1, np.int64)
    ao_L_p = np.zeros((B, nao_max, 3), np.float64)
    exps_p = np.ones((B, nao_max, 3), np.float64)
    coefs_p = np.zeros((B, nao_max, 3), np.float64)
    hdiag_p = np.full((B, nao_max), _PAD_HDIAG, np.float64)
    ao_mask_p = np.zeros((B, nao_max), np.float64)
    val_z_p = np.zeros((B, natm_max), np.float64)
    n_occ_p = np.zeros((B,), np.int64)

    for g, (t, z) in enumerate(zip(tables, z_lists)):
        ao_atom, ao_l, ao_L, exps, coefs, hdiag, n_val = t
        k = naos[g]
        ao_atom_p[g, :k] = ao_atom
        ao_l_p[g, :k] = ao_l
        ao_L_p[g, :k] = ao_L
        exps_p[g, :k] = exps
        coefs_p[g, :k] = coefs
        hdiag_p[g, :k] = hdiag
        ao_mask_p[g, :k] = 1.0
        val_z_p[g, :len(z)] = [_ELEMENTS[int(zz)]["valence_Z"] for zz in z]
        n_occ_p[g] = _closed_shell_n_occ(n_val, charges[g], z)

    return (ao_atom_p, ao_l_p, ao_L_p, exps_p, coefs_p, hdiag_p, ao_mask_p,
            val_z_p, n_occ_p, nao_max)


TB_PACK_KEYS = ("tb_ao_atom", "tb_ao_l", "tb_ao_L", "tb_exps", "tb_coefs",
                "tb_hdiag", "tb_ao_mask", "tb_val_z", "tb_n_occ",
                "tb_local_idx")


def build_tb_pack(z_lists: List[Tuple[int, ...]],
                  charges: Optional[List[int]] = None,
                  ) -> Dict[str, torch.Tensor]:
    """Padded AO bookkeeping for a batch, as CPU tensors.

    ``tb_features`` otherwise derives this inside the training step, which costs
    two ``.cpu()`` calls on batch tensors -- i.e. a full device sync per batch,
    625 times an epoch, stalling every overlap the dataloader would give us.
    None of it depends on the coordinates or on the model, only on the element
    sequences, so a collate function can build it in a worker process instead
    and the training step just consumes tensors.
    """
    natm_max = max((len(z) for z in z_lists), default=1)
    (ao_atom_p, ao_l_p, ao_L_p, exps_p, coefs_p, hdiag_p, ao_mask_p,
     val_z_p, n_occ_p, _nao_max) = _assemble_batch(z_lists, natm_max, charges)

    local = (np.concatenate([np.arange(len(z), dtype=np.int64) for z in z_lists])
             if z_lists else np.zeros(0, np.int64))

    dt = torch.float64
    return {
        "tb_ao_atom": torch.from_numpy(ao_atom_p),
        "tb_ao_l": torch.from_numpy(ao_l_p),
        "tb_ao_L": torch.from_numpy(ao_L_p).to(dt),
        "tb_exps": torch.from_numpy(exps_p).to(dt),
        "tb_coefs": torch.from_numpy(coefs_p).to(dt),
        "tb_hdiag": torch.from_numpy(hdiag_p).to(dt),
        "tb_ao_mask": torch.from_numpy(ao_mask_p).to(dt),
        "tb_val_z": torch.from_numpy(val_z_p).to(dt),
        "tb_n_occ": torch.from_numpy(n_occ_p),
        "tb_local_idx": torch.from_numpy(local),
    }


def _batched_features(coords_pad, ao_atom, ao_l, ao_L, exps, coefs, hdiag,
                      ao_mask, val_z, n_occ):
    """Node (B,natm,12) and pair (B,natm,natm,6) features for a padded batch.

    Mirrors ``_molecule_matrices`` + ``_molecule_features`` exactly, with a
    leading batch axis and masked padding. Runs in float64 for the same reason
    the reference path does: float32 ``eigh`` backward produces NaNs near the
    degeneracies that a tight-binding Hamiltonian is full of.
    """
    dt = torch.float64
    B, natm_max = val_z.shape
    nao_max = ao_mask.shape[1]

    coords_bohr = coords_pad.to(dt) * ANG_TO_BOHR
    ao_coords = torch.gather(
        coords_bohr, 1, ao_atom.unsqueeze(-1).expand(B, nao_max, 3))

    S = _maybe_compile_overlap()(ao_coords, ao_L, exps, coefs)
    # Zero the padded rows/columns and give the padded diagonal a value well
    # outside the real overlap spectrum, so S is exactly block diagonal with
    # disjoint blocks (see _PAD_OVERLAP).
    mm = ao_mask[:, :, None] * ao_mask[:, None, :]
    S = S * mm + torch.diag_embed((1.0 - ao_mask) * _PAD_OVERLAP)

    hsum = 0.5 * (hdiag[:, :, None] + hdiag[:, None, :])
    H = _K_EHT * S * hsum
    H = (H - torch.diag_embed(torch.diagonal(H, dim1=-2, dim2=-1))
         + torch.diag_embed(hdiag))

    X = _sym_orthogonalize(S)

    # Occupancy over MO index; padded levels sit at _PAD_HDIAG so they sort to
    # the top and are never selected.
    mo_idx = torch.arange(nao_max, device=coords_pad.device)
    occ = (mo_idx[None, :] < n_occ[:, None]).to(dt)
    P_o, W_o = _density_matrices(X @ H @ X, occ)
    P = X @ P_o @ X
    W = X @ W_o @ X
    F = H

    G = torch.zeros((B, nao_max, natm_max), dtype=dt, device=coords_pad.device)
    b_ix = torch.arange(B, device=coords_pad.device)[:, None].expand(B, nao_max)
    a_ix = mo_idx[None, :].expand(B, nao_max)
    G[b_ix, a_ix, ao_atom] = ao_mask

    Gt = G.transpose(-1, -2)

    def blocksum(M):
        return Gt @ M @ G

    def frob(M):
        return torch.sqrt(torch.clamp(blocksum(M * M), min=1e-18))

    def atom_sum(v):  # (B,nao) -> (B,natm)
        return (Gt @ v.unsqueeze(-1)).squeeze(-1)

    PS = P @ S
    pop_ao = torch.diagonal(PS, dim1=-2, dim2=-1)
    pop_atom = atom_sum(pop_ao)

    PHt = blocksum(P * H.transpose(-1, -2))
    PFt = blocksum(P * F.transpose(-1, -2))
    dg = lambda M: torch.diagonal(M, dim1=-2, dim2=-1)  # noqa: E731

    node = torch.zeros((B, natm_max, 12), dtype=dt, device=coords_pad.device)
    node[:, :, 0] = val_z - pop_atom
    node[:, :, 1] = pop_atom
    for l in range(N_L_CHANNELS):
        node[:, :, 2 + l] = atom_sum(pop_ao * (ao_l == l).to(dt))
    node[:, :, 7] = dg(PHt)
    node[:, :, 8] = dg(PFt)
    node[:, :, 9] = atom_sum(dg(W))
    node[:, :, 10] = dg(frob(P))
    node[:, :, 11] = pop_atom

    pair = torch.zeros((B, natm_max, natm_max, 6), dtype=dt,
                       device=coords_pad.device)
    pair[:, :, :, 0] = blocksum(PS * PS.transpose(-1, -2))
    pair[:, :, :, 1] = frob(P)
    pair[:, :, :, 2] = frob(F)
    pair[:, :, :, 3] = frob(S)
    pair[:, :, :, 4] = frob(H)
    pair[:, :, :, 5] = PFt
    return node, pair


def tb_features(atom_positions: torch.Tensor,
                atom_numbers: torch.Tensor,
                edge_index: torch.Tensor,
                batch_atom: torch.Tensor,
                num_graphs: int,
                pack: Optional[Dict[str, torch.Tensor]] = None,
                charges: Optional[List[int]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched differentiable TB features.

    Returns (node_scalars [n_atoms,12], edge_features [n_edges,6]), both
    differentiable w.r.t. ``atom_positions`` (Angstrom).

    ``pack`` is the output of ``build_tb_pack`` for this batch, already on the
    right device. Supplying it keeps the AO bookkeeping -- and the two device
    syncs deriving it needs -- out of the training step. Without it the
    bookkeeping is derived here, which is what one-off callers (feature stats,
    the ASE calculator) want.

    ``charges`` gives the per-molecule total charge; it applies only to the
    derive-here path, since a supplied ``pack`` already has the occupations
    baked in.
    """
    dev = atom_positions.device
    n_atoms = atom_positions.shape[0]
    src, dst = edge_index[0], edge_index[1]
    edge_mol = batch_atom[src]
    dt = torch.float64

    if pack is not None:
        local_idx = pack["tb_local_idx"]
        natm_max = pack["tb_val_z"].shape[1]

        def to(t, d):
            return t.to(device=dev, dtype=d)

        ao_atom_t = to(pack["tb_ao_atom"], torch.long)
        ao_l_t = to(pack["tb_ao_l"], torch.long)
        ao_L_t = to(pack["tb_ao_L"], dt)
        exps_t = to(pack["tb_exps"], dt)
        coefs_t = to(pack["tb_coefs"], dt)
        hdiag_t = to(pack["tb_hdiag"], dt)
        ao_mask_t = to(pack["tb_ao_mask"], dt)
        val_z_t = to(pack["tb_val_z"], dt)
        n_occ_t = to(pack["tb_n_occ"], torch.long)
    else:
        # Per-atom LOCAL index within its molecule, fully vectorized. Assumes
        # atoms are grouped contiguously by molecule, as produced by the graph
        # collate -- the same assumption the offset arithmetic below relies on.
        counts = torch.bincount(batch_atom, minlength=num_graphs)
        offsets = torch.zeros_like(counts)
        offsets[1:] = torch.cumsum(counts, 0)[:-1]
        local_idx = torch.arange(n_atoms, device=dev) - offsets[batch_atom]

        # A single host transfer for the whole batch; all bookkeeping below is
        # numpy against the composition cache, so no per-molecule GPU sync.
        z_cpu = atom_numbers.detach().cpu().numpy()
        counts_cpu = np.bincount(batch_atom.detach().cpu().numpy(),
                                 minlength=num_graphs)
        natm_max = max(int(counts_cpu.max()), 1)
        z_lists: List[Tuple[int, ...]] = []
        q_list: List[int] = []
        at = 0
        for g in range(num_graphs):
            c = int(counts_cpu[g])
            if c:
                z_lists.append(tuple(int(z) for z in z_cpu[at:at + c]))
                q_list.append(int(charges[g]) if charges is not None else 0)
            else:
                # An empty graph slot cannot be built from an empty AO table;
                # stand in a hydride so the batch shape holds. Nothing indexes
                # that slot, because no atom or edge maps to it, so its features
                # are discarded -- the -1 charge is only there to make it an
                # even electron count the closed-shell occupation accepts.
                z_lists.append((1,))
                q_list.append(-1)
            at += c

        (ao_atom_p, ao_l_p, ao_L_p, exps_p, coefs_p, hdiag_p, ao_mask_p,
         val_z_p, n_occ_p, _nao_max) = _assemble_batch(z_lists, natm_max, q_list)

        def to(arr, d):
            return torch.as_tensor(arr, dtype=d, device=dev)

        ao_atom_t = to(ao_atom_p, torch.long)
        ao_l_t = to(ao_l_p, torch.long)
        ao_L_t = to(ao_L_p, dt)
        exps_t = to(exps_p, dt)
        coefs_t = to(coefs_p, dt)
        hdiag_t = to(hdiag_p, dt)
        ao_mask_t = to(ao_mask_p, dt)
        val_z_t = to(val_z_p, dt)
        n_occ_t = to(n_occ_p, torch.long)

    coords_pad = atom_positions.new_zeros((num_graphs, natm_max, 3))
    coords_pad = coords_pad.index_put((batch_atom, local_idx), atom_positions)

    node, pair = _batched_features(
        coords_pad, ao_atom_t, ao_l_t, ao_L_t, exps_t, coefs_t, hdiag_t,
        ao_mask_t, val_z_t, n_occ_t)

    out_dt = atom_positions.dtype
    node_out = node[batch_atom, local_idx].to(out_dt)
    edge_out = pair[edge_mol, local_idx[src], local_idx[dst]].to(out_dt)
    return node_out, edge_out


@torch.no_grad()
def tb_feature_stats(dataset, max_frames: int = 512):
    """Mean/std of TB node (12) and edge (6) features over a dataset sample,
    for the model's FeatureStandardizer buffers."""
    import numpy as np
    nsum = np.zeros(12); nsq = np.zeros(12); ncnt = 0
    esum = np.zeros(6); esq = np.zeros(6); ecnt = 0
    n = min(len(dataset), max_frames)
    for i in range(n):
        d = dataset[i]
        pos = d.atom_positions.to(torch.get_default_dtype())
        Z = d.atom_numbers
        ei = d.edge_index
        ba = torch.zeros(pos.shape[0], dtype=torch.long)
        node, edge = tb_features(pos, Z, ei, ba, 1)
        node = node.numpy(); edge = edge.numpy()
        nsum += node.sum(0); nsq += (node ** 2).sum(0); ncnt += node.shape[0]
        if edge.shape[0]:
            esum += edge.sum(0); esq += (edge ** 2).sum(0); ecnt += edge.shape[0]
    nmean = nsum / max(ncnt, 1); nstd = np.sqrt(np.maximum(nsq / max(ncnt, 1) - nmean ** 2, 1e-12))
    emean = esum / max(ecnt, 1); estd = np.sqrt(np.maximum(esq / max(ecnt, 1) - emean ** 2, 1e-12))
    return (nmean.astype("float32"), nstd.astype("float32"),
            emean.astype("float32"), estd.astype("float32"))
