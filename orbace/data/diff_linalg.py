# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Degeneracy-safe differentiable linear algebra for the featurizer.

``torch.linalg.eigh`` has a backward pass with ``1/(lambda_i - lambda_j)`` terms
that diverge to NaN/Inf when eigenvalues are degenerate -- which is common for
symmetric molecules (e.g. methane's degenerate orbitals). This module provides
custom autograd functions for the two spectral operations the featurizer needs:

  * ``lowdin_invsqrt(S)``           -> S^{-1/2}            (matrix function)
  * ``fock_solve(F, n_occ, T)``     -> (P_density, eps)    (occupation function)

Both use the Daleckii--Krein (Loewner) formula for the matrix-function gradient,
where the divided difference (f(li)-f(lj))/(li-lj) is replaced by f'(li) when
|li-lj| is tiny -- finite at degeneracy. The eigendecomposition is kept internal
to the custom Function (its eigenvectors never enter the outer autograd graph),
and the saved spectral quantities are treated as constants in the backward. The
result is first-order-exact gradients (so forces are exact) without NaNs.

NOTE: because the saved decomposition is constant in backward, second derivatives
*through the eigensolve itself* are zero. This is exact when the featurizer has no
learnable parameters (the default): force-loss training differentiates the network
w.r.t. its own weights, and d(features)/d(positions) is weight-independent, so the
omitted term is identically zero. If ``learnable_params=True``, gradients w.r.t.
the VSIP/K featurizer parameters become approximate.
"""

from typing import Callable, Tuple

import torch
from torch import Tensor

_DEGEN_TOL = 1e-5


def _loewner(lam: Tensor, f_lam: Tensor, fp_lam: Tensor) -> Tensor:
    """Loewner matrix L_ij = (f(li)-f(lj))/(li-lj), safe at degeneracy."""
    diff = lam.unsqueeze(-1) - lam.unsqueeze(-2)
    num = f_lam.unsqueeze(-1) - f_lam.unsqueeze(-2)
    small = diff.abs() < _DEGEN_TOL
    diag_val = 0.5 * (fp_lam.unsqueeze(-1) + fp_lam.unsqueeze(-2))
    safe_diff = torch.where(small, torch.ones_like(diff), diff)
    return torch.where(small, diag_val, num / safe_diff)


class _SymMatFun(torch.autograd.Function):
    """Apply scalar function f to a symmetric matrix with safe backward."""

    @staticmethod
    def forward(ctx, A: Tensor, f: Callable, fp: Callable) -> Tensor:
        A = 0.5 * (A + A.transpose(-1, -2))
        lam, V = torch.linalg.eigh(A)
        f_lam = f(lam)
        out = (V * f_lam.unsqueeze(-2)) @ V.transpose(-1, -2)
        ctx.save_for_backward(lam, V, f_lam, fp(lam))
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        lam, V, f_lam, fp_lam = ctx.saved_tensors
        L = _loewner(lam, f_lam, fp_lam)
        g = 0.5 * (grad_out + grad_out.transpose(-1, -2))
        VtGV = V.transpose(-1, -2) @ g @ V
        dA = V @ (L * VtGV) @ V.transpose(-1, -2)
        return 0.5 * (dA + dA.transpose(-1, -2)), None, None


def lowdin_invsqrt(S: Tensor, clamp: float = 1e-8) -> Tensor:
    """S^{-1/2} via symmetric (Löwdin) orthogonalization, degeneracy-safe."""
    def f(l):
        return l.clamp(min=clamp).rsqrt()

    def fp(l):
        lc = l.clamp(min=clamp)
        return torch.where(l > clamp, -0.5 * lc.pow(-1.5), torch.zeros_like(l))

    return _SymMatFun.apply(S, f, fp)


class _FockSolve(torch.autograd.Function):
    """Eigensolve a (symmetric) Fock matrix, returning density P and energies eps.

    P = V diag(occ(eps)) V^T with closed-shell occupations (hard aufbau or Fermi
    smearing). Backward is degeneracy-safe (Loewner for the occupation function;
    plain v v^T for the eigenvalue features).
    """

    @staticmethod
    def forward(ctx, F: Tensor, n_occ: int, T: float) -> Tuple[Tensor, Tensor]:
        F = 0.5 * (F + F.transpose(-1, -2))
        lam, V = torch.linalg.eigh(F)
        n = lam.shape[-1]
        if T > 0.0 and 0 < n_occ < n:
            # Self-consistent mu via Newton iteration (detached from the
            # autograd graph; the backward uses the implicit-function-theorem
            # derivative instead). The midpoint is a good starting guess.
            # This avoids the non-differentiable midpoint formula at degeneracy.
            lam_d = lam.detach()
            N_target = 2.0 * n_occ
            mu = 0.5 * (lam_d[n_occ - 1] + lam_d[n_occ])
            for _ in range(8):
                z = -(lam_d - mu) / T
                sig = torch.sigmoid(z)
                occ_trial = 2.0 * sig
                docc = -(2.0 / T) * sig * (1.0 - sig)
                residual = occ_trial.sum() - N_target
                if residual.abs() < 1e-14:
                    break
                mu = mu - residual / (-docc.sum())
            z = -(lam - mu) / T
            sig = torch.sigmoid(z)
            occ = 2.0 * sig
            occ_p = -(2.0 / T) * sig * (1.0 - sig)
        else:
            occ = torch.zeros_like(lam)
            occ[:n_occ] = 2.0
            occ_p = torch.zeros_like(lam)
        P = (V * occ.unsqueeze(-2)) @ V.transpose(-1, -2)
        ctx.save_for_backward(lam, V, occ, occ_p)
        return P, lam

    @staticmethod
    def backward(ctx, grad_P: Tensor, grad_eps: Tensor):
        lam, V, occ, occ_p = ctx.saved_tensors
        dF = lam.new_zeros((lam.shape[-1], lam.shape[-1]))
        if grad_P is not None:
            L = _loewner(lam, occ, occ_p)
            g = 0.5 * (grad_P + grad_P.transpose(-1, -2))
            VtGV = V.transpose(-1, -2) @ g @ V
            dF = dF + V @ (L * VtGV) @ V.transpose(-1, -2)
            # Chemical-potential response: mu depends on the eigenvalues
            # (particle-number constraint). By the implicit function theorem,
            # d(mu)/d(eps_j) = occ_p[j] / sum_k occ_p[k], giving a rank-1
            # correction that is basis-independent within degenerate subspaces.
            occ_p_sum = occ_p.sum()
            if occ_p_sum.abs() > 1e-30:
                phi_diag = VtGV.diagonal()
                alpha = (phi_diag * (-occ_p)).sum()
                dF = dF + (alpha / occ_p_sum) * (
                    V @ torch.diag(occ_p) @ V.transpose(-1, -2)
                )
        if grad_eps is not None:
            dF = dF + (V * grad_eps.unsqueeze(-2)) @ V.transpose(-1, -2)
        dF = 0.5 * (dF + dF.transpose(-1, -2))
        return dF, None, None


def fock_solve(F: Tensor, n_occ: int, fermi_T: float = 0.0) -> Tuple[Tensor, Tensor]:
    """Return (density P_OAO, orbital energies eps) from an OAO Fock matrix."""
    return _FockSolve.apply(F, n_occ, fermi_T)
