# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Spectral-derivative validation: Daleckii-Krein against the broadened form.

This is the SI protocol of "Validating the spectral derivative". Where
test_tb_gradient_exact.py asks only whether the shipped gradient is right, this
file asks *why* the alternative is wrong, and pins the three claims the paper
makes about it:

  Scheme 1  the divided-difference (Daleckii-Krein) form of Eq. (eq:dk) agrees
            with high-precision central differences, and is invariant to an
            orthogonal rotation of eigenvectors inside a degenerate block
            (Proposition, basis independence).
  Scheme 2  the broadened eigenvector form of Eq. (eq:broadening), 1/d ->
            d/(d^2+eps), carries an error that does NOT vanish as eps -> 0.
  Scheme 3  that broadened error depends on which basis LAPACK happened to
            return inside each degenerate block, so it is not even a
            well-defined function of the geometry.

Molecules are chosen for exact degeneracy: CH4's t2 triplet, NH3's e doublet,
benzene's pi pairs. H2O is carried as a control -- it has no degeneracy, so
broadening must converge there, which is what makes the failure elsewhere
attributable to degeneracy rather than to broadening as such.

WHY BROADENING IS NOT MERELY INACCURATE. Both schemes assemble
dY = V [L o (V^T dA V)] V^T. They differ only in L. For a degenerate pair
sharing an occupation, W's divided difference is

    L_ij = (2 occ_i e_i - 2 occ_j e_j) / (e_i - e_j) = 2 occ        (exact)

while broadening gives

    L_ij = (2 occ d) * d/(d^2 + eps) = 2 occ d^2/(d^2 + eps) -> 0   (wrong)

so a finite term is annihilated rather than perturbed. Within a block this
leaves L = f'(w) I instead of the correct f'(w) 11^T, and a Hadamard product
with I extracts a diagonal, which is not invariant under rotation of the block
-- hence Scheme 3. P is untouched because its numerator, 2(occ_i - occ_j), is
identically zero across a degenerate pair; check_error_lives_in_W pins that, and
it is why the error is specific to the energy-partition features.

Run either way:  python tests/test_spectral_derivative.py
                 pytest tests/test_spectral_derivative.py
"""

import contextlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbace.data import tb_features  # noqa: E402
from orbace.data.tb_features import (  # noqa: E402
    _ELEMENTS, _loewner, _molecule_features, _molecule_matrices,
    _sym_orthogonalize,
)


@pytest.fixture(autouse=True)
def _float64():
    """Scoped to this module's tests.

    Setting the default dtype at import time instead leaks float64 into every
    other test pytest collects afterwards, which breaks six of them.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


_S3 = 0.8660254037844386  # sin 60, for the benzene hexagon
_RC, _RH = 1.39, 2.48     # ring radius and H radius, C-C 1.39 / C-H 1.09


def _hexagon(r):
    return [[r, 0.0, 0.0], [r / 2, r * _S3, 0.0], [-r / 2, r * _S3, 0.0],
            [-r, 0.0, 0.0], [-r / 2, -r * _S3, 0.0], [r / 2, -r * _S3, 0.0]]


def _c3v_tripod(rs, rz):
    """Three sites at azimuth 0, 120, 240 -- exactly C3v up to float64 rounding.

    Built from the shared literals -0.5 and sin 60 rather than from per-atom
    coordinates, so the three radii agree to the last bit and the e doublet is
    degenerate to ~1e-16. Ammonia written out by hand instead is C3v only to
    about 3e-4, which opens the doublet to 6e-5 -- enough that broadening
    converges on it and the molecule silently stops testing degeneracy at all.
    """
    return [[rs, 0.0, rz],
            [-rs / 2, rs * _S3, rz],
            [-rs / 2, -rs * _S3, rz]]


# name -> (atomic numbers, coordinates, has an exact occupied degeneracy)
MOLECULES = {
    "H2O (control)": ([8, 1, 1],
                      [[0.0, 0.0, 0.117], [0.0, 0.757, -0.467],
                       [0.0, -0.757, -0.467]], False),
    "CH4 (Td)": ([6, 1, 1, 1, 1],
                 [[0.0, 0.0, 0.0], [0.629, 0.629, 0.629],
                  [-0.629, -0.629, 0.629], [-0.629, 0.629, -0.629],
                  [0.629, -0.629, -0.629]], True),
    # N-H 1.012 A, H-N-H 106.7 deg: sin/cos of the polar angle are 0.92643 and
    # 0.37647, so the H shell sits at radius 0.93755 and height 0.38099.
    "NH3 (C3v)": ([7, 1, 1, 1],
                  [[0.0, 0.0, 0.0]] + _c3v_tripod(0.93755, 0.38099), True),
    "C6H6 (D6h)": ([6] * 6 + [1] * 6, _hexagon(_RC) + _hexagon(_RH), True),
}

EPS_SWEEP = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)

FD_STEP = 1e-5
FD_TOL = 1e-7          # central differences in float64 bottom out near 1e-9
BASIS_TOL = 1e-9       # DK must be invariant to a block rotation
GROSS_TOL = 1e-2       # "wrong by percents", not "slightly off"
PLATEAU_RATIO = 0.5    # err(eps_min) >= this * err(eps_mid) means no convergence


# --------------------------------------------------------------------------
# the two schemes
# --------------------------------------------------------------------------
def _loewner_broadened(w, fw, fpw, tol=None, *, eps):
    """Eq. (eq:broadening): the same L, with 1/d replaced by d/(d^2+eps).

    The diagonal is f'(w) in both schemes -- eigenvector perturbation theory
    excludes j == i, so the i == i term comes from the eigenvalue derivative and
    is never broadened. Every difference between the schemes is off-diagonal.
    """
    dw = w.unsqueeze(-1) - w.unsqueeze(-2)
    df = fw.unsqueeze(-1) - fw.unsqueeze(-2)
    off = df * dw / (dw * dw + eps)
    eye = torch.eye(w.shape[-1], dtype=torch.bool, device=w.device)
    return torch.where(eye, fpw.unsqueeze(-1).expand_as(off), off)


@contextlib.contextmanager
def broadened(eps, only="all"):
    """Swap the divided difference for the broadened form inside tb_features.

    Both spectral functions read _loewner from the module globals at call time,
    so patching it reroutes the backward of both while leaving every forward
    value, and therefore the FD reference, untouched.

    ``only`` selects which one to damage, which matters more than it looks.
    S^{-1/2} is a spectral function too, and the overlap matrix of a symmetric
    molecule is just as degenerate as its Hamiltonian, so broadening everything
    corrupts P as well -- not through P's own divided difference but through the
    orthogonalizer upstream of it. Attributing the error to W specifically
    requires damaging only the density-matrix step, which is ``only="density"``.
    """
    if only == "all":
        real = tb_features._loewner
        tb_features._loewner = lambda w, fw, fpw, tol=None: _loewner_broadened(
            w, fw, fpw, eps=eps)
        try:
            yield
        finally:
            tb_features._loewner = real
        return

    assert only == "density", only
    real_backward = tb_features._DensityMatrices.backward

    def patched(ctx, gP, gW):
        e, V, occ, fP, fW = ctx.saved_tensors
        LP = _loewner_broadened(e, fP, torch.zeros_like(e), eps=eps)
        LW = _loewner_broadened(e, fW, 2.0 * occ, eps=eps)
        Vt = V.transpose(-1, -2)
        M = LP * (Vt @ gP @ V) + LW * (Vt @ gW @ V)
        return tb_features._sym(V @ M @ Vt), None

    tb_features._DensityMatrices.backward = staticmethod(patched)
    try:
        yield
    finally:
        tb_features._DensityMatrices.backward = real_backward


@contextlib.contextmanager
def block_rotated_eigh(seed, tol=1e-8):
    """Rotate eigenvectors within each degenerate block by a random orthogonal Q.

    A = V diag(w) V^T is preserved exactly, and so is Y = V f(w) V^T, because
    f(w) is constant across a block. Only the basis handed to the backward
    changes -- which is the arbitrary choice the paper says a correct derivative
    must not depend on.
    """
    real = torch.linalg.eigh

    def patched(A, *args, **kwargs):
        w, V = real(A, *args, **kwargs)
        if V.ndim != 2:            # batched path: leave it alone
            return w, V
        gen = torch.Generator(device="cpu").manual_seed(seed)
        Vr, n, i = V.clone(), w.shape[-1], 0
        while i < n:
            j = i + 1
            while j < n and (w[j] - w[i]).abs() < tol:
                j += 1
            if j - i > 1:
                M = torch.randn(j - i, j - i, dtype=V.dtype, generator=gen)
                Q, R = torch.linalg.qr(M)
                Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
                Vr[:, i:j] = V[:, i:j] @ Q
            i = j
        return w, Vr

    torch.linalg.eigh = patched
    try:
        yield
    finally:
        torch.linalg.eigh = real


# --------------------------------------------------------------------------
# machinery
# --------------------------------------------------------------------------
def _projected_loss(coords, zs, wn, wp):
    node, pair = _molecule_features(coords, zs)
    return (node * wn).sum() + (pair * wp).sum()


def _weights(zs, xyz):
    coords = torch.tensor(xyz, requires_grad=True)
    node, pair = _molecule_features(coords, zs)
    torch.manual_seed(0)
    return torch.randn_like(node), torch.randn_like(pair)


def _analytic(zs, xyz, wn, wp):
    coords = torch.tensor(xyz, requires_grad=True)
    g, = torch.autograd.grad(_projected_loss(coords, zs, wn, wp), coords)
    return g


def _finite_difference(zs, xyz, wn, wp):
    base = torch.tensor(xyz)
    g = torch.zeros_like(base)
    for a in range(base.shape[0]):
        for k in range(3):
            for sign in (+1, -1):
                shifted = base.clone()
                shifted[a, k] += sign * FD_STEP
                with torch.no_grad():
                    g[a, k] += sign * _projected_loss(
                        shifted, zs, wn, wp) / (2 * FD_STEP)
    return g


def _occupied_gap(zs, xyz):
    with torch.no_grad():
        _P, _W, S, H, _F, _aa, _al = _molecule_matrices(torch.tensor(xyz), zs)
        X = _sym_orthogonalize(S)
        e = torch.linalg.eigvalsh(X @ H @ X)
    n_occ = int(round(sum(_ELEMENTS[int(z)]["valence_Z"] for z in zs))) // 2
    eo = e[:n_occ]
    gaps = (eo[1:] - eo[:-1]).abs()
    return gaps.min().item() if gaps.numel() else float("inf")


def _err(g, ref):
    return (g - ref).abs().max().item()


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------
def test_scheme1_dk_matches_finite_differences():
    """Scheme 1: the shipped divided-difference form is FD-exact everywhere."""
    print("\nScheme 1: Daleckii-Krein vs central differences")
    ok = True
    for name, (zs, xyz, degen) in MOLECULES.items():
        wn, wp = _weights(zs, xyz)
        ref = _finite_difference(zs, xyz, wn, wp)
        err = _err(_analytic(zs, xyz, wn, wp), ref)
        scale = max(ref.abs().max().item(), 1.0)
        good = err < FD_TOL * scale
        ok &= good
        print(f"  {name:<14s} gap={_occupied_gap(zs, xyz):.2e}  "
              f"max|auto-FD|={err:.3e}  (scale {scale:.2e})  "
              f"{'OK' if good else 'FAIL'}")
    assert ok


def test_scheme1_dk_is_basis_independent():
    """Scheme 1: rotating a degenerate block must not move the gradient."""
    print("\nScheme 1: basis independence of Daleckii-Krein")
    ok = True
    for name, (zs, xyz, degen) in MOLECULES.items():
        wn, wp = _weights(zs, xyz)
        g0 = _analytic(zs, xyz, wn, wp)
        with block_rotated_eigh(seed=1):
            g1 = _analytic(zs, xyz, wn, wp)
        with block_rotated_eigh(seed=2):
            g2 = _analytic(zs, xyz, wn, wp)
        scale = max(g0.abs().max().item(), 1.0)
        drift = max(_err(g1, g0), _err(g2, g0))
        good = drift < BASIS_TOL * scale
        ok &= good
        print(f"  {name:<14s} max|rotated-original|={drift:.3e}  "
              f"(scale {scale:.2e})  {'OK' if good else 'FAIL'}")
    assert ok


def test_scheme2_broadened_does_not_converge():
    """Scheme 2: broadened error plateaus at a nonzero floor as eps -> 0.

    The control matters as much as the degenerate cases: for H2O the error must
    fall to FD noise, proving the sweep is capable of converging and that the
    plateau elsewhere is degeneracy and not an artefact of the sweep.
    """
    print("\nScheme 2: broadened eigenvector form, eps sweep")
    ok = True
    for name, (zs, xyz, degen) in MOLECULES.items():
        wn, wp = _weights(zs, xyz)
        ref = _finite_difference(zs, xyz, wn, wp)
        scale = max(ref.abs().max().item(), 1.0)
        errs = []
        for eps in EPS_SWEEP:
            with broadened(eps):
                errs.append(_err(_analytic(zs, xyz, wn, wp), ref))
        rel = [e / scale for e in errs]
        if degen:
            # still grossly wrong at the tightest eps, and no longer improving
            good = (rel[-1] > GROSS_TOL
                    and errs[-1] >= PLATEAU_RATIO * errs[len(errs) // 2])
        else:
            good = rel[-1] < FD_TOL     # control: broadening converges
        ok &= good
        print(f"  {name:<14s} " + " ".join(
            f"{e:.1e}" for e in rel) + f"   {'OK' if good else 'FAIL'}")
    print("     eps = " + " ".join(f"{e:.0e}" for e in EPS_SWEEP)
          + "   (relative max error)")
    assert ok


def test_scheme3_broadened_is_basis_dependent():
    """Scheme 3: the broadened error changes with the intra-block basis."""
    print("\nScheme 3: broadened form under a permuted degenerate basis")
    eps = EPS_SWEEP[-1]
    ok = True
    for name, (zs, xyz, degen) in MOLECULES.items():
        wn, wp = _weights(zs, xyz)
        with broadened(eps):
            g0 = _analytic(zs, xyz, wn, wp)
            with block_rotated_eigh(seed=1):
                g1 = _analytic(zs, xyz, wn, wp)
        scale = max(g0.abs().max().item(), 1.0)
        spread = _err(g1, g0) / scale
        # degenerate: the answer depends on an arbitrary choice, so it moves.
        # control: no block to rotate, so it must not move at all.
        good = spread > GROSS_TOL if degen else spread < BASIS_TOL
        ok &= good
        print(f"  {name:<14s} max|basisA-basisB|/scale={spread:.3e}  "
              f"{'OK' if good else 'FAIL'}")
    assert ok


def test_error_lives_in_W():
    """The annihilated term is W's; P's is identically zero anyway.

    Acts directly on the Loewner matrices for a synthetic exactly-degenerate
    occupied pair, which isolates the mechanism from any molecule.
    """
    print("\nMechanism: which spectral function the error lives in")
    occ = torch.tensor([1.0, 1.0])
    e = torch.tensor([-0.4, -0.4])              # exactly degenerate, occupied
    fP, fW = 2.0 * occ, 2.0 * occ * e

    LP = _loewner(e, fP, torch.zeros_like(e))
    LW = _loewner(e, fW, 2.0 * occ)
    LP_b = _loewner_broadened(e, fP, torch.zeros_like(e), eps=1e-12)
    LW_b = _loewner_broadened(e, fW, 2.0 * occ, eps=1e-12)

    p_same = torch.allclose(LP, LP_b)
    w_off_exact = LW[0, 1].item()
    w_off_broad = LW_b[0, 1].item()
    print(f"  P off-diagonal: exact={LP[0, 1].item():+.3f} "
          f"broadened={LP_b[0, 1].item():+.3f}  identical={p_same}")
    print(f"  W off-diagonal: exact={w_off_exact:+.3f} "
          f"broadened={w_off_broad:+.3f}  (true limit 2*occ={2 * occ[0]:+.3f})")

    assert p_same, "broadening must not change P"
    assert abs(w_off_exact - 2.0 * occ[0].item()) < 1e-12, "DK must give 2*occ"
    assert abs(w_off_broad) < 1e-12, "broadening must annihilate W's term"


def _matrix_loss(coords, zs, which, wt):
    P, W = _molecule_matrices(coords, zs)[:2]
    return ((P if which == "P" else W) * wt).sum()


def _matrix_grads(zs, xyz, which):
    """Analytic and FD gradient of a random projection of P or W itself."""
    base = torch.tensor(xyz)
    torch.manual_seed(0)
    wt = torch.randn_like(_molecule_matrices(base, zs)[0 if which == "P" else 1])

    coords = torch.tensor(xyz, requires_grad=True)
    g, = torch.autograd.grad(_matrix_loss(coords, zs, which, wt), coords)

    fd = torch.zeros_like(base)
    for a in range(base.shape[0]):
        for k in range(3):
            for sign in (+1, -1):
                shifted = base.clone()
                shifted[a, k] += sign * FD_STEP
                with torch.no_grad():
                    fd[a, k] += sign * _matrix_loss(
                        shifted, zs, which, wt) / (2 * FD_STEP)
    return g, fd


def test_per_matrix_errors_for_si_table():
    """Split the error by spectral function, which is what Table tab:spectral wants.

    P is the reference column, and its residual is not identically zero -- which
    is worth stating precisely rather than rounding to "unaffected". Across a
    degenerate pair P's numerator 2(occ_i - occ_j) vanishes, so degeneracy
    contributes nothing. What remains is the occupied/virtual pairs, where the
    denominator is the HOMO-LUMO gap and broadening perturbs 1/d by O(eps/d^2).
    That is a smooth error that disappears as eps -> 0 (2e-6 at eps=1e-6 against
    5e-10 at eps=1e-12), whereas W's is O(1) and eps-independent. The claim the
    column supports is therefore the ratio, not a zero: W's error exceeds P's by
    three to four orders of magnitude.

    Note this isolates the density-matrix step. Broadening S^{-1/2} as well
    damages P by ~2e-1, because the overlap matrix of a symmetric molecule is
    itself degenerate -- so "the error is specific to W" holds within the P/W
    pair, not across every spectral function in the featurizer.
    """
    print("\nSI Table tab:spectral -- relative error vs finite difference")
    print(f"  {'molecule':<10s} {'scheme':<26s} {'W':>10s} {'P':>10s}")
    rows = [("CH4 (Td)", "divided difference", None),
            ("CH4 (Td)", "broadened, eps=1e-6", 1e-6),
            ("CH4 (Td)", "broadened, eps=1e-12", 1e-12),
            ("C6H6 (D6h)", "divided difference", None),
            ("C6H6 (D6h)", "broadened, eps=1e-6", 1e-6)]

    ok = True
    for mol, label, eps in rows:
        zs, xyz, _degen = MOLECULES[mol]
        out = {}
        for which in ("W", "P"):
            if eps is None:
                g, fd = _matrix_grads(zs, xyz, which)
            else:
                with broadened(eps, only="density"):
                    g, fd = _matrix_grads(zs, xyz, which)
            out[which] = _err(g, fd) / max(fd.abs().max().item(), 1.0)
        if eps is None:
            good = out["W"] < FD_TOL and out["P"] < FD_TOL
        else:
            # W grossly wrong, and wrong by orders of magnitude more than P
            good = (out["W"] > GROSS_TOL
                    and out["W"] > 100.0 * out["P"])
        ok &= good
        print(f"  {mol:<10s} {label:<26s} {out['W']:>10.2e} {out['P']:>10.2e}"
              f"   W/P={out['W'] / max(out['P'], 1e-300):>8.0f}"
              f"  {'OK' if good else 'FAIL'}")
    assert ok


def main():
    torch.set_default_dtype(torch.float64)   # the fixture's job under pytest
    test_scheme1_dk_matches_finite_differences()
    test_scheme1_dk_is_basis_independent()
    test_scheme2_broadened_does_not_converge()
    test_scheme3_broadened_is_basis_dependent()
    test_error_lives_in_W()
    test_per_matrix_errors_for_si_table()
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
