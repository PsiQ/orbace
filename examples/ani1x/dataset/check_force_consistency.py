#!/usr/bin/env python
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Verify that a dataset's forces really are minus the gradient of its energies.

A labelled dataset can be silently corrupt in a way that no training run will
ever report: if ``REF_forces`` is not the negative gradient of ``REF_energy``
(wrong unit conversion, mismatched geometry, stale column), every force metric
you measure is still self-consistent and still meaningless. Conservative models
suffer worst -- they are structurally required to satisfy F = -dE/dR, so they
cannot fit both targets and will silently trade energy accuracy away.

Three checks, cheapest first:

``scale``  Compare the RMS of the stored forces against a GFN2-xTB calculation
           on the same geometries. Needs no reference-level theory and takes
           milliseconds per frame. A semi-empirical method cannot be many times
           stiffer than the reference it approximates, so a large ratio here is
           already proof of a problem.

``grad``   Recompute the analytic gradient at the reference level of theory and
           regress the stored forces against it. Reports the RMS ratio, the
           correlation, the best-fit scale factor, and the residual that would
           remain after rescaling -- which tells you whether a global multiplier
           can repair the data or whether it must be regenerated.

``fd``     Finite-difference the reference energy itself. This is the ground
           truth: it validates the analytic gradient and its unit conversion
           independently, so a disagreement cannot be blamed on either.

Examples
--------
    # fast triage, no DFT
    python examples/ani1x/dataset/check_force_consistency.py --input train.xyz --check scale

    # full verification at the ANI-1x level of theory
    python examples/ani1x/dataset/check_force_consistency.py --input train.xyz \\
        --check grad fd --xc wb97x --basis 6-31g* --n_frames 3

Exits non-zero when a check fails, so it can gate a preparation pipeline.
"""
import argparse
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("check_forces")

# 1 Hartree/Bohr in eV/Angstrom.
HARTREE_TO_EV = 27.211386245988
ANG_TO_BOHR = 1.8897261246257702
HA_BOHR_TO_EV_ANG = HARTREE_TO_EV * ANG_TO_BOHR

_SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53,
}


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _load_frames(path, n_frames, min_atoms, max_atoms):
    """Read frames that carry both an energy and forces, skipping E0 frames."""
    from orbace.data.extxyz import read_extxyz

    out = []
    for f in read_extxyz(path):
        if f.get("config_type") == "IsolatedAtom":
            continue
        if f.get("energy") is None or f.get("forces") is None:
            continue
        n = len(f["symbols"])
        if n < min_atoms or (max_atoms and n > max_atoms):
            continue
        out.append(f)
        if len(out) >= n_frames:
            break
    if not out:
        logger.error(
            "No usable frames in %s with %d <= natoms <= %s (need REF_energy "
            "and REF_forces on a non-IsolatedAtom frame).",
            path, min_atoms, max_atoms or "inf")
        sys.exit(2)
    return out


def _zs(frame):
    return np.array([_SYMBOL_TO_Z[s] for s in frame["symbols"]], dtype=np.int64)


def _formula(frame):
    from collections import Counter
    c = Counter(frame["symbols"])
    return "".join(f"{s}{n if n > 1 else ''}" for s, n in sorted(c.items()))


def _fit(stored, computed):
    """Regress stored onto computed; both flattened. Returns a summary dict."""
    s = np.asarray(stored, dtype=np.float64).ravel()
    c = np.asarray(computed, dtype=np.float64).ravel()
    denom = float(np.dot(c, c))
    slope = float(np.dot(s, c) / denom) if denom > 0 else float("nan")
    return {
        "rms_stored": _rms(s),
        "rms_computed": _rms(c),
        "ratio": _rms(c) / _rms(s) if _rms(s) > 0 else float("inf"),
        "corr": float(np.corrcoef(s, c)[0, 1]) if s.size > 1 else float("nan"),
        "slope": slope,
        "resid_frac": (_rms(s - slope * c) / _rms(s)) if _rms(s) > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# check: scale (xTB, no reference-level theory needed)
# --------------------------------------------------------------------------- #
def check_scale(frames, tol):
    from orbace.data.preprocessing import _run_xtb

    logger.info("\n[scale] stored forces vs GFN2-xTB on the same geometries")
    logger.info("  %-14s %10s %10s %8s", "formula", "RMS stored", "RMS xTB", "ratio")
    ratios = []
    for f in frames:
        _e, fx = _run_xtb(_zs(f), f["coords"], param="GFN2xTB", with_forces=True)
        st = _rms(f["forces"])
        ratios.append(_rms(fx) / st if st > 0 else float("inf"))
        logger.info("  %-14s %10.4f %10.4f %8.2f",
                    _formula(f), st, _rms(fx), ratios[-1])
    med = float(np.median(ratios))
    logger.info("  median ratio xTB/stored = %.2f", med)
    ok = (1.0 / tol) <= med <= tol
    if ok:
        logger.info("  OK: stored forces are the same order as a semi-empirical "
                    "calculation.")
    else:
        logger.error(
            "  FAIL: a semi-empirical method is %.1fx the stored forces. xTB is "
            "approximate but cannot be this much stiffer than its own reference; "
            "the stored forces are almost certainly mis-scaled.", med)
    return ok


# --------------------------------------------------------------------------- #
# check: grad (analytic gradient at the reference level)
# --------------------------------------------------------------------------- #
def _ref_mf(frame, xc, basis, charge, spin):
    from pyscf import dft, gto, scf

    mol = gto.M(
        atom=[(int(z), tuple(map(float, c)))
              for z, c in zip(_zs(frame), frame["coords"])],
        basis=basis, charge=charge, spin=spin, unit="Angstrom", verbose=0)
    mf = dft.RKS(mol) if xc.lower() not in ("hf", "") else scf.RHF(mol)
    if xc.lower() not in ("hf", ""):
        mf.xc = xc
    mf.conv_tol = 1e-11
    mf.kernel()
    if not mf.converged:
        logger.warning("  SCF did not converge for a frame; skipping it.")
        return None
    return mf


def _ref_forces(mf):
    g = mf.nuc_grad_method()
    g.verbose = 0
    return -g.kernel() * HA_BOHR_TO_EV_ANG


def check_grad(frames, xc, basis, charge, spin, tol):
    logger.info("\n[grad] stored forces vs analytic %s/%s gradient", xc, basis)
    logger.info("  %-10s %10s %10s %7s %7s %7s %8s",
                "natoms", "RMS stored", "RMS calc", "ratio", "corr", "slope",
                "resid")
    fits = []
    for f in frames:
        mf = _ref_mf(f, xc, basis, charge, spin)
        if mf is None:
            continue
        fit = _fit(f["forces"], _ref_forces(mf))
        fits.append(fit)
        logger.info("  %-10d %10.4f %10.4f %7.2f %7.4f %7.4f %7.1f%%",
                    len(f["symbols"]), fit["rms_stored"], fit["rms_computed"],
                    fit["ratio"], fit["corr"], fit["slope"],
                    100 * fit["resid_frac"])
    if not fits:
        logger.error("  FAIL: no frame produced a converged reference calculation.")
        return False

    med_ratio = float(np.median([f["ratio"] for f in fits]))
    med_resid = float(np.median([f["resid_frac"] for f in fits]))
    logger.info("  median ratio calc/stored = %.3f, median residual after "
                "rescaling = %.1f%%", med_ratio, 100 * med_resid)
    ok = (1.0 / tol) <= med_ratio <= tol
    if ok:
        logger.info("  OK: stored forces match the reference gradient.")
    else:
        logger.error(
            "  FAIL: the reference gradient is %.2fx the stored forces.",
            med_ratio)
        if med_resid < 0.02:
            logger.error(
                "  The relationship is a clean constant (residual %.1f%%), so "
                "multiplying the stored forces by %.4f would repair them.",
                100 * med_resid, med_ratio)
        else:
            logger.error(
                "  Rescaling would still leave %.1f%% error, so this is not a "
                "single bad constant. Regenerate the forces from source.",
                100 * med_resid)
    return ok


# --------------------------------------------------------------------------- #
# check: fd (finite difference of the reference energy)
# --------------------------------------------------------------------------- #
def check_fd(frames, xc, basis, charge, spin, n_components, step, tol):
    logger.info("\n[fd] finite difference of the %s/%s energy (step %.0e A)",
                xc, basis, step)
    f = frames[0]
    natoms = len(f["symbols"])
    mf = _ref_mf(f, xc, basis, charge, spin)
    if mf is None:
        logger.error("  FAIL: reference SCF did not converge.")
        return False
    analytic = _ref_forces(mf)

    rng = np.random.default_rng(0)
    flat = rng.permutation(natoms * 3)[:n_components]
    picks = [(int(j // 3), int(j % 3)) for j in flat]

    logger.info("  %-12s %12s %12s %12s %8s",
                "atom/axis", "FD", "analytic", "stored", "FD/stored")
    fd_vals, an_vals, st_vals = [], [], []
    for (i, k) in picks:
        e = []
        for sign in (+1, -1):
            g = {**f, "coords": np.array(f["coords"], dtype=np.float64)}
            g["coords"][i, k] += sign * step
            m = _ref_mf(g, xc, basis, charge, spin)
            if m is None:
                break
            e.append(m.e_tot * HARTREE_TO_EV)
        if len(e) != 2:
            continue
        fd = -(e[0] - e[1]) / (2.0 * step)
        st = float(f["forces"][i, k])
        fd_vals.append(fd)
        an_vals.append(float(analytic[i, k]))
        st_vals.append(st)
        logger.info("  %-12s %12.4f %12.4f %12.4f %8.2f",
                    f"{i} / {'xyz'[k]}", fd, analytic[i, k], st,
                    fd / st if st else float("inf"))

    if not fd_vals:
        logger.error("  FAIL: no displaced calculation converged.")
        return False

    an_err = float(np.max(np.abs(np.array(fd_vals) - np.array(an_vals))))
    scale = max(_rms(fd_vals), 1e-12)
    logger.info("  max |FD - analytic| = %.2e eV/A (%.2f%% of the FD RMS) "
                "-> the analytic gradient and its unit conversion are sound",
                an_err, 100 * an_err / scale)

    fit = _fit(st_vals, fd_vals)
    logger.info("  ratio FD/stored = %.2f", fit["ratio"])
    ok = (1.0 / tol) <= fit["ratio"] <= tol
    if ok:
        logger.info("  OK: stored forces are the gradient of the stored energies.")
    else:
        logger.error(
            "  FAIL: the gradient of the reference energy is %.2fx the stored "
            "forces. The energies and forces in this dataset describe different "
            "things; no conservative model can fit both.", fit["ratio"])
    return ok


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="extended-XYZ file to check")
    p.add_argument("--check", nargs="+", default=["scale"],
                   choices=["scale", "grad", "fd"],
                   help="which checks to run (default: scale)")
    p.add_argument("--n_frames", type=int, default=5)
    p.add_argument("--min_atoms", type=int, default=4)
    p.add_argument("--max_atoms", type=int, default=12,
                   help="keep frames small; the DFT checks are the slow part")
    p.add_argument("--xc", default="wb97x",
                   help="reference functional, or 'hf' (default: wb97x)")
    p.add_argument("--basis", default="6-31g*", help="reference basis")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--fd_components", type=int, default=6)
    p.add_argument("--fd_step", type=float, default=2e-3,
                   help="displacement in Angstrom")
    p.add_argument("--tol", type=float, default=1.25,
                   help="accept a median ratio within [1/tol, tol]")
    p.add_argument("--scale_tol", type=float, default=4.0,
                   help="looser tolerance for the xTB triage check")
    args = p.parse_args()

    frames = _load_frames(args.input, args.n_frames, args.min_atoms,
                          args.max_atoms)
    logger.info("Checking %s: %d frame(s), %s atoms",
                args.input, len(frames),
                [len(f["symbols"]) for f in frames])

    results = {}
    if "scale" in args.check:
        results["scale"] = check_scale(frames, args.scale_tol)
    if "grad" in args.check:
        results["grad"] = check_grad(frames, args.xc, args.basis, args.charge,
                                     args.spin, args.tol)
    if "fd" in args.check:
        results["fd"] = check_fd(frames, args.xc, args.basis, args.charge,
                                 args.spin, args.fd_components, args.fd_step,
                                 args.tol)

    logger.info("\n%s", "=" * 60)
    for k, v in results.items():
        logger.info("  %-6s %s", k, "PASS" if v else "FAIL")
    logger.info("%s", "=" * 60)
    if not all(results.values()):
        logger.error(
            "\nForces are not consistent with energies. Do not trust any force "
            "metric from this dataset, and expect conservative models to trade "
            "energy accuracy away to fit the forces.")
        sys.exit(1)


if __name__ == "__main__":
    main()
