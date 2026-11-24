#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Unified OrbACE data preparation -- one entry point for every model variant.

Reads an extxyz, extracts features (HF and localization for the orbital
family; the atom family stores geometry only and rebuilds its tight-binding
features in the forward pass), resolves and subtracts the isolated-atom E0 (and
an optional delta-learning baseline), and writes one frame_*.npz per structure.

  # Orbital-space OrbACE (localized-orbital features):
  python prepare.py --method orbital --input train.xyz --output_dir data/ --basis sto-3g --localizer boys

  # AtomOrbACE. Stores geometry + targets only: the gauge-invariant
  # extended-Hückel features are recomputed in-forward at training time, so no
  # SCF runs here unless an HF baseline asks for one.
  python prepare.py --method atom --input train.xyz --output_dir idata/ --basis sto-3g
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from orbace.data.extxyz import read_extxyz
from orbace.tools.isolated_atoms import (
    _SYMBOL_TO_Z, compute_baseline_atomic_e0s, compute_e0_from_extxyz, subtract_e0,
)
from orbace.tools.timing import record_timing, timed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _store_electronic_state(save_dict, charge, spin):
    """Record a frame's charge and spin, but only when they are not neutral.

    Omitting the neutral case keeps files from existing datasets byte-identical
    and lets consumers read a missing key as neutral, which is what every
    dataset prepared before this existed actually is.
    """
    if charge:
        save_dict["charge"] = np.int64(charge)
    if spin:
        save_dict["spin"] = np.int64(spin)


# ============================================================================
#  Per-frame processing (one function per method; must stay top-level so the
#  ProcessPoolExecutor can pickle them).
# ============================================================================
def _process_frame_orbital(idx, frame, basis, charge, spin, localizer, device,
                           baseline_model, e0s, output_dir,
                           baseline_basis=None, baseline_xc=None,
                           features_from=None, hf_self_forces=True):
    from orbace.data.preprocessing import (
        _build_mol, _run_hf, _localize, compute_baseline_energy,
        hf_baseline_from_mf, is_hf_sto3g_baseline, is_hf_self_baseline, is_sto3g_basis,
    )
    from orbace.data.orbital_features import extract_all_orbital_data
    try:
        mol = _build_mol(frame["symbols"], frame["coords"],
                         basis=basis, charge=charge, spin=spin, unit="Angstrom")
        mf = _run_hf(mol, device=device)

        use_baseline = bool(baseline_model) and baseline_model != "none"
        baseline_e, baseline_f = 0.0, None
        hf_self_e, hf_self_f = None, None
        if use_baseline:
            _at_feat_basis = baseline_basis is None or (
                (baseline_basis or "").lower().replace("-", "").replace("_", "")
                == (basis or "").lower().replace("-", "").replace("_", ""))
            if (is_hf_self_baseline(baseline_model) and _at_feat_basis) or (
                    is_hf_sto3g_baseline(baseline_model) and is_sto3g_basis(basis)):
                baseline_e, baseline_f = hf_baseline_from_mf(mf)
            else:
                baseline_e, baseline_f = compute_baseline_energy(
                    model=baseline_model, symbols=frame["symbols"], coords=frame["coords"],
                    charge=charge, spin=spin, basis_hf=basis, baseline_basis=baseline_basis,
                    baseline_xc=baseline_xc, device=device, with_forces=True)
        else:
            # No-baseline (feature-seed) pass: store the HF-self baseline (from the
            # SAME SCF) so an hf_sto3g delta dir can be built later WITHOUT re-running
            # HF (see --features_from). Computed on the canonical mf, pre-localization.
            # The gradient is ~2/3 of the per-frame cost, so it is skippable when
            # this directory will never seed an HF delta set.
            hf_self_e, hf_self_f = hf_baseline_from_mf(
                mf, with_forces=hf_self_forces)

        # Localize AFTER the baseline gradient (which needs canonical orbitals).
        mf = _localize(mf, localizer)
        orb_data = extract_all_orbital_data(mol, mf, localizer)
        save_dict = {**orb_data}
        if hf_self_e is not None:
            # float64: the total HF energy is large (~1e4 eV); float32 rounding
            # (~1e-4 eV) would perturb the reused residual target vs a direct prep.
            save_dict["hf_self_energy"] = np.float64(hf_self_e)
            if hf_self_f is not None:
                save_dict["hf_self_forces"] = hf_self_f.astype(np.float32)

        if use_baseline:
            # float64 for the same reason as hf_self_energy: this is a large
            # absolute total (~1e4 eV), and multi-fidelity training
            # reconstructs its ABSOLUTE target as residual + baseline, so
            # float32 rounding here (~1e-3 eV) lands straight in the target.
            save_dict["baseline_energy"] = np.float64(baseline_e)
            if baseline_f is not None:
                save_dict["baseline_forces"] = baseline_f.astype(np.float32)
        if frame["energy"] is not None:
            atom_numbers = np.array([mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int64)
            save_dict["energy"] = np.float32(
                subtract_e0(frame["energy"], atom_numbers, e0s) - baseline_e)
        if frame["forces"] is not None:
            f_stored = frame["forces"].astype(np.float32)
            if baseline_f is not None:
                f_stored = f_stored - baseline_f
            save_dict["forces"] = f_stored

        _store_electronic_state(save_dict, charge, spin)
        np.savez(os.path.join(output_dir, f"frame_{idx:06d}.npz"), **save_dict)
        return idx, True, None
    except Exception as e:  # noqa: BLE001
        return idx, False, str(e)


def _process_frame_atom(idx, frame, basis, charge, spin, localizer, device,
                        baseline_model, e0s, output_dir,
                        baseline_basis=None, baseline_xc=None,
                        features_from=None, hf_self_forces=True):
    # hf_self_forces is accepted for call-signature parity with the orbital and
    # re-baseline processors; this prep stores no HF-self baseline of its own.
    from orbace.data.preprocessing import (
        _build_mol, _run_hf, compute_baseline_energy, scf_baseline_from_mf,
        is_hf_self_baseline, is_hf_sto3g_baseline, is_sto3g_basis,
    )
    try:
        mol = _build_mol(frame["symbols"], frame["coords"],
                         basis=basis, charge=charge, spin=spin, unit="Angstrom")

        use_baseline = bool(baseline_model) and baseline_model != "none"
        # An HF-family baseline needs an SCF (mf) to produce baseline E/F. Any
        # OTHER baseline (xtb/gfn*/dft) is computed independently. This prep runs
        # no SCF of its own, so HF runs only when an HF baseline requires it.
        _at_feat_basis = baseline_basis is None or (
            (baseline_basis or "").lower().replace("-", "").replace("_", "")
            == (basis or "").lower().replace("-", "").replace("_", ""))
        hf_baseline = use_baseline and (
            (is_hf_self_baseline(baseline_model) and _at_feat_basis)
            or (is_hf_sto3g_baseline(baseline_model) and is_sto3g_basis(basis)))
        # The model recomputes its own extended-Hückel features from geometry
        # in-forward, so QC/SCF features are never needed here. Skip the (costly)
        # HF SCF unless an HF baseline requires it.
        mf = _run_hf(mol, device=device) if hf_baseline else None

        baseline_e, baseline_f = 0.0, None
        if use_baseline:
            if hf_baseline:
                baseline_e, baseline_f = scf_baseline_from_mf(mf)
            else:
                baseline_e, baseline_f = compute_baseline_energy(
                    model=baseline_model, symbols=frame["symbols"], coords=frame["coords"],
                    charge=charge, spin=spin, basis_hf=basis, baseline_basis=baseline_basis,
                    baseline_xc=baseline_xc, device=device, with_forces=True)

        # Store geometry + targets only: the Hückel features are recomputed
        # in-forward, so there is nothing else to persist.
        save_dict = {
            "atom_positions": np.asarray(frame["coords"], dtype=np.float32),
            "atom_numbers": np.array(
                [mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int64),
        }

        if use_baseline:
            # float64 for the same reason as hf_self_energy: this is a large
            # absolute total (~1e4 eV), and multi-fidelity training
            # reconstructs its ABSOLUTE target as residual + baseline, so
            # float32 rounding here (~1e-3 eV) lands straight in the target.
            save_dict["baseline_energy"] = np.float64(baseline_e)
            if baseline_f is not None:
                save_dict["baseline_forces"] = baseline_f.astype(np.float32)
        if frame["energy"] is not None:
            atom_numbers = np.array([mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int64)
            save_dict["energy"] = np.float32(
                subtract_e0(frame["energy"], atom_numbers, e0s) - baseline_e)
        if frame["forces"] is not None:
            f_stored = frame["forces"].astype(np.float32)
            if baseline_f is not None:
                f_stored = f_stored - baseline_f
            save_dict["forces"] = f_stored

        _store_electronic_state(save_dict, charge, spin)
        np.savez(os.path.join(output_dir, f"frame_{idx:06d}.npz"), **save_dict)
        return idx, True, None
    except Exception as e:  # noqa: BLE001
        return idx, False, str(e)


def _process_frame_reuse(idx, frame, basis, charge, spin, localizer, device,
                         baseline_model, e0s, output_dir,
                         baseline_basis=None, baseline_xc=None,
                         features_from=None, hf_self_forces=True):
    # hf_self_forces is accepted (and unused) so all three processors share one
    # positional call signature; this pass never runs an SCF of its own.
    """Cheap re-baseline pass: REUSE precomputed features from ``features_from``
    (a no-baseline feature dir) and only (re)compute the delta-learning baseline
    + residual target. No HF SCF / localization is re-run.

      * hf_sto3g/hf_self baseline -> read the stored ``hf_self_*`` (same SCF that
        produced the features), so no HF is needed.
      * xtb/gfn*/dft baseline     -> a cheap independent baseline calc.

    This eliminates the 3x redundant feature computation when a split is prepared
    for multiple baselines (features are baseline-independent)."""
    from orbace.data.preprocessing import (
        compute_baseline_energy, is_hf_self_baseline, is_hf_sto3g_baseline, is_sto3g_basis,
    )
    _DROP = {"energy", "forces", "baseline_energy", "baseline_forces",
             "hf_self_energy", "hf_self_forces"}
    try:
        src_path = os.path.join(features_from, f"frame_{idx:06d}.npz")
        if not os.path.exists(src_path):
            return idx, False, f"missing feature file {src_path}"
        src = np.load(src_path, allow_pickle=True)
        save_dict = {k: src[k] for k in src.files if k not in _DROP}
        atom_numbers = np.asarray(src["atom_numbers"], dtype=np.int64)

        use_baseline = bool(baseline_model) and baseline_model != "none"
        baseline_e, baseline_f = 0.0, None
        if use_baseline:
            _at_feat_basis = baseline_basis is None or (
                (baseline_basis or "").lower().replace("-", "").replace("_", "")
                == (basis or "").lower().replace("-", "").replace("_", ""))
            is_hf = ((is_hf_self_baseline(baseline_model) and _at_feat_basis)
                     or (is_hf_sto3g_baseline(baseline_model) and is_sto3g_basis(basis)))
            if is_hf:
                if "hf_self_energy" not in src.files:
                    return idx, False, ("feature dir lacks hf_self_* -- regenerate the "
                                        "no-baseline feature dir with the current prepare.py")
                baseline_e = float(src["hf_self_energy"])
                baseline_f = (np.asarray(src["hf_self_forces"], dtype=np.float32)
                              if "hf_self_forces" in src.files else None)
                if baseline_f is None and frame["forces"] is not None:
                    # Refusing here is the whole point: falling through would
                    # subtract the baseline from the energy but not from the
                    # forces, producing a frame whose energy is a residual and
                    # whose forces are absolute. Training on that mismatch
                    # converges to a self-consistent, meaningless model.
                    return idx, False, (
                        "feature dir has hf_self_energy but no hf_self_forces "
                        "(prepared with --no_hf_self_forces), while this frame "
                        "has reference forces. Re-prepare the feature dir "
                        "without --no_hf_self_forces, or drop forces from the "
                        "target")
            else:
                baseline_e, baseline_f = compute_baseline_energy(
                    model=baseline_model, symbols=frame["symbols"], coords=frame["coords"],
                    charge=charge, spin=spin, basis_hf=basis, baseline_basis=baseline_basis,
                    baseline_xc=baseline_xc, device=device, with_forces=True)
            # float64 for the same reason as hf_self_energy: this is a large
            # absolute total (~1e4 eV), and multi-fidelity training
            # reconstructs its ABSOLUTE target as residual + baseline, so
            # float32 rounding here (~1e-3 eV) lands straight in the target.
            save_dict["baseline_energy"] = np.float64(baseline_e)
            if baseline_f is not None:
                save_dict["baseline_forces"] = baseline_f.astype(np.float32)

        if frame["energy"] is not None:
            save_dict["energy"] = np.float32(
                subtract_e0(frame["energy"], atom_numbers, e0s) - baseline_e)
        if frame["forces"] is not None:
            f_stored = frame["forces"].astype(np.float32)
            if baseline_f is not None:
                f_stored = f_stored - baseline_f
            save_dict["forces"] = f_stored

        np.savez(os.path.join(output_dir, f"frame_{idx:06d}.npz"), **save_dict)
        return idx, True, None
    except Exception as e:  # noqa: BLE001
        return idx, False, str(e)


def _check_force_scale(frames, n_probe=4, ratio_tol=4.0, strict=False):
    """Warn when the stored forces are not the gradient of the stored energies.

    Compares the force RMS against GFN2-xTB on the same geometries. This costs
    milliseconds and needs no reference-level theory, but it catches the failure
    mode that is otherwise invisible: a dataset whose forces are mis-scaled
    trains and evaluates perfectly self-consistently while every force number it
    produces is meaningless, and conservative models quietly sacrifice energy
    accuracy because they cannot satisfy F = -dE/dR against inconsistent labels.

    A semi-empirical method is approximate but cannot be several times stiffer
    than the reference it approximates, so a large ratio is diagnostic. Use
    ``examples/ani1x/dataset/check_force_consistency.py`` to confirm against the real gradient.
    """
    probe = [f for f in frames if f.get("forces") is not None][:n_probe]
    if not probe:
        return True
    try:
        from orbace.data.preprocessing import _run_xtb
    except Exception:  # noqa: BLE001 - xtb is optional
        return True

    ratios = []
    for f in probe:
        try:
            nums = np.array([_SYMBOL_TO_Z[s] for s in f["symbols"]], dtype=np.int64)
            _e, fx = _run_xtb(nums, f["coords"], param="GFN2xTB", with_forces=True)
        except Exception:  # noqa: BLE001 - never let a probe break preparation
            return True
        stored = float(np.sqrt(np.mean(np.asarray(f["forces"], np.float64) ** 2)))
        if stored > 0:
            ratios.append(float(np.sqrt(np.mean(fx.astype(np.float64) ** 2))) / stored)
    if not ratios:
        return True

    med = float(np.median(ratios))
    if (1.0 / ratio_tol) <= med <= ratio_tol:
        logger.info(f"  Force sanity check: xTB/stored force RMS = {med:.2f} (ok)")
        return True

    msg = (f"Force sanity check FAILED: GFN2-xTB forces are {med:.1f}x the "
           f"stored forces on {len(ratios)} probe frame(s). The stored forces "
           f"are very likely NOT the gradient of the stored energies -- force "
           f"metrics will be meaningless and conservative models will trade "
           f"away energy accuracy. Verify with:\n"
           f"    python examples/ani1x/dataset/check_force_consistency.py --input <xyz> "
           f"--check grad fd")
    if strict:
        logger.error(msg)
        raise SystemExit(msg)
    logger.warning(msg)
    return False


_PROCESSORS = {"orbital": _process_frame_orbital, "atom": _process_frame_atom}


# ============================================================================
#  CLI
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(description="Unified OrbACE data preparation")
    p.add_argument("--method", required=True, choices=["orbital", "atom"],
                   help="orbital = localized-orbital features; atom = AtomOrbACE")
    p.add_argument("--input", required=True, help="Input extxyz file")
    p.add_argument("--output_dir", default=None, help="Output dir (default: data/ or idata/)")
    p.add_argument("--basis", default="sto-3g", help="Basis set for HF")
    p.add_argument("--localizer", default="boys", choices=["boys", "pm", "canonical"],
                   help="(orbital only) orbital localization method")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--max_configs", type=int, default=0, help="Max configs (0=all)")
    p.add_argument("--start", type=int, default=0, help="Process frames[start:end] slice")
    p.add_argument("--end", type=int, default=-1, help="Exclusive end (-1 = to end)")
    p.add_argument("--index_offset", type=int, default=0,
                   help="Added to output frame indices for chunked multi-job prep")
    p.add_argument("--e0s_json", default=None, help="Load shared isolated-atom E0s from JSON")
    p.add_argument("--skip_meta", action="store_true",
                   help="Do not (re)write e0s.json/metadata.json (for chunk jobs)")
    p.add_argument("--no_hf_self_forces", action="store_true",
                   help="On a no-baseline pass, skip the HF-self analytic "
                        "gradient (~2/3 of per-frame cost, so roughly a 3x "
                        "speedup). Only safe if this directory will never be "
                        "reused via --features_from to build an HF delta set "
                        "with force targets; that reuse then fails loudly.")
    p.add_argument("--no_force_check", action="store_true",
                   help="Skip the cheap xTB force-scale sanity check")
    p.add_argument("--strict_force_check", action="store_true",
                   help="Abort (instead of warning) if the force-scale check fails")
    p.add_argument("--device", default="auto",
                   help="'auto' (GPU HF if available else CPU), 'cpu', or 'cuda'/'gpu'")
    p.add_argument("--baseline_model", default="none",
                   choices=["none", "xtb", "gfn2", "gfn1", "gfn0",
                            "hf_sto3g", "hf", "hf_self", "hf_321g", "hf_631g", "hf_svp",
                            "dft", "dft_svp", "wb97x_svp", "wb97x_631gs", "pbe_svp", "b3lyp_svp"],
                   help="Cheap baseline for delta-learning (stores baseline_energy per "
                        "frame and subtracts it from the target).")
    p.add_argument("--baseline_basis", default=None, help="Basis for HF/DFT baselines")
    p.add_argument("--baseline_xc", default=None, help="XC functional for 'dft' baseline")
    p.add_argument("--num_workers", type=int, default=1,
                   help="Parallel frame workers (ignored on --device cuda)")
    p.add_argument("--features_from", default=None,
                   help="Reuse precomputed features from this (no-baseline) dir and only "
                        "(re)compute the delta baseline + target -- NO HF SCF re-run. "
                        "Features are baseline-independent, so this avoids recomputing them "
                        "once per baseline. hf baselines read the stored hf_self_* (free); "
                        "xtb/dft baselines run their own cheap calc.")
    return p


def main():
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = "data" if args.method == "orbital" else "idata"
    if args.features_from:
        processor = _process_frame_reuse
        if not os.path.isdir(args.features_from):
            raise SystemExit(f"ERROR: --features_from dir not found: {args.features_from}")
    else:
        processor = _PROCESSORS[args.method]

    if args.device in ("auto", "gpu"):
        try:
            from orbace.data.scf_backend import gpu_available as _gpu_ok
            args.device = "cuda" if (args.device == "gpu" or _gpu_ok()) else "cpu"
        except Exception:
            args.device = "cpu"
    if args.device not in ("cpu", "cuda"):
        args.device = "cpu"
    logger.info(f"Method: {args.method}"
                + f" | SCF backend: {'gpu4pyscf (GPU)' if args.device == 'cuda' else 'PySCF (CPU)'}")

    os.makedirs(args.output_dir, exist_ok=True)
    _tctx = os.path.basename(args.output_dir.rstrip("/"))
    logger.info(f"Reading: {args.input}")
    all_frames = read_extxyz(args.input)
    frames = [f for f in all_frames if f.get("config_type") != "IsolatedAtom"]
    n_isolated = len(all_frames) - len(frames)
    if n_isolated:
        logger.info(f"  Found {n_isolated} IsolatedAtom frame(s) for E0")
    if args.max_configs > 0:
        frames = frames[:args.max_configs]

    n_total = len(frames)
    start = max(0, args.start)
    end = n_total if args.end < 0 else min(args.end, n_total)
    base_index = args.index_offset + start
    frames = frames[start:end]
    logger.info(f"  {n_total} configs; slice [{start}:{end}] ({len(frames)} frames), "
                f"output index base={base_index}")

    all_symbols = set()
    for f in frames:
        all_symbols.update(f["symbols"])
    logger.info(f"  Elements (this slice): {sorted(all_symbols)}")

    if not args.no_force_check:
        _check_force_scale(frames, strict=args.strict_force_check)

    if args.no_hf_self_forces and not args.features_from:
        logger.warning(
            "--no_hf_self_forces: skipping the HF-self analytic gradient. This "
            "dir cannot later seed an HF delta set with force targets via "
            "--features_from (that reuse will abort rather than emit mismatched "
            "energy/force targets).")

    e0s_path = os.path.join(args.output_dir, "e0s.json")
    e0s_from_json = bool(args.e0s_json)
    if e0s_from_json:
        logger.info(f"Loading isolated-atom E0s from {args.e0s_json}")
        with open(args.e0s_json) as f:
            e0s = {int(k): float(v) for k, v in json.load(f).items()}
    else:
        logger.info("Resolving reference isolated-atom E0s...")
        with timed(args.method, "prepare", "e0_resolve", _tctx):
            e0s = compute_e0_from_extxyz(all_frames, basis=args.basis, device=args.device)
    for Z, e0 in sorted(e0s.items()):
        logger.info(f"  E0_ref(Z={Z}): {e0:.6f} eV")

    use_baseline = args.baseline_model and args.baseline_model != "none"
    if use_baseline:
        logger.info(f"Delta-learning baseline: {args.baseline_model}")
        if e0s_from_json:
            # E0s loaded from JSON are AUTHORITATIVE (already the delta reference
            # if they were produced by a delta run). Re-subtracting the baseline
            # here would double-count it and corrupt delta finetune data.
            logger.info("Using E0s from JSON as-is (no additional baseline subtraction)")
        else:
            with timed(args.method, "prepare", "baseline_e0", _tctx):
                baseline_atom_e0s = compute_baseline_atomic_e0s(
                    sorted(all_symbols), args.baseline_model, basis=args.basis,
                    baseline_basis=args.baseline_basis, baseline_xc=args.baseline_xc, device=args.device)
            e0s = {Z: e0s[Z] - baseline_atom_e0s.get(Z, 0.0) for Z in e0s}
            logger.info("Using delta E0s (E0_ref - E0_baseline)")

    if not args.skip_meta:
        with open(e0s_path, "w") as f:
            json.dump({str(k): v for k, v in e0s.items()}, f, indent=2)

    if args.features_from:
        _bl_note = ("read stored hf_self (no SCF)"
                    if use_baseline and str(args.baseline_model).startswith("hf")
                    else ("cheap baseline calc" if use_baseline else "no baseline"))
        logger.info(f"Re-baselining from {args.features_from} ({_bl_note}); "
                    "features reused, NO HF/localization re-run...")
    elif args.method == "atom":
        _scf_note = ("HF SCF for HF baseline only" if use_baseline else "no SCF")
        logger.info(f"Preparing atom data (geometry+targets, {_scf_note}; "
                    "Hückel features recomputed in-forward)...")
    else:
        logger.info(f"Preparing {args.method} features (HF/{args.basis}"
                    + (f" + {args.localizer}" if args.method == "orbital" else "") + ")...")
    t0 = time.time()
    n_success = 0
    n_workers = max(1, int(args.num_workers))
    if args.device == "cuda" and n_workers > 1:
        logger.info("GPU SCF: ignoring --num_workers (single GPU has no gain).")
        n_workers = 1

    def _progress(done):
        if done % 25 == 0 or done == len(frames):
            el = time.time() - t0
            logger.info(f"  {done}/{len(frames)} frames ({el:.0f}s, {done/max(el,1e-6):.1f} frames/s)")

    def _charge_spin(frame):
        """Per-frame charge/spin, falling back to the run-wide defaults.

        A frame that states neither is not the same as one that states neutral:
        only the former should inherit --charge/--spin.
        """
        q, s = frame.get("charge"), frame.get("spin")
        return (args.charge if q is None else int(q),
                args.spin if s is None else int(s))

    n_charged = sum(1 for f in frames if f.get("charge"))
    n_open = sum(1 for f in frames if f.get("spin"))
    if n_charged or n_open:
        logger.info(f"  Per-frame electronic state: {n_charged} charged, "
                    f"{n_open} open-shell frame(s)")
        if args.method == "atom" and n_open:
            raise SystemExit(
                f"{n_open} frame(s) carry a non-zero spin, but the atom method's "
                "Hückel features are closed-shell only; drop those frames.")

    call_args = lambda li, frame: (  # noqa: E731
        base_index + li, frame, args.basis, *_charge_spin(frame), args.localizer,
        args.device, args.baseline_model, e0s, args.output_dir,
        args.baseline_basis, args.baseline_xc, args.features_from,
        not args.no_hf_self_forces)

    _t_feat = time.perf_counter()
    if n_workers == 1:
        for li, frame in enumerate(frames):
            i, ok, err = processor(*call_args(li, frame))
            n_success += int(ok)
            if not ok:
                logger.warning(f"  Frame {i} failed: {err}")
            _progress(li + 1)
    else:
        logger.info(f"Parallel prep: {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(processor, *call_args(li, frame))
                    for li, frame in enumerate(frames)]
            done = 0
            for fut in as_completed(futs):
                i, ok, err = fut.result()
                done += 1
                n_success += int(ok)
                if not ok:
                    logger.warning(f"  Frame {i} failed: {err}")
                _progress(done)

    record_timing(args.method, "prepare", "featurize",
                  time.perf_counter() - _t_feat, _tctx)
    logger.info(f"Done: {n_success}/{len(frames)} frames -> {args.output_dir}/  "
                f"({time.time()-t0:.0f}s)")

    if frames and n_success == 0:
        # Exiting non-zero stops a pipeline here instead of letting the next
        # stage train or evaluate against an empty directory.
        raise SystemExit(
            f"ERROR: every frame failed; {args.output_dir}/ has no data. See the "
            f"per-frame warnings above for the cause.")

    if not args.skip_meta:
        metadata = {
            "basis": args.basis, "charge": args.charge, "spin": args.spin,
            "n_frames": n_success, "input_file": os.path.basename(args.input),
            "e0s": {str(k): v for k, v in e0s.items()},
            "baseline_model": args.baseline_model,
            "baseline_basis": args.baseline_basis, "baseline_xc": args.baseline_xc,
        }
        if args.method == "orbital":
            metadata["localizer"] = args.localizer
        else:
            metadata["featurization"] = "invariant_density_matrix"
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  E0s saved to: {e0s_path}")


if __name__ == "__main__":
    main()
