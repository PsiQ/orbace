#!/usr/bin/env python
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Convert the official ANI-1x HDF5 release into OrbACE extended XYZ.

Download the source file first (5.6 GB) from the FigShare collection linked at
https://github.com/aiqm/ANI1x_datasets:

    curl -L -o ani1x-release.h5 https://ndownloader.figshare.com/files/18112775
    # md5 98090dd6679106da861f52bed825ffb7

Units
-----
Table 1 of the data paper (Smith et al., Sci. Data 7, 134 (2020)) defines the
layout, and the entry worth stating explicitly is the forces:

    coordinates              Angstrom
    wb97x_{dz,tz}.energy     Hartree
    wb97x_{dz,tz}.forces     Hartree / ANGSTROM      <-- not Hartree/Bohr
    ccsd(t)_cbs.energy       Hartree

So energies and forces take the *same* Hartree->eV factor here. Converting the
forces with the Hartree/Bohr factor instead would inflate them by 1.8897, and no
training metric would ever reveal it: the labels stay self-consistent with each
other and disagree only with physics. Always run

    python examples/ani1x/dataset/check_force_consistency.py --input <out.xyz> --check grad fd

on the result. For ``--level dz`` that check recomputes the gradient at
wB97x/6-31G*, which is exactly the level of theory of the labels, so a correct
conversion returns a ratio of 1.00.

Levels of theory
----------------
    dz -> wB97x/6-31G*      4,956,005 conformers (all of ANI-1x)
    tz -> wB97x/def2-TZVPP  same geometries, a subset of conformers

Restricting to the CCSD(T) subset
---------------------------------
``--require 'ccsd(t)_cbs.energy'`` keeps only the 489,571 conformers that also
carry a CCSD(T)*/CBS energy -- the ANI-1ccx geometries. Those conformers are
then automatically carried into the output comment line as ``CCSDT_energy``
(eV), so the same files can later drive CCSD(T) delta-learning on identical
geometries without re-joining against the HDF5.

Isolated-atom references
------------------------
The HDF5 ships no free-atom energies, so E0s are computed here with spin-
polarised DFT at the matching level (H doublet, C triplet, N quartet, O
triplet) and written as ``config_type=IsolatedAtom`` frames at the top of every
output file, which is where ``scripts/prepare.py`` looks for them.

Examples
--------
    # the ANI-1ccx geometries with DFT labels, split for a campaign
    python examples/ani1x/dataset/ani1x_to_extxyz.py --h5 ani1x-release.h5 --out_dir data_clean \\
        --level dz --require 'ccsd(t)_cbs.energy' \\
        --splits train:50000,valid:5000,finetune:10000,test:25000

    # one file, small molecules only, for a quick label check
    python examples/ani1x/dataset/ani1x_to_extxyz.py --h5 ani1x-release.h5 \\
        --out small.xyz --max_frames 40 --max_atoms 8
"""
import argparse
import json
import logging
import os
import re
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ani1x")

HARTREE_TO_EV = 27.211386245988

_Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O"}

# 2S for the neutral ground-state atom, used for the isolated-atom references.
_ATOM_SPIN = {1: 1, 6: 2, 7: 3, 8: 2}

_LEVELS = {
    "dz": {"key": "wb97x_dz", "xc": "wb97x", "basis": "6-31g*",
           "label": "wB97x/6-31G*"},
    "tz": {"key": "wb97x_tz", "xc": "wb97x", "basis": "def2-tzvpp",
           "label": "wB97x/def2-TZVPP"},
}


def _tag(key):
    """HDF5 key -> a comment-line field name the extxyz parser can handle."""
    if key == "ccsd(t)_cbs.energy":
        return "CCSDT_energy"
    return re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")


def compute_e0s(elements, xc, basis):
    """Spin-polarised free-atom energies in eV, at the label level of theory."""
    from pyscf import dft, gto

    e0s = {}
    for z in sorted(elements):
        spin = _ATOM_SPIN[z]
        mol = gto.M(atom=[(int(z), (0.0, 0.0, 0.0))], basis=basis, spin=spin,
                    charge=0, verbose=0)
        mf = dft.UKS(mol)
        mf.xc = xc
        mf.conv_tol = 1e-10
        mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            mf.kernel()
        if not mf.converged:
            logger.error("Free-atom SCF failed for Z=%d; cannot write E0.", z)
            sys.exit(1)
        e0s[int(z)] = float(mf.e_tot) * HARTREE_TO_EV
        logger.info("  E0 %-2s (Z=%d, 2S=%d) = %.6f eV",
                    _Z_TO_SYMBOL[z], z, spin, e0s[int(z)])
    return e0s


def _valid_mask(grp, scalar_keys):
    """Conformers whose every required per-conformer scalar is present.

    ANI-1x stores the more expensive levels of theory sparsely and pads the rest
    with NaN, so this is what restricts the output to e.g. the CCSD(T) subset.
    """
    n = grp["coordinates"].shape[0]
    mask = np.ones(n, dtype=bool)
    for k in scalar_keys:
        v = np.asarray(grp[k][()], dtype=np.float64).reshape(n, -1)
        mask &= ~np.isnan(v).any(axis=1)
    return mask


def survey(h5path, level_key, scalar_keys, max_atoms):
    """One cheap metadata pass: per-molecule valid conformer indices."""
    import h5py

    names, valid, natoms, elements = [], [], [], set()
    with h5py.File(h5path, "r") as f:
        for name, grp in f.items():
            if any(k not in grp for k in scalar_keys):
                continue
            if f"{level_key}.forces" not in grp:
                continue
            z = np.asarray(grp["atomic_numbers"][()], dtype=np.int64)
            if max_atoms and len(z) > max_atoms:
                continue
            idx = np.flatnonzero(_valid_mask(grp, scalar_keys))
            if len(idx) == 0:
                continue
            names.append(name)
            valid.append(idx.astype(np.int32))
            natoms.append(len(z))
            elements.update(int(v) for v in z)
    return names, valid, np.array(natoms), elements


def build_plan(valid, targets, mode, max_per_molecule, stride, rng):
    """Decide which (molecule, conformer) pair goes to which split.

    ``molecule`` mode keeps every conformer of a molecule inside a single split,
    so the test set measures generalisation to unseen molecules. ``conformer``
    mode samples conformers independently: the conventional ANI-1x split, but
    other conformers of a test molecule may appear in training, which makes it
    easier and not a transferability measurement.
    """
    mol_ids, local_ids = [], []
    for m, idx in enumerate(valid):
        pool = idx[::max(1, stride)]
        if max_per_molecule and len(pool) > max_per_molecule:
            pool = rng.choice(pool, size=max_per_molecule, replace=False)
        mol_ids.append(np.full(len(pool), m, dtype=np.int32))
        local_ids.append(np.asarray(pool, dtype=np.int32))
    mol_ids = np.concatenate(mol_ids) if mol_ids else np.empty(0, np.int32)
    local_ids = np.concatenate(local_ids) if local_ids else np.empty(0, np.int32)
    n_pool = len(mol_ids)

    total = sum(targets.values())
    if total > n_pool:
        logger.error("Requested %d frames but only %d are available under the "
                     "current --stride/--max_per_molecule/--max_atoms caps.",
                     total, n_pool)
        sys.exit(1)

    if mode == "conformer":
        order = rng.permutation(n_pool)
    else:
        by_mol = np.argsort(mol_ids, kind="stable")
        bounds = np.searchsorted(mol_ids[by_mol], np.arange(len(valid) + 1))
        order = np.concatenate(
            [by_mol[bounds[m]:bounds[m + 1]] for m in rng.permutation(len(valid))])

    plan, assigned, cursor = {}, {}, 0
    for split, want in targets.items():
        take = order[cursor:cursor + want]
        cursor += want
        assigned[split] = len(take)
        for gi in take:
            plan.setdefault(int(mol_ids[gi]), []).append(
                (int(local_ids[gi]), split))
    return plan, assigned, n_pool


def _write_frame(fh, symbols, coords, forces, energy, lattice,
                 config_type=None, extra=None):
    fh.write(f"{len(symbols)}\n")
    cmt = (f'Lattice="{lattice} 0.0 0.0 0.0 {lattice} 0.0 0.0 0.0 {lattice}" '
           f'Properties=species:S:1:pos:R:3:REF_forces:R:3')
    if config_type:
        cmt += f" config_type={config_type}"
    cmt += f" REF_energy={energy!r}"
    for k, v in (extra or {}).items():
        cmt += f" {k}={v!r}"
    cmt += ' pbc="T T T"'
    fh.write(cmt + "\n")
    for s, c, fo in zip(symbols, coords, forces):
        fh.write(f"{s:<2s} {c[0]:15.8f} {c[1]:15.8f} {c[2]:15.8f} "
                 f"{fo[0]:15.8f} {fo[1]:15.8f} {fo[2]:15.8f}\n")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True, help="ani1x-release.h5")
    p.add_argument("--out", default=None,
                   help="single output path (use --splits/--out_dir instead to "
                        "emit a whole train/valid/test set)")
    p.add_argument("--out_dir", default=None, help="output directory for --splits")
    p.add_argument("--splits", default=None,
                   help="comma-separated name:count list, e.g. "
                        "'train:50000,valid:5000,test:25000'. Each split is "
                        "written to <name>.xyz with the IsolatedAtom E0 frames "
                        "prepended, and splits share no conformer.")
    p.add_argument("--split_mode", default="conformer",
                   choices=("conformer", "molecule"),
                   help="'conformer' samples conformers independently (the "
                        "conventional ANI-1x split); 'molecule' keeps each "
                        "molecule wholly within one split, so the test set "
                        "measures transfer to unseen molecules")
    p.add_argument("--level", default="dz", choices=sorted(_LEVELS),
                   help="label level of theory (default: dz = wB97x/6-31G*)")
    p.add_argument("--require", default="",
                   help="comma-separated extra HDF5 scalar keys that must be "
                        "present; e.g. 'ccsd(t)_cbs.energy' restricts to the "
                        "ANI-1ccx subset. Required keys are also carried into "
                        "the output comment line (in eV).")
    p.add_argument("--max_frames", type=int, default=0,
                   help="single-file mode: stop after this many conformers")
    p.add_argument("--max_per_molecule", type=int, default=0,
                   help="cap conformers taken from one molecule (0 = no cap); "
                        "a small cap buys chemical diversity per unit of "
                        "downstream SCF cost")
    p.add_argument("--stride", type=int, default=1,
                   help="take every Nth valid conformer within a molecule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_atoms", type=int, default=0,
                   help="skip molecules larger than this (0 = no limit)")
    p.add_argument("--e0s_json", default=None,
                   help="reuse free-atom E0s (eV) instead of computing them")
    p.add_argument("--lattice", type=float, default=7.0,
                   help="cubic box edge written to the Lattice field")
    args = p.parse_args()

    if not os.path.exists(args.h5):
        logger.error("HDF5 not found: %s", args.h5)
        sys.exit(2)
    if bool(args.splits) == bool(args.out):
        logger.error("Give exactly one of --out (single file) or --splits.")
        sys.exit(2)
    if args.splits and not args.out_dir:
        logger.error("--splits requires --out_dir.")
        sys.exit(2) 

    lvl = _LEVELS[args.level]
    ekey, fkey = f"{lvl['key']}.energy", f"{lvl['key']}.forces"
    extra_keys = [k.strip() for k in args.require.split(",") if k.strip()]
    scalar_keys = [ekey] + extra_keys

    logger.info("Source : %s", args.h5)
    logger.info("Level  : %s  (%s, %s)", lvl["label"], ekey, fkey)
    logger.info("Units  : energy Ha -> eV, forces Ha/Angstrom -> eV/Angstrom "
                "(both x %.6f)", HARTREE_TO_EV)
    if extra_keys:
        logger.info("Require: %s (also carried as %s)",
                    ", ".join(extra_keys),
                    ", ".join(_tag(k) for k in extra_keys))

    rng = np.random.default_rng(args.seed)

    logger.info("Surveying source (metadata pass) ...")
    names, valid, natoms, elements = survey(args.h5, lvl["key"], scalar_keys,
                                            args.max_atoms)
    n_avail = int(sum(len(v) for v in valid))
    unknown = elements - set(_Z_TO_SYMBOL)
    if unknown:
        logger.error("Unsupported element(s) Z=%s in source.", sorted(unknown))
        sys.exit(1)
    logger.info("  %d molecules, %d eligible conformers, elements %s",
                len(names), n_avail,
                sorted(_Z_TO_SYMBOL[z] for z in elements))

    if args.splits:
        targets = {}
        for part in args.splits.split(","):
            nm, _, cnt = part.partition(":")
            targets[nm.strip()] = int(cnt)
    else:
        targets = {"__single__": args.max_frames or n_avail}

    plan, assigned, n_pool = build_plan(
        valid, targets, args.split_mode, args.max_per_molecule, args.stride, rng)
    logger.info("Plan (%s split of a %d-conformer pool): %s",
                args.split_mode, n_pool,
                ", ".join(f"{k}={v}" for k, v in assigned.items()))

    if args.e0s_json:
        with open(args.e0s_json) as fh:
            e0s = {int(k): float(v) for k, v in json.load(fh).items()}
        logger.info("Reusing E0s from %s", args.e0s_json)
        if elements - set(e0s):
            logger.error("E0 JSON lacks Z=%s", sorted(elements - set(e0s)))
            sys.exit(1)
    else:
        logger.info("Computing free-atom E0s at %s (spin-polarised) ...",
                    lvl["label"])
        e0s = compute_e0s(elements, lvl["xc"], lvl["basis"])

    # Open one handle per split and prepend the E0 frames to each.
    if args.splits:
        os.makedirs(args.out_dir, exist_ok=True)
        paths = {s: os.path.join(args.out_dir, f"{s}.xyz") for s in targets}
    else:
        paths = {"__single__": args.out}
    handles = {s: open(pth, "w") for s, pth in paths.items()}
    for fh in handles.values():
        for z in sorted(elements):
            _write_frame(fh, [_Z_TO_SYMBOL[z]], np.zeros((1, 3)),
                         np.zeros((1, 3)), e0s[z], args.lattice,
                         config_type="IsolatedAtom")

    stats = {s: {"n": 0, "fsq": 0.0, "nf": 0, "fmax": 0.0, "na": []}
             for s in targets}
    n_dropped = 0

    n_target = sum(assigned.values())
    n_done = 0
    t_start = time.time()

    try:
        import h5py
        with h5py.File(args.h5, "r") as f:
            for m, name in enumerate(names):
                if m not in plan:
                    continue
                grp = f[name]
                z = np.asarray(grp["atomic_numbers"][()], dtype=np.int64)
                symbols = [_Z_TO_SYMBOL[int(v)] for v in z]
                wanted = plan[m]
                rows = sorted({j for j, _ in wanted})
                coords = np.asarray(grp["coordinates"][rows], dtype=np.float64)
                energy = np.asarray(grp[ekey][rows], dtype=np.float64)
                forces = np.asarray(grp[fkey][rows], dtype=np.float64)
                carried = {k: np.asarray(grp[k][rows], dtype=np.float64)
                           for k in extra_keys}
                pos = {j: i for i, j in enumerate(rows)}

                for j, split in wanted:
                    i = pos[j]
                    fo = forces[i] * HARTREE_TO_EV
                    if not (np.isfinite(fo).all() and np.isfinite(coords[i]).all()
                            and np.isfinite(energy[i])):
                        n_dropped += 1
                        continue
                    extra = {_tag(k): float(carried[k][i]) * HARTREE_TO_EV
                             for k in extra_keys}
                    _write_frame(handles[split], symbols, coords[i], fo,
                                 float(energy[i]) * HARTREE_TO_EV,
                                 args.lattice, extra=extra)
                    st = stats[split]
                    st["n"] += 1
                    st["fsq"] += float(np.sum(fo ** 2))
                    st["nf"] += fo.size
                    st["fmax"] = max(st["fmax"], float(np.abs(fo).max()))
                    st["na"].append(len(z))
                    n_done += 1

                if n_done // 25000 != (n_done - len(wanted)) // 25000:
                    dt = time.time() - t_start
                    logger.info("  %d/%d frames (%.0fs, %.0f frames/s)",
                                n_done, n_target, dt, n_done / max(dt, 1e-9))
    finally:
        for fh in handles.values():
            fh.close()

    e0_path = (os.path.join(args.out_dir, "e0s.json") if args.splits
               else os.path.splitext(args.out)[0] + "_e0s.json")
    with open(e0_path, "w") as fh:
        json.dump({str(k): v for k, v in e0s.items()}, fh, indent=2)

    logger.info("Done. E0s -> %s", e0_path)
    if n_dropped:
        logger.warning("Dropped %d conformer(s) with non-finite values.", n_dropped)
    worst_rms = None
    for s, st in stats.items():
        if st["n"] == 0:
            continue
        rms = float(np.sqrt(st["fsq"] / max(st["nf"], 1)))
        worst_rms = rms if worst_rms is None else worst_rms
        na = np.array(st["na"])
        logger.info("  %-10s %7d frames | atoms %d/%d/%d (min/med/max) | "
                    "force RMS %.4f eV/A (max |F| %.2f) -> %s",
                    s, st["n"], na.min(), int(np.median(na)), na.max(),
                    rms, st["fmax"], paths[s])
    if worst_rms is not None and not 0.5 <= worst_rms <= 6.0:
        logger.warning(
            "Force RMS is outside the 0.5-6 eV/Angstrom range typical of "
            "ANI-1x sampling. Check the unit convention before spending SCF "
            "time on this dataset.")
    first = next(iter(paths.values()))
    logger.info("Verify the labels before preparing features:")
    logger.info("  python examples/ani1x/dataset/check_force_consistency.py --input %s "
                "--check scale grad fd%s", first,
                "" if args.level == "dz" else f" --basis {lvl['basis']}")


if __name__ == "__main__":
    main()
