#!/usr/bin/env python
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Convert the revised MD17 (rMD17) release into OrbACE extended XYZ.

Download the per-molecule NPZ files and the official index splits first, from
the FigShare record (DOI 10.6084/m9.figshare.12672038):

    https://api.figshare.com/v2/articles/12672038

Protocol
--------
rMD17 ships 100,000 conformers per molecule together with five predefined
1000-frame train/test index pairs. The MLIP literature this benchmark exists to
compare against (MACE, NequIP, Allegro, BOTNet, and NN-xTB) uses one of those
pairs and splits the 1000 training indices 950/50 into train/validation. Using
the shipped indices rather than resampling is what makes the numbers
comparable, so ``--split`` selects the pair and nothing is drawn at random.

Units
-----
rMD17 energies are kcal/mol and forces kcal/mol/Angstrom; both are converted to
eV and eV/Angstrom here.

Isolated atoms
--------------
The release ships no free-atom energies. Every conformer of a given molecule
has identical composition, so any E0 convention only shifts that molecule's
energies by a constant: it cannot change force errors, and it cancels in the
energy MAE as long as train and test share it. No IsolatedAtom frames are
written, which leaves ``prepare.py`` to compute E0s via HF once and reuse them
across the splits through ``--e0s_json``.

Example
-------
    python examples/benchmarks/rmd17/rmd17_to_extxyz.py \\
        --npz_dir bench_data/rmd17/raw --out_dir bench_data/rmd17 \\
        --split 01 --n_valid 50

writing bench_data/rmd17/<molecule>/extxyz/{train,valid,test}.xyz
"""
import argparse
import logging
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# examples/benchmarks/rmd17/ -> repo root is three levels up
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from examples.benchmarks.common.extxyz_io import (  # noqa: E402
    KCAL_PER_MOL_TO_EV, symbol, write_extxyz)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rmd17_to_extxyz")

MOLECULES = (
    "aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "paracetamol", "salicylic", "toluene", "uracil",
)


def _read_indices(path):
    """Read a rMD17 index CSV (one zero-based frame index per line)."""
    idx = np.loadtxt(path, dtype=np.int64, ndmin=1)
    return np.atleast_1d(idx)


def _frames(data, indices):
    symbols = [symbol(z) for z in data["nuclear_charges"]]
    coords = data["coords"]
    energies = data["energies"]
    forces = data["forces"]
    for i in indices:
        yield {
            "symbols": symbols,
            "coords": coords[i],
            "energy": float(energies[i]) * KCAL_PER_MOL_TO_EV,
            "forces": forces[i] * KCAL_PER_MOL_TO_EV,
        }


def convert_molecule(npz_dir, out_dir, molecule, split, n_valid):
    npz_path = os.path.join(npz_dir, f"rmd17_{molecule}.npz")
    if not os.path.isfile(npz_path):
        logger.warning("  %-14s SKIP (missing %s)", molecule,
                       os.path.basename(npz_path))
        return None

    train_idx = _read_indices(os.path.join(npz_dir, f"index_train_{split}.csv"))
    test_idx = _read_indices(os.path.join(npz_dir, f"index_test_{split}.csv"))
    if n_valid >= len(train_idx):
        raise SystemExit(
            f"--n_valid {n_valid} leaves no training frames out of "
            f"{len(train_idx)} indices in index_train_{split}.csv")

    # The shipped index order is already shuffled; take the validation slice
    # off the end so the training set stays a prefix of the official list.
    valid_idx = train_idx[len(train_idx) - n_valid:]
    fit_idx = train_idx[:len(train_idx) - n_valid]

    data = np.load(npz_path)
    n_frames = len(data["energies"])
    for name, idx in (("train", fit_idx), ("valid", valid_idx), ("test", test_idx)):
        if idx.max() >= n_frames:
            raise SystemExit(
                f"{molecule}: index {idx.max()} exceeds {n_frames} frames")

    mol_dir = os.path.join(out_dir, molecule, "extxyz")
    os.makedirs(mol_dir, exist_ok=True)
    counts = {}
    for name, idx in (("train", fit_idx), ("valid", valid_idx), ("test", test_idx)):
        path = os.path.join(mol_dir, f"{name}.xyz")
        counts[name] = write_extxyz(path, _frames(data, idx))

    logger.info("  %-14s %d atoms   train=%d valid=%d test=%d",
                molecule, len(data["nuclear_charges"]),
                counts["train"], counts["valid"], counts["test"])
    return counts


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz_dir", required=True,
                   help="directory holding rmd17_<molecule>.npz and index_*.csv")
    p.add_argument("--out_dir", required=True,
                   help="dataset root; each molecule is written to "
                        "<out_dir>/<molecule>/extxyz/")
    p.add_argument("--split", default="01", choices=("01", "02", "03", "04", "05"),
                   help="which shipped train/test index pair to use")
    p.add_argument("--n_valid", type=int, default=50,
                   help="validation frames held out of the 1000 training "
                        "indices (default 50, matching the MLIP protocol)")
    p.add_argument("--molecules", default=",".join(MOLECULES),
                   help="comma-separated subset of molecules to convert")
    args = p.parse_args()

    molecules = [m.strip() for m in args.molecules.split(",") if m.strip()]
    unknown = set(molecules) - set(MOLECULES)
    if unknown:
        raise SystemExit(f"unknown molecules: {sorted(unknown)}")

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("rMD17 split %s -> %s", args.split, args.out_dir)
    done = 0
    for molecule in molecules:
        if convert_molecule(args.npz_dir, args.out_dir, molecule,
                            args.split, args.n_valid):
            done += 1
    logger.info("converted %d/%d molecules", done, len(molecules))


if __name__ == "__main__":
    main()
