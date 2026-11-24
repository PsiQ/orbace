#!/usr/bin/env python
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Normalise the 3BPA dataset into OrbACE extended XYZ.

Download the release first:

    base=https://raw.githubusercontent.com/davkovacs/BOTNet-datasets/main/dataset_3BPA
    for f in iso_atoms train_300K test_300K test_600K test_1200K; do
        curl -sSLO "$base/$f.xyz"
    done

The files are already extended XYZ in eV with an ``energy=`` key and a
``forces`` column, which ``orbace.data.extxyz`` reads natively. Two things still
need doing, and both matter:

1. ``iso_atoms.xyz`` carries the free-atom energies but is not tagged, so the
   E0 machinery ignores it. Its four frames are re-emitted as
   ``config_type=IsolatedAtom`` and prepended to every split, which pins all
   splits to the same E0 convention.
2. The training frames carry a ``dihedrals`` field that is irrelevant here and
   is dropped.

Protocol
--------
The standard benchmark trains on the 500 configurations sampled at 300 K and
evaluates at 300 K, 600 K and 1200 K. The elevated-temperature sets are the
point of the benchmark: they broaden the conformational ensemble well past the
training distribution, so they measure extrapolation rather than fit quality.
A validation split is carved out of the 500 training frames, since the release
ships none.

Example
-------
    python examples/benchmarks/3bpa/3bpa_to_extxyz.py \\
        --src_dir bench_data/3bpa --out_dir bench_data/3bpa/extxyz --n_valid 50
"""
import argparse
import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# examples/benchmarks/3bpa/ -> repo root is three levels up
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from examples.benchmarks.common.extxyz_io import write_extxyz  # noqa: E402
from orbace.data.extxyz import read_extxyz  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("3bpa_to_extxyz")

TEST_SETS = ("test_300K", "test_600K", "test_1200K")


def _load_isolated(path):
    """Free-atom energies in eV, keyed by element symbol."""
    e0s = {}
    for frame in read_extxyz(path):
        if len(frame["symbols"]) != 1:
            logger.warning("  iso_atoms frame has %d atoms - skipping",
                           len(frame["symbols"]))
            continue
        if frame.get("energy") is None:
            logger.warning("  iso_atoms frame for %s has no energy - skipping",
                           frame["symbols"][0])
            continue
        e0s[frame["symbols"][0]] = float(frame["energy"])
    return e0s


def _as_frames(path):
    for frame in read_extxyz(path):
        if frame.get("config_type") == "IsolatedAtom":
            continue
        yield {
            "symbols": frame["symbols"],
            "coords": frame["coords"],
            "energy": frame["energy"],
            "forces": frame["forces"],
        }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_dir", required=True,
                   help="directory holding the downloaded 3BPA .xyz files")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_valid", type=int, default=50,
                   help="validation frames carved out of the 500 training "
                        "configurations (default 50)")
    args = p.parse_args()

    iso_path = os.path.join(args.src_dir, "iso_atoms.xyz")
    if not os.path.isfile(iso_path):
        raise SystemExit(f"missing {iso_path}")
    e0s = _load_isolated(iso_path)
    logger.info("E0s from iso_atoms.xyz: %s",
                ", ".join(f"{k}={v:.4f} eV" for k, v in sorted(e0s.items())))

    os.makedirs(args.out_dir, exist_ok=True)

    train_all = list(_as_frames(os.path.join(args.src_dir, "train_300K.xyz")))
    if args.n_valid >= len(train_all):
        raise SystemExit(f"--n_valid {args.n_valid} leaves no training frames "
                         f"out of {len(train_all)}")
    split_at = len(train_all) - args.n_valid
    n_train = write_extxyz(os.path.join(args.out_dir, "train.xyz"),
                           train_all[:split_at], isolated=e0s)
    n_valid = write_extxyz(os.path.join(args.out_dir, "valid.xyz"),
                           train_all[split_at:], isolated=e0s)
    logger.info("  train.xyz  %d frames", n_train)
    logger.info("  valid.xyz  %d frames", n_valid)

    for name in TEST_SETS:
        src = os.path.join(args.src_dir, f"{name}.xyz")
        if not os.path.isfile(src):
            logger.warning("  %s SKIP (missing)", name)
            continue
        n = write_extxyz(os.path.join(args.out_dir, f"{name}.xyz"),
                         _as_frames(src), isolated=e0s)
        logger.info("  %-14s %d frames", f"{name}.xyz", n)


if __name__ == "__main__":
    main()
