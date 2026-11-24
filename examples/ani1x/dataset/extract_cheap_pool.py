#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Extract unlabeled ANI-1x geometries for the cheap level of a multi-fidelity run.

The multi-fidelity arms learn the expensive level as a correction to, or
alongside, a cheap one, and nothing requires the two to cover the same frames.
To measure what extra cheap data is worth at a fixed expensive-label budget we
need geometries that carry NO reference energy -- and, critically, that the
442k-frame ANI test split has never seen.

``ani1x-release.h5`` holds ~4.96M conformers and only ~459k were exported into
the pool extxyz from which train/valid/finetune/test were cut, so the remainder
is exactly the pool we want. Rather than reproduce the
exporter's ``--stride``/``--seed`` plan and trust that it was reconstructed
correctly, this excludes by GEOMETRY: every exported frame is hashed on its
rounded coordinates and element list, and any candidate hitting that set is
skipped. Getting the original flags wrong then cannot silently leak a test
geometry into training; the exclusion is checked against the data itself.

Frames are written with positions only. ``prepare.py`` stores ``energy`` /
``forces`` solely for frames whose source carried them, so the result is a
cheap-only prep that trains the baseline head and nothing else.

Usage:
    python extract_cheap_pool.py --h5 ani1x-release.h5 \
        --exclude ani_all.extxyz \
        --out cheap_pool.xyz --n_frames 100000
"""

import argparse
import hashlib
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_cheap_pool")

_Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O"}
# Coordinates are float32 in the h5 and were written with %15.8f, so both sides
# round to the same 4 decimals for the same conformer. Coarser than the data
# and far finer than the spacing between distinct conformers.
_ROUND = 4


def _hash(symbols, coords) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(",".join(symbols).encode())
    h.update(np.round(np.asarray(coords, dtype=np.float64), _ROUND).tobytes())
    return h.digest()


def exported_hashes(path: str) -> set:
    """Hash every frame of the extxyz the released splits were cut from."""
    seen = set()
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            n = int(line.split()[0])
            fh.readline()                       # comment / properties
            syms, xyz = [], []
            for _ in range(n):
                parts = fh.readline().split()
                syms.append(parts[0])
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            seen.add(_hash(syms, xyz))
            if len(seen) % 100_000 == 0:
                logger.info("  hashed %s exported frames", f"{len(seen):,}")
    return seen


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True)
    p.add_argument("--exclude", required=True,
                   help="extxyz whose geometries must NOT appear in the pool")
    p.add_argument("--out", required=True)
    p.add_argument("--n_frames", type=int, default=100_000)
    p.add_argument("--max_atoms", type=int, default=0,
                   help="skip molecules larger than this (0 = no limit)")
    p.add_argument("--seed", type=int, default=1234,
                   help="shuffles molecule order so the pool is not all one size")
    p.add_argument("--lattice", type=float, default=7.0)
    a = p.parse_args()

    import h5py

    logger.info("Hashing the exported frames in %s ...", a.exclude)
    excluded = exported_hashes(a.exclude)
    logger.info("Exclusion set: %s geometries", f"{len(excluded):,}")

    rng = np.random.default_rng(a.seed)
    written = skipped = scanned = 0
    with h5py.File(a.h5, "r") as f, open(a.out, "w") as out:
        groups = sorted(f.keys())
        keep, syms_of = [], {}
        for gi, name in enumerate(groups):
            z = np.asarray(f[name]["atomic_numbers"])
            if a.max_atoms and len(z) > a.max_atoms:
                continue
            if any(int(v) not in _Z_TO_SYMBOL for v in z):
                continue
            syms_of[gi] = [_Z_TO_SYMBOL[int(v)] for v in z]
            n_conf = f[name]["coordinates"].shape[0]
            keep.append(np.stack([np.full(n_conf, gi, np.int32),
                                  np.arange(n_conf, dtype=np.int32)], 1))
        cand = np.concatenate(keep)
        logger.info("Candidate conformers under --max_atoms %d: %s",
                    a.max_atoms, f"{len(cand):,}")

        # Uniform over CONFORMERS, not molecules. Molecules differ hugely in
        # conformer count, so drawing a molecule first would over-weight the
        # large ones and hand the cheap pool a size distribution the training
        # set does not have -- and, for the orbital arm, a much steeper SCF bill.
        order = rng.permutation(len(cand))
        # Read in molecule order so the h5 is traversed sequentially; the draw
        # above already decided membership, this only fixes visit order.
        take = cand[order[:int(a.n_frames * 1.3) + 1000]]
        take = take[np.argsort(take[:, 0], kind="stable")]

        for gi, ci in take:
            if written >= a.n_frames:
                break
            syms = syms_of[int(gi)]
            c = f[groups[int(gi)]]["coordinates"][int(ci)]
            scanned += 1
            if _hash(syms, c) in excluded:
                skipped += 1
                continue
            out.write(f"{len(syms)}\n")
            out.write(
                f'Lattice="{a.lattice} 0.0 0.0 0.0 {a.lattice} 0.0 0.0 0.0 '
                f'{a.lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
            for s, xyz in zip(syms, c):
                out.write(f"{s:<2s} {xyz[0]:15.8f} {xyz[1]:15.8f} {xyz[2]:15.8f}\n")
            written += 1
            if written % 20_000 == 0:
                logger.info("  wrote %s / %s", f"{written:,}", f"{a.n_frames:,}")

    logger.info("Done: %s cheap-only frames -> %s", f"{written:,}", a.out)
    logger.info("Scanned %s conformers; %s skipped as already exported (%.1f%%).",
                f"{scanned:,}", f"{skipped:,}", 100 * skipped / max(scanned, 1))
    if skipped == 0:
        logger.warning(
            "NOTHING was excluded. Either the h5 and the extxyz disagree on "
            "coordinate precision, or --exclude is not the file the released "
            "splits were cut from. Both would let test geometries into "
            "training, so check before preparing this pool.")
        sys.exit(1)
    if written < a.n_frames:
        logger.warning("Pool is short: wanted %s, got %s.",
                       f"{a.n_frames:,}", f"{written:,}")


if __name__ == "__main__":
    main()
