#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Extract the held-out (non-FPS-selected) frames into a flat extxyz file.

Reads the `test` index array saved in fps_indices.npz (the complement of the
train/finetune/valid FPS selection) and streams the source extxyz, writing the
selected frames plus ALL IsolatedAtom frames at the top.

Usage:
  python extract_test_xyz.py \
    --extxyz ani_all.extxyz \
    --indices fps_indices.npz --key test \
    --out test.xyz
"""

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_test_xyz")


def write_extxyz_subset(extxyz_path, training_indices, out_path, log_every=50000):
    """Stream extxyz, skip IsolatedAtom frames for indexing, write selected frames.

    IsolatedAtom frames are emitted at the top of the output (they carry the E0
    references) and do not advance the training index counter.
    """
    idx_set = set(int(i) for i in training_indices)
    isolated_blocks = []
    written = 0
    train_idx = 0
    fout = None

    with open(extxyz_path) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            natoms = int(s)
            comment = fin.readline()
            atom_lines = [fin.readline() for _ in range(natoms)]

            if "IsolatedAtom" in comment:
                isolated_blocks.append((line, comment, atom_lines))
                continue

            if train_idx in idx_set:
                if fout is None:
                    fout = open(out_path, "w")
                    for iso_line, iso_comment, iso_atoms in isolated_blocks:
                        fout.write(iso_line)
                        fout.write(iso_comment)
                        for al in iso_atoms:
                            fout.write(al)
                fout.write(line)
                fout.write(comment)
                for al in atom_lines:
                    fout.write(al)
                written += 1
                if written % log_every == 0:
                    logger.info("  written %d / %d", written, len(idx_set))
            train_idx += 1

    if fout is not None:
        fout.close()
    logger.info("%s: %d frames (scanned %d non-isolated, %d isolated-atom blocks)",
                out_path, written, train_idx, len(isolated_blocks))
    return written


def main():
    p = argparse.ArgumentParser(description="Extract held-out frames into extxyz")
    p.add_argument("--extxyz", default="ani_all.extxyz")
    p.add_argument("--indices", default="fps_indices.npz")
    p.add_argument("--key", default="test", help="array name inside the npz")
    p.add_argument("--out", default="test.xyz")
    args = p.parse_args()

    idx = np.load(args.indices)[args.key]
    logger.info("Loaded %d '%s' indices from %s (min=%d, max=%d)",
                len(idx), args.key, args.indices, idx.min(), idx.max())
    write_extxyz_subset(args.extxyz, idx, args.out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
