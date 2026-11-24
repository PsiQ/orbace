#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Farthest Point Sampling (FPS) to select diverse subsets.

Two modes:
  --mode orbital  (default for legacy)  Uses prepared NPZ orbital features.
  --mode geometry                       Uses extxyz geometry only (no HF/NPZ).

Usage (geometry mode):
  python fps_select.py \
    --mode geometry --extxyz ani_all.extxyz \
    --n_train 10000 --n_finetune 5000 --n_valid 1000 --seed 42

Usage (orbital mode — legacy OrbACE):
  python fps_select.py \
    --mode orbital --data_dir data/ --extxyz ani_all.extxyz \
    --n_train 10000 --n_finetune 5000 --n_valid 1000 \
    --train_dir data_train/ --finetune_dir data_finetune/ --seed 42

Outputs:
  - train.xyz, finetune.xyz, valid.xyz, test.xyz: extxyz subsets. test.xyz is the
    FULL remainder by default (any mode); cap it with --n_test N if desired.
  - (orbital mode only) data_train/, data_finetune/, data_test/: symlinked NPZ dirs.
"""

import argparse
import json
import logging
import os
import sys
import shutil

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# examples/ani1x/dataset/ -> repo root is three levels up
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fps_select")


_SYM2Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18,
}


def build_geometry_descriptors(extxyz_path):
    """Build per-frame descriptors from geometry only (no HF/NPZ needed).

    For each non-IsolatedAtom frame, computes:
      - normalized atom composition vector (max Z=18, so 19 bins)
      - sorted Coulomb-matrix eigenvalues (padded to max_atoms)
    Returns (descriptors array, frame_indices into the non-isolated ordering).
    """
    from orbace.data.extxyz import read_extxyz

    logger.info("Building geometry descriptors from %s ...", extxyz_path)
    frames = read_extxyz(extxyz_path)

    MAX_Z = 18
    comp_dim = MAX_Z + 1
    eigenvalue_dim = 0
    frame_data = []

    for frame in frames:
        if frame.get("config_type") == "IsolatedAtom":
            continue
        coords = frame["coords"]
        syms = frame["symbols"]
        Z = np.array([_SYM2Z.get(s, 0) for s in syms])
        n = len(Z)
        eigenvalue_dim = max(eigenvalue_dim, n)

        comp = np.bincount(Z, minlength=comp_dim)[:comp_dim].astype(np.float32)
        comp = comp / max(comp.sum(), 1.0)

        # Coulomb matrix (upper triangle filled with Z_i*Z_j / |r_i - r_j|)
        C = np.zeros((n, n))
        for i in range(n):
            C[i, i] = 0.5 * Z[i] ** 2.4
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                if d > 1e-10:
                    C[i, j] = C[j, i] = Z[i] * Z[j] / d
        eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
        frame_data.append((comp, eigvals))

    desc_list = []
    for comp, eigvals in frame_data:
        padded = np.zeros(eigenvalue_dim)
        padded[:len(eigvals)] = eigvals
        desc_list.append(np.concatenate([comp, padded]))

    descs = np.array(desc_list, dtype=np.float32)
    mu = descs.mean(axis=0, keepdims=True)
    sigma = descs.std(axis=0, keepdims=True) + 1e-8
    descs = (descs - mu) / sigma
    logger.info("  geometry descriptor shape: %s (%d frames)", descs.shape, len(descs))
    return descs


def build_descriptors(data_dir, n_frames):
    """Build a lightweight descriptor for each frame from the NPZ orbital features.

    For each frame, computes:
      - mean of scalar orbital features (10-dim)
      - std of scalar orbital features (10-dim)
      - atom composition vector (max Z=10 for HCNO: 4 elements)
    Total: 24-dim descriptor per frame.
    """
    logger.info(f"Building descriptors from {n_frames} frames...")
    MAX_Z = 10
    desc_list = []
    for i in range(n_frames):
        path = os.path.join(data_dir, f"frame_{i:06d}.npz")
        d = np.load(path)
        feat = d["orbital_features"]  # (n_orb, 10)
        mean_f = feat.mean(axis=0)
        std_f = feat.std(axis=0)
        z = d["atom_numbers"]
        comp = np.bincount(z, minlength=MAX_Z + 1)[:MAX_Z + 1].astype(np.float32)
        comp = comp / max(comp.sum(), 1.0)
        desc_list.append(np.concatenate([mean_f, std_f, comp]))
        if (i + 1) % 50000 == 0:
            logger.info(f"  {i+1}/{n_frames}")
    descs = np.array(desc_list, dtype=np.float32)
    # Normalize each column to zero mean, unit variance for balanced FPS distances.
    mu = descs.mean(axis=0, keepdims=True)
    sigma = descs.std(axis=0, keepdims=True) + 1e-8
    descs = (descs - mu) / sigma
    logger.info(f"  descriptor shape: {descs.shape}")
    return descs


def fps_greedy(descs, n_select, seed=42):
    """Greedy farthest point sampling.

    Returns the indices of n_select points that are maximally spread in
    descriptor space. O(n_select * N) with batched distance computation.
    """
    rng = np.random.RandomState(seed)
    N = descs.shape[0]
    selected = [rng.randint(N)]
    min_dist = np.full(N, np.inf, dtype=np.float64)

    for k in range(1, n_select):
        last = descs[selected[-1]]
        d = np.sum((descs - last[None, :]) ** 2, axis=1)
        np.minimum(min_dist, d, out=min_dist)
        min_dist[selected[-1]] = -1.0
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        if (k + 1) % 2000 == 0:
            logger.info(f"  FPS: {k+1}/{n_select}")

    return np.array(selected)


def write_subset_dir(data_dir, indices, out_dir, metadata):
    """Symlink selected frame NPZs into out_dir with contiguous naming."""
    os.makedirs(out_dir, exist_ok=True)
    for new_i, orig_i in enumerate(indices):
        src = os.path.abspath(os.path.join(data_dir, f"frame_{orig_i:06d}.npz"))
        dst = os.path.join(out_dir, f"frame_{new_i:06d}.npz")
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    meta = dict(metadata)
    meta["n_frames"] = len(indices)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    e0s_src = os.path.join(data_dir, "e0s.json")
    if os.path.exists(e0s_src):
        shutil.copy2(e0s_src, os.path.join(out_dir, "e0s.json"))
    logger.info(f"  {out_dir}: {len(indices)} frames (symlinked)")


def write_extxyz_subset(extxyz_path, training_indices, out_path):
    """Extract specific training-frame indices from the extxyz.

    Streams the file, skipping IsolatedAtom frames (they don't count toward the
    training index), and writes selected frames plus ALL IsolatedAtom frames
    at the top of the output.
    """
    idx_set = set(training_indices)
    isolated_blocks = []
    written = 0
    train_idx = 0

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
                if written == 0:
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
            train_idx += 1

    if written > 0:
        fout.close()
    logger.info(f"  {out_path}: {written} frames")


def main():
    p = argparse.ArgumentParser(description="FPS subset selection")
    p.add_argument("--mode", choices=["geometry", "orbital"], default="geometry",
                   help="Descriptor mode: 'geometry' (no NPZ) or 'orbital' (legacy, needs NPZ)")
    p.add_argument("--data_dir", default="data", help="Directory of prepared frame_*.npz (orbital mode)")
    p.add_argument("--extxyz", default="ani_all.extxyz", help="extxyz file")
    p.add_argument("--n_train", type=int, default=10000)
    p.add_argument("--n_finetune", type=int, default=5000)
    p.add_argument("--n_valid", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=0,
                   help="Optional cap on the held-out test set size (sampled from "
                        "the remainder), for any mode; <=0 (default) keeps ALL "
                        "remaining frames as the test set")
    p.add_argument("--train_dir", default="data_train")
    p.add_argument("--finetune_dir", default="data_finetune")
    p.add_argument("--test_dir", default="data_test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "geometry":
        descs = build_geometry_descriptors(args.extxyz)
        n_total = descs.shape[0]
        metadata = None
    else:
        meta_path = os.path.join(args.data_dir, "metadata.json")
        with open(meta_path) as f:
            metadata = json.load(f)
        n_total = metadata["n_frames"]
        descs = build_descriptors(args.data_dir, n_total)

    n_selected = args.n_train + args.n_finetune + args.n_valid
    if n_selected > n_total:
        logger.error(f"Requested {n_selected} frames but only {n_total} available")
        sys.exit(1)

    logger.info(f"Running FPS to select {n_selected} diverse frames for train/finetune/valid...")
    selected = fps_greedy(descs, n_selected, seed=args.seed)

    i = 0
    train_idx = sorted(selected[i: i + args.n_train]); i += args.n_train
    finetune_idx = sorted(selected[i: i + args.n_finetune]); i += args.n_finetune
    valid_idx = sorted(selected[i: i + args.n_valid]); i += args.n_valid

    used = set(train_idx) | set(finetune_idx) | set(valid_idx)
    test_idx = [i for i in range(n_total) if i not in used]

    # By default the test set is the FULL remainder (everything not used for
    # train/finetune/valid), for any dataset/mode. Capping is strictly opt-in
    # via --n_test > 0 (useful only when downstream eval re-featurizes from
    # test.xyz and the remainder is impractically large).
    if args.n_test > 0 and len(test_idx) > args.n_test:
        rng_test = np.random.RandomState(args.seed + 1)
        test_idx = sorted(rng_test.choice(test_idx, size=args.n_test, replace=False).tolist())
        logger.info(f"  Capped test set to {args.n_test} randomly sampled remainder frames")
    else:
        test_idx = sorted(test_idx)

    logger.info(f"Selected: train={len(train_idx)}, finetune={len(finetune_idx)}, "
                f"valid={len(valid_idx)}, test={len(test_idx)}")

    if args.mode == "orbital" and metadata is not None:
        write_subset_dir(args.data_dir, train_idx, args.train_dir, metadata)
        write_subset_dir(args.data_dir, finetune_idx, args.finetune_dir, metadata)
        write_subset_dir(args.data_dir, test_idx, args.test_dir, metadata)

    logger.info("Extracting extxyz subsets...")
    write_extxyz_subset(args.extxyz, train_idx, "train.xyz")
    write_extxyz_subset(args.extxyz, finetune_idx, "finetune.xyz")
    write_extxyz_subset(args.extxyz, valid_idx, "valid.xyz")

    # Always materialize test.xyz = the (full) remaining set, for any dataset.
    # In orbital mode the fast eval path is still --npz_data <test_dir>/ (reuses
    # precomputed features); test.xyz is provided for inspection / from-scratch
    # eval and is 1:1 with the test_dir NPZ subset.
    write_extxyz_subset(args.extxyz, test_idx, "test.xyz")
    logger.info("  Wrote test.xyz (%d frames)", len(test_idx))
    if args.mode == "orbital":
        logger.info("  (fast eval: use --npz_data %s/ to reuse features)", args.test_dir)

    np.savez("fps_indices.npz",
             train=np.array(train_idx),
             finetune=np.array(finetune_idx),
             valid=np.array(valid_idx),
             test=np.array(test_idx))
    logger.info("Saved fps_indices.npz")
    logger.info("Done.")


if __name__ == "__main__":
    main()
