#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Split a large extxyz into fixed-size chunk files for parallel data prep.

This is the one-time front half of a chunked, multi-job data-preparation
workflow:

  1. Stream the (possibly huge) input extxyz frame-by-frame WITHOUT loading it
     all into memory, writing the training frames verbatim into
     ``chunks/chunk_XXXXX.extxyz`` files of ``--chunk_size`` frames each.
  2. Collect the element set and any ``config_type=IsolatedAtom`` frames while
     streaming, then resolve the isolated-atom energies (E0) ONCE and write the
     shared ``<data_dir>/e0s.json`` + ``<data_dir>/metadata.json``.
  3. Write ``chunks/manifest.json`` describing the split.

Each chunk is then processed by an independent job (a SLURM array task) that
runs ``prepare.py`` on one chunk file with ``--e0s_json <data_dir>/e0s.json
--skip_meta --index_offset <chunk_id * chunk_size>``, all writing into the same
``<data_dir>/`` with globally-unique frame indices.
"""

import argparse
import json
import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# examples/ani1x/dataset/ -> repo root is three levels up
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from orbace.data.extxyz import _parse_comment_line
from orbace.tools.isolated_atoms import (
    extract_isolated_atom_e0s,
    compute_isolated_atom_energies,
    _SYMBOL_TO_Z,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("split_extxyz")


def _z_of(symbol: str) -> int:
    Z = _SYMBOL_TO_Z.get(symbol)
    if Z is None:
        from pyscf.data.elements import charge as _pyscf_charge
        Z = _pyscf_charge(symbol)
    return int(Z)


def main():
    p = argparse.ArgumentParser(description="Split extxyz into chunks + resolve E0 once")
    p.add_argument("--input", required=True, help="Large input extxyz file")
    p.add_argument("--chunks_dir", default="chunks", help="Where to write chunk_*.extxyz")
    p.add_argument("--data_dir", default="data", help="Where to write e0s.json/metadata.json")
    p.add_argument("--chunk_size", type=int, default=2000, help="Frames per chunk file")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--localizer", default="pm", choices=["boys", "pm", "canonical"])
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument(
        "--baseline_model", default="none",
        choices=["none", "xtb", "gfn2", "gfn1", "gfn0", "hf_sto3g"],
        help="Recorded in metadata so chunk jobs / training stay consistent.",
    )
    p.add_argument(
        "--e0s_json", default=None,
        help="Reference isolated-atom E0s (JSON: {Z: eV}) to use directly. "
        "Required when the extxyz has NO config_type=IsolatedAtom frames and "
        "the labels are at a non-HF reference level (E0 must match that level, "
        "not HF/sto-3g). If omitted, E0s come from IsolatedAtom frames in the "
        "file, falling back to HF only for missing elements.",
    )
    p.add_argument("--e0_device", default="cpu", help="SCF backend for E0 atoms (cpu/cuda).")
    args = p.parse_args()

    os.makedirs(args.chunks_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    elements = set()
    isolated_frames = []
    total = 0
    chunk_idx = 0
    in_chunk = 0
    fout = None

    def _chunk_path(k):
        return os.path.join(args.chunks_dir, f"chunk_{k:05d}.extxyz")

    logger.info(f"Streaming {args.input} -> {args.chunks_dir}/ (chunk_size={args.chunk_size})")
    with open(args.input) as f:
        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            natoms = int(s)
            comment = f.readline()
            atom_lines = [f.readline() for _ in range(natoms)]
            info = _parse_comment_line(comment.strip())

            if info.get("config_type") == "IsolatedAtom" and natoms == 1:
                sym = atom_lines[0].split()[0]
                energy = None
                for ek in ("REF_energy", "energy", "Energy"):
                    if ek in info:
                        energy = float(info[ek])
                        break
                isolated_frames.append(
                    {"symbols": [sym], "energy": energy, "config_type": "IsolatedAtom"}
                )
                continue

            if fout is None or in_chunk >= args.chunk_size:
                if fout is not None:
                    fout.close()
                fout = open(_chunk_path(chunk_idx), "w")
                chunk_idx += 1
                in_chunk = 0

            fout.write(line)
            fout.write(comment)
            for al in atom_lines:
                fout.write(al)
                elements.add(al.split()[0])
            in_chunk += 1
            total += 1
            if total % 50000 == 0:
                logger.info(f"  {total} training frames split into {chunk_idx} chunks...")

    if fout is not None:
        fout.close()

    n_chunks = chunk_idx
    logger.info(f"Split done: {total} training frames -> {n_chunks} chunks")
    logger.info(f"  Elements: {sorted(elements)}")

    # Resolve E0 once for the whole dataset.
    if args.e0s_json:
        # Use the provided reference E0s directly (labels are reference-level;
        # do NOT recompute via HF, which would mix levels of theory).
        logger.info(f"Using reference E0s from {args.e0s_json}")
        with open(args.e0s_json) as jf:
            e0s = {int(k): float(v) for k, v in json.load(jf).items()}
        missing = [s for s in sorted(elements) if _z_of(s) not in e0s]
        if missing:
            raise SystemExit(
                f"--e0s_json {args.e0s_json} is missing E0 for elements present "
                f"in the data: {missing} (Z={[_z_of(s) for s in missing]}). "
                f"Add them and re-run."
            )
    else:
        # E0s from IsolatedAtom frames first, HF fallback for missing elements.
        e0s = extract_isolated_atom_e0s(isolated_frames)
        missing = [s for s in sorted(elements) if _z_of(s) not in e0s]
        if missing:
            logger.info(f"Computing E0 via HF/{args.basis} for: {missing}")
            e0s.update(
                compute_isolated_atom_energies(missing, basis=args.basis, device=args.e0_device)
            )
    for Z, e0 in sorted(e0s.items()):
        logger.info(f"  E0(Z={Z}): {e0:.6f} eV")

    with open(os.path.join(args.data_dir, "e0s.json"), "w") as f:
        json.dump({str(k): v for k, v in e0s.items()}, f, indent=2)

    metadata = {
        "basis": args.basis,
        "localizer": args.localizer,
        "charge": args.charge,
        "spin": args.spin,
        "n_frames": total,
        "input_file": os.path.basename(args.input),
        "e0s": {str(k): v for k, v in e0s.items()},
        "baseline_model": args.baseline_model,
    }
    with open(os.path.join(args.data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    manifest = {
        "n_chunks": n_chunks,
        "chunk_size": args.chunk_size,
        "total_frames": total,
        "elements": sorted(elements),
        "chunks_dir": os.path.abspath(args.chunks_dir),
        "data_dir": os.path.abspath(args.data_dir),
    }
    with open(os.path.join(args.chunks_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Wrote e0s.json + metadata.json -> {args.data_dir}/")
    logger.info(f"Wrote manifest.json -> {args.chunks_dir}/ (n_chunks={n_chunks})")


if __name__ == "__main__":
    main()
