# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Read extended XYZ files (REF_energy / REF_forces / Properties / Lattice)."""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _parse_comment_line(comment: str) -> Dict[str, str]:
    info = {}
    i = 0
    while i < len(comment):
        if comment[i] in (" ", "\t"):
            i += 1
            continue
        eq = comment.find("=", i)
        if eq == -1:
            break
        key = comment[i:eq].strip()
        i = eq + 1
        if i < len(comment) and comment[i] == '"':
            close = comment.find('"', i + 1)
            if close == -1:
                close = len(comment)
            info[key] = comment[i + 1 : close]
            i = close + 1
        else:
            end = comment.find(" ", i)
            if end == -1:
                end = len(comment)
            info[key] = comment[i:end]
            i = end
    return info


def _parse_properties(props_str: str) -> List[tuple]:
    parts = props_str.split(":")
    columns = []
    i = 0
    while i + 2 < len(parts):
        columns.append((parts[i], parts[i + 1], int(parts[i + 2])))
        i += 3
    return columns


def _parse_charge_spin(info: Dict[str, str]) -> tuple:
    """Pull total charge and spin off a comment line.

    Spin is normalised to the PySCF convention (number of unpaired electrons,
    2S) because that is what every downstream consumer expects. Files in the
    wild write either that or the multiplicity 2S+1, so the key name decides
    which one was meant; there is no way to tell them apart from the value
    alone, since both are 1 for a doublet.
    """
    charge = None
    for key in ("charge", "total_charge", "Charge", "net_charge"):
        if key in info:
            charge = int(round(float(info[key])))
            break

    spin = None
    for key in ("spin", "Spin", "nunpaired", "multiplicity", "spin_multiplicity"):
        if key in info:
            value = int(round(float(info[key])))
            spin = value - 1 if "multiplicity" in key else value
            break

    return charge, spin


def read_extxyz(path: str) -> List[Dict]:
    """Read extxyz (REF_energy, REF_forces, Properties, Lattice).

    Returns a list of dicts with keys:
        symbols, coords (ndarray), energy (float|None),
        forces (ndarray|None), lattice (ndarray|None),
        pbc (ndarray), charge (int|None), spin (int|None),
        config_type (str, optional).

    charge and spin are None when the frame does not state them, which lets
    callers tell "not specified" apart from an explicit neutral singlet and
    fall back to their own default.
    """
    frames: List[Dict] = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        n = int(line)
        comment = lines[i + 1].strip()
        i += 2
        info = _parse_comment_line(comment)

        energy: Optional[float] = None
        for ekey in ("REF_energy", "energy", "Energy"):
            if ekey in info:
                energy = float(info[ekey])
                break

        col_spec = None
        if "Properties" in info:
            col_spec = _parse_properties(info["Properties"])

        lattice = None
        if "Lattice" in info:
            vals = [float(x) for x in info["Lattice"].split()]
            if len(vals) == 9:
                lattice = np.array(vals).reshape(3, 3)

        pbc_str = info.get("pbc", "F F F")
        pbc = np.array([c.upper() in ("T", "TRUE", "1") for c in pbc_str.split()])

        charge, spin = _parse_charge_spin(info)

        symbols, coords, forces = [], [], []
        for _ in range(n):
            parts = lines[i].split()
            i += 1
            if col_spec is not None:
                col = 0
                row: Dict = {}
                for name, dtype, ncols in col_spec:
                    if dtype == "S":
                        row[name] = parts[col]
                        col += 1
                    else:
                        row[name] = [float(parts[col + j]) for j in range(ncols)]
                        col += ncols
                symbols.append(row.get("species", parts[0]))
                if "pos" in row:
                    coords.append(row["pos"])
                for fkey in ("REF_forces", "forces", "force"):
                    if fkey in row:
                        forces.append(row[fkey])
                        break
            else:
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    forces.append([float(parts[4]), float(parts[5]), float(parts[6])])

        frame: Dict = {
            "symbols": symbols,
            "coords": np.array(coords),
            "energy": energy,
            "forces": np.array(forces) if len(forces) == n else None,
            "lattice": lattice,
            "pbc": pbc,
            "charge": charge,
            "spin": spin,
        }
        cfg = info.get("config_type", "")
        if cfg:
            frame["config_type"] = cfg
        frames.append(frame)
    return frames
