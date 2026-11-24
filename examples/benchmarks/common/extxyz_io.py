# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Write OrbACE-flavoured extended XYZ from benchmark datasets.

``orbace.data.extxyz`` reads ``REF_energy`` / ``REF_forces`` and treats frames
tagged ``config_type=IsolatedAtom`` as the E0 reference, so every benchmark
converter in this directory funnels through the same writer to keep that
contract in one place.

Energies are eV and forces eV/Angstrom throughout OrbACE. Converters are
responsible for unit conversion before calling in; the constants below cover
the units the public benchmark releases actually ship.
"""
from typing import Dict, Iterable, Optional, Sequence

KCAL_PER_MOL_TO_EV = 0.043364104241800934
HARTREE_TO_EV = 27.211386245988

_SYMBOLS = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
    13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}


def symbol(z: int) -> str:
    """Element symbol for an atomic number, restricted to OrbACE's scope."""
    try:
        return _SYMBOLS[int(z)]
    except KeyError:
        raise ValueError(
            f"atomic number {z} is outside the element set OrbACE supports; "
            "see _ELEMENTS in orbace/data/tb_features.py") from None


def write_frame(fh, symbols: Sequence[str], coords, energy: float,
                forces=None, config_type: Optional[str] = None,
                box: float = 50.0, extra: Optional[Dict] = None,
                charge: Optional[int] = None,
                spin: Optional[int] = None) -> None:
    """Append one frame. ``forces=None`` writes zeros (isolated-atom frames).

    ``spin`` is the unpaired electron count (2S), not the multiplicity, matching
    what ``orbace.data.extxyz`` hands back. Both it and ``charge`` are omitted
    when None so neutral singlet frames stay free of redundant keys.
    """
    n = len(symbols)
    if forces is None:
        forces = [(0.0, 0.0, 0.0)] * n
    fh.write(f"{n}\n")
    cmt = (f'Lattice="{box} 0.0 0.0 0.0 {box} 0.0 0.0 0.0 {box}" '
           f'Properties=species:S:1:pos:R:3:REF_forces:R:3')
    if config_type:
        cmt += f" config_type={config_type}"
    cmt += f" REF_energy={float(energy)!r}"
    if charge is not None:
        cmt += f" charge={int(charge)}"
    if spin is not None:
        cmt += f" spin={int(spin)}"
    for k, v in (extra or {}).items():
        cmt += f" {k}={v!r}"
    cmt += ' pbc="F F F"\n'
    fh.write(cmt)
    for s, c, fo in zip(symbols, coords, forces):
        fh.write(f"{s:<2s} {c[0]:15.8f} {c[1]:15.8f} {c[2]:15.8f} "
                 f"{fo[0]:15.8f} {fo[1]:15.8f} {fo[2]:15.8f}\n")


def write_extxyz(path: str, frames: Iterable[Dict],
                 isolated: Optional[Dict[str, float]] = None,
                 box: float = 50.0) -> int:
    """Write frames to ``path``, prepending IsolatedAtom E0 frames if given.

    ``frames`` yields dicts with ``symbols``, ``coords``, ``energy`` and
    optionally ``forces``, ``charge`` and ``spin``. ``isolated`` maps element
    symbol to E0 in eV. Returns the number of non-isolated frames written.
    """
    n_written = 0
    with open(path, "w") as fh:
        for sym, e0 in (isolated or {}).items():
            write_frame(fh, [sym], [(0.0, 0.0, 0.0)], e0,
                        config_type="IsolatedAtom", box=box)
        for fr in frames:
            write_frame(fh, fr["symbols"], fr["coords"], fr["energy"],
                        fr.get("forces"), box=box, extra=fr.get("extra"),
                        charge=fr.get("charge"), spin=fr.get("spin"))
            n_written += 1
    return n_written
