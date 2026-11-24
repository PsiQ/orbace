# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Isolated atom energies (E0) for E0 subtraction.

Prefers ``config_type=IsolatedAtom`` frames already present in the extxyz
(same level of theory as the labels). Falls back to running HF on each
element when no such frames exist.

Uses gpu4pyscf when ``device="cuda"`` and the library is available (fallback path).
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

HARTREE_TO_EV = 27.211386245988

# Maps element symbol → atomic number (PySCF can also do this, but avoids
# an import for the lightweight extraction path).
_SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
}


def extract_isolated_atom_e0s(frames: list) -> Dict[int, float]:
    """Pull E0s from ``config_type=IsolatedAtom`` frames.

    Each such frame is a single-atom structure whose ``REF_energy`` (already in
    eV) is used directly as that element's E0.

    Returns
    -------
    e0s : {atomic_number: energy_in_eV}  — empty dict if no IsolatedAtom frames.
    """
    e0s: Dict[int, float] = {}
    for frame in frames:
        cfg = frame.get("config_type", "")
        if cfg != "IsolatedAtom":
            continue
        if len(frame["symbols"]) != 1:
            logger.warning(
                "IsolatedAtom frame has %d atoms (expected 1) — skipping",
                len(frame["symbols"]),
            )
            continue
        if frame["energy"] is None:
            logger.warning("IsolatedAtom frame for %s has no energy — skipping", frame["symbols"][0])
            continue
        sym = frame["symbols"][0]
        Z = _SYMBOL_TO_Z.get(sym)
        if Z is None:
            from pyscf.data.elements import charge as _pyscf_charge
            Z = _pyscf_charge(sym)
        e0s[int(Z)] = frame["energy"]
        logger.info("  E0 from extxyz IsolatedAtom: %s (Z=%d) = %.6f eV", sym, Z, frame["energy"])
    return e0s


def compute_isolated_atom_energies(
    elements: List[str],
    basis: str = "sto-3g",
    method: str = "hf",
    device: Optional[str] = None,
) -> Dict[int, float]:
    """Run HF on each unique element as an isolated atom.

    Parameters
    ----------
    device : str | None
        ``"cuda"`` / ``"gpu"`` to use gpu4pyscf, else CPU PySCF.

    Returns:
        e0s: {atomic_number: energy_in_eV}
    """
    from pyscf import gto
    from orbace.data.scf_backend import run_hf

    unique = sorted(set(elements))
    e0s = {}

    for symbol in unique:
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=_get_spin(symbol))
        mf = run_hf(mol, device=device, max_cycle=300)

        Z = mol.atom_charge(0)
        e0_eV = mf.e_tot * HARTREE_TO_EV
        e0s[int(Z)] = e0_eV
        logger.info(f"  E0({symbol}, Z={Z}): {mf.e_tot:.10f} Ha = {e0_eV:.6f} eV")

    return e0s


def _get_spin(symbol: str) -> int:
    """Ground-state spin (2S) for common elements."""
    spins = {
        "H": 1, "He": 0, "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3,
        "O": 2, "F": 1, "Ne": 0, "Na": 1, "Mg": 0, "Al": 1, "Si": 2,
        "P": 3, "S": 2, "Cl": 1, "Ar": 0, "K": 1, "Ca": 0,
        "Sc": 1, "Ti": 2, "V": 3, "Cr": 6, "Mn": 5, "Fe": 4,
        "Co": 3, "Ni": 2, "Cu": 1, "Zn": 0, "Ga": 1, "Ge": 2,
        "As": 3, "Se": 2, "Br": 1, "Kr": 0,
    }
    return spins.get(symbol, 0)


def compute_baseline_atomic_e0s(
    elements: List[str],
    baseline_model: str,
    basis: str = "sto-3g",
    charge: int = 0,
    spin_override: Optional[int] = None,
    baseline_basis: Optional[str] = None,
    baseline_xc: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[int, float]:
    """Compute the baseline method's energy for each isolated atom.

    Used to build delta-E0s = E0_ref - E0_baseline for delta-learning, so the
    stored energies are the clean residual ``(E_ref - E_baseline) - sum(delta_E0)``.

    For HF/DFT baselines this runs a single-atom SCF at the baseline basis (and
    functional, for DFT). For xTB this runs a single-atom xTB calculation.
    ``basis`` is the feature basis (used by the ``hf``/``hf_self`` baseline);
    ``baseline_basis`` / ``baseline_xc`` override the SCF basis / functional for
    the fixed-basis HF and DFT baselines.
    Elements where the baseline fails (e.g. xTB on H) get E0_baseline = 0.
    """
    from orbace.data.preprocessing import compute_baseline_energy

    _SYM_TO_Z = _SYMBOL_TO_Z.copy()
    unique = sorted(set(elements))
    e0s: Dict[int, float] = {}

    for sym in unique:
        Z = _SYM_TO_Z.get(sym)
        if Z is None:
            from pyscf.data.elements import charge as _pyscf_charge
            Z = _pyscf_charge(sym)

        atom_spin = _get_spin(sym) if spin_override is None else spin_override
        coords = np.array([[0.0, 0.0, 0.0]])
        try:
            e_baseline = compute_baseline_energy(
                model=baseline_model,
                symbols=[sym],
                coords=coords,
                charge=0,
                spin=atom_spin,
                basis_hf=basis,
                baseline_basis=baseline_basis,
                baseline_xc=baseline_xc,
                device=device,
                with_forces=False,
            )
            e0s[int(Z)] = float(e_baseline)
            logger.info(f"  E0_baseline({sym}, Z={Z}): {e_baseline:.6f} eV [{baseline_model}]")
        except Exception as exc:
            e0s[int(Z)] = 0.0
            logger.warning(
                f"  E0_baseline({sym}) failed ({exc}); using 0.0"
            )

    return e0s


def subtract_e0(
    energy_eV: float,
    atom_numbers: np.ndarray,
    e0s: Dict[int, float],
) -> float:
    """Subtract isolated atom energies from total energy.

    E_interaction = E_total - sum_i E0(Z_i)
    """
    e0_sum = sum(e0s.get(int(z), 0.0) for z in atom_numbers)
    return energy_eV - e0_sum


def compute_e0_from_extxyz(
    frames: list,
    basis: str = "sto-3g",
    device: Optional[str] = None,
) -> Dict[int, float]:
    """Get E0s for every element in *frames*.

    1. If the extxyz contains ``config_type=IsolatedAtom`` frames, their
       ``REF_energy`` is used directly (same level of theory as labels).
    2. For any elements **not** covered by IsolatedAtom frames, falls back to
       running HF with the given basis / device.
    """
    all_symbols = set()
    for frame in frames:
        if frame.get("config_type") != "IsolatedAtom":
            all_symbols.update(frame["symbols"])

    e0s = extract_isolated_atom_e0s(frames)

    missing = []
    for sym in sorted(all_symbols):
        Z = _SYMBOL_TO_Z.get(sym)
        if Z is None:
            from pyscf.data.elements import charge as _pyscf_charge
            Z = _pyscf_charge(sym)
        if Z not in e0s:
            missing.append(sym)

    if missing:
        logger.info(
            "No IsolatedAtom frames for %s — computing E0 via HF/%s",
            missing,
            basis,
        )
        computed = compute_isolated_atom_energies(missing, basis=basis, device=device)
        e0s.update(computed)
    elif e0s:
        logger.info(
            "All E0s from IsolatedAtom frames in extxyz (%d elements)",
            len(e0s),
        )
    else:
        logger.info("Computing E0s for elements: %s", sorted(all_symbols))
        e0s = compute_isolated_atom_energies(sorted(all_symbols), basis=basis, device=device)

    return e0s
