# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""End-to-end pipeline: molecular geometry -> OrbitalData.

Takes an XYZ geometry (or PySCF mol), runs HF, optionally localizes
orbitals, extracts orbital features, and builds the orbital graph.
Supports both isolated and periodic systems.

Default workflows use **CPU PySCF** for HF; pass ``device="cuda"`` only if you
want **gpu4pyscf** (optional). PyTorch training uses CUDA separately when available.
"""

from typing import Optional

import numpy as np

from .orbital_features import extract_all_orbital_data
from .orbital_graph import OrbitalData, build_orbital_graph
from .scf_backend import run_hf as _run_hf_backend
from .scf_backend import run_dft as _run_dft_backend


def _build_mol(
    symbols,
    coords,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    unit: str = "Angstrom",
):
    from pyscf import gto
    atom_lines = []
    for s, (x, y, z) in zip(symbols, coords):
        atom_lines.append(f"{s} {x} {y} {z}")
    atom_str = "; ".join(atom_lines)
    mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin, unit=unit)
    return mol


def _run_hf(mol, max_cycle: int = 200, device: Optional[str] = None, dm0=None):
    """Run HF via PySCF on CPU (default) or gpu4pyscf when ``device`` requests CUDA.

    ``dm0`` warm-starts SCF from a previous converged density matrix (MD).
    """
    return _run_hf_backend(mol, device=device, max_cycle=max_cycle, dm0=dm0)


def _run_dft(mol, xc: str, max_cycle: int = 200, device: Optional[str] = None, dm0=None):
    """Run KS-DFT via PySCF (CPU) or gpu4pyscf when ``device`` requests CUDA."""
    return _run_dft_backend(mol, xc, device=device, max_cycle=max_cycle, dm0=dm0)


HARTREE_TO_EV = 27.211386245988


def _run_xtb(
    atom_numbers: np.ndarray,
    coords_angstrom: np.ndarray,
    param: str = "GFN2xTB",
    charge: int = 0,
    uhf: int = 0,
    with_forces: bool = False,
):
    """Run a GFN-xTB single point; return (E_eV,) or (E_eV, F_eV_per_Ang).

    Used as a cheap baseline for delta-learning: the NN learns the residual
    (E_true - E_xtb), which is much smoother and smaller than absolute energy.
    Requires xtb-python. GFN2 is the default and most accurate member.
    """
    try:
        from xtb.interface import Calculator, Param
    except ImportError as exc:
        raise ImportError(
            "xtb-python not installed; `pip install xtb` to use xtb baseline."
        ) from exc

    ANG_TO_BOHR = 1.8897261246257702
    param_map = {
        "GFN0xTB": Param.GFN0xTB,
        "GFN1xTB": Param.GFN1xTB,
        "GFN2xTB": Param.GFN2xTB,
    }
    p = param_map.get(param, Param.GFN2xTB)

    coords_bohr = np.asarray(coords_angstrom, dtype=np.float64) * ANG_TO_BOHR
    nums = np.asarray(atom_numbers, dtype=np.int64)
    calc = Calculator(p, numbers=nums, positions=coords_bohr, charge=float(charge), uhf=int(uhf))
    calc.set_verbosity(0)
    res = calc.singlepoint()
    energy_eV = float(res.get_energy()) * HARTREE_TO_EV
    if not with_forces:
        return energy_eV
    # xtb returns gradient in Hartree/Bohr; forces = -gradient.
    # 1 Hartree/Bohr = HARTREE_TO_EV * ANG_TO_BOHR eV/Ang.
    grad = np.asarray(res.get_gradient(), dtype=np.float64)
    forces_eV_per_Ang = -grad * (HARTREE_TO_EV * ANG_TO_BOHR)
    return energy_eV, forces_eV_per_Ang.astype(np.float32)


def scf_baseline_from_mf(mf, with_forces: bool = True):
    """Baseline (energy_eV, forces_eV_per_Ang) from an ALREADY-CONVERGED mf.

    Works for any PySCF mean-field (HF or KS-DFT): ``mf.nuc_grad_method()``
    dispatches to the correct analytic gradient.

    ``with_forces=False`` returns ``(energy, None)`` and skips the gradient,
    which is the dominant cost here -- roughly 7x the SCF itself at STO-3G, and
    about two thirds of the whole per-frame preparation. The energy is free
    (``mf.e_tot`` is already converged).

    IMPORTANT: call this on the *canonical* mean-field (before localization).
    The analytic nuclear gradient assumes canonical orbitals (F C = eps C);
    feeding a localized mf (non-diagonal Fock) yields a wrong energy-weighted
    density matrix and hence wrong forces. ``mf.e_tot`` itself is invariant to
    localization, but the gradient is not.

    Lets data prep reuse the SCF it already ran for orbital features instead of
    running a second SCF for feature-basis HF delta-learning baselines.
    """
    ANG_TO_BOHR = 1.8897261246257702
    energy_eV = float(mf.e_tot) * HARTREE_TO_EV
    if not with_forces:
        return energy_eV, None
    # Silence the analytic-gradient printout: PySCF dumps a full per-atom
    # gradient table (+ SCF cycle lines) at INFO verbosity, which at 442k frames
    # x 32 workers floods the log with tens of millions of lines and adds heavy
    # concurrent-write I/O. The gradient result is unchanged.
    g = mf.nuc_grad_method()
    g.verbose = 0
    grad = g.kernel()
    forces_eV_per_Ang = -np.asarray(grad, dtype=np.float64) * (HARTREE_TO_EV * ANG_TO_BOHR)
    return energy_eV, forces_eV_per_Ang.astype(np.float32)


# Backward-compatible alias (older imports expect this name).
hf_baseline_from_mf = scf_baseline_from_mf


def is_hf_sto3g_baseline(model: str) -> bool:
    """True for the FIXED minimal-basis (STO-3G) Hartree-Fock baseline."""
    return (model or "").lower() in ("hf_sto3g", "hf-sto3g", "hf_minimal")


def is_hf_self_baseline(model: str) -> bool:
    """True for the HF baseline taken at the SAME basis as the features.

    Unlike ``hf_sto3g`` (always STO-3G), this baseline follows whatever basis
    the orbital features are computed in, so the NN residual is level-consistent
    for any basis and the feature SCF is always reused (no second SCF).
    """
    return (model or "").lower() in ("hf", "hf_self", "hf_feature")


def is_hf_baseline(model: str) -> bool:
    """True for any Hartree-Fock baseline variant."""
    return is_hf_sto3g_baseline(model) or is_hf_self_baseline(model)


def is_sto3g_basis(basis: str) -> bool:
    """True when *basis* is STO-3G (ignoring case / separators)."""
    return (basis or "").lower().replace("-", "").replace("_", "") == "sto3g"


# Curated presets: keyword -> (kind, basis, xc). ``xc`` is None for HF.
# Users pick one of these via --baseline_model, or use the generic "hf"/"dft"
# keywords together with --baseline_basis / --baseline_xc for full control.
_SCF_BASELINE_PRESETS = {
    "hf_sto3g": ("hf", "sto-3g", None),
    "hf_321g": ("hf", "3-21g", None),
    "hf_631g": ("hf", "6-31g", None),
    "hf_svp": ("hf", "def2-svp", None),
    "dft_svp": ("dft", "def2-svp", "wb97x"),
    "wb97x_svp": ("dft", "def2-svp", "wb97x"),
    "wb97x_631gs": ("dft", "6-31g*", "wb97x"),
    "pbe_svp": ("dft", "def2-svp", "pbe"),
    "b3lyp_svp": ("dft", "def2-svp", "b3lyp"),
}


def resolve_scf_baseline(
    model: str,
    feature_basis: str = "sto-3g",
    baseline_basis: str = None,
    baseline_xc: str = None,
):
    """Resolve an SCF-type baseline to ``(kind, basis, xc)`` or ``None``.

    ``kind`` is ``"hf"`` or ``"dft"``; ``xc`` is ``None`` for HF. Returns
    ``None`` for non-SCF baselines (xTB / none), so the caller can dispatch.

    Resolution order:
      - ``hf`` / ``hf_self`` / ``hf_feature`` : HF at ``baseline_basis`` if the
        user set one, else the same basis as the orbital features.
      - ``dft`` : KS-DFT at ``baseline_basis`` (default ``def2-svp``) with
        ``baseline_xc`` (default ``wb97x``).
      - a curated preset name (see ``_SCF_BASELINE_PRESETS``), with any explicit
        ``--baseline_basis`` / ``--baseline_xc`` overriding the preset defaults.
    """
    m = (model or "").lower()
    if is_hf_self_baseline(m):
        return ("hf", baseline_basis or feature_basis, None)
    if m == "dft":
        return ("dft", baseline_basis or "def2-svp", baseline_xc or "wb97x")
    if m in _SCF_BASELINE_PRESETS:
        kind, dbasis, dxc = _SCF_BASELINE_PRESETS[m]
        basis = baseline_basis or dbasis
        xc = (baseline_xc or dxc) if kind == "dft" else None
        return (kind, basis, xc)
    return None


def compute_baseline_energy(
    model: str,
    symbols,
    coords,
    atom_numbers=None,
    charge: int = 0,
    spin: int = 0,
    basis_hf: str = "sto-3g",
    baseline_basis: str = None,
    baseline_xc: str = None,
    with_forces: bool = False,
    device: str = None,
):
    """Dispatch to the requested baseline model and return energy (eV).

    When ``with_forces=True`` returns ``(energy, forces)`` with forces in
    eV/Angstrom and shape ``(n_atoms, 3)``.

    Supported models:
      - ``"xtb"`` / ``"gfn2"``   : GFN2-xTB semi-empirical TB (fastest; decent
                                   forces because it is fit to geometries)
      - ``"gfn1"`` / ``"gfn0"``  : earlier GFN parameterizations
      - ``"hf_sto3g"``           : minimal-basis (STO-3G) Hartree-Fock
      - ``"hf"`` / ``"hf_self"`` : HF at the feature basis (or --baseline_basis)
      - ``"hf_321g"``/``"hf_svp"``: HF at a fixed larger basis (better forces)
      - ``"dft"``                : KS-DFT at ``--baseline_basis`` (def2-svp) with
                                   ``--baseline_xc`` (wb97x); best forces/cost
      - ``"wb97x_svp"``/``"pbe_svp"``/``"b3lyp_svp"`` : curated DFT presets
      - ``"none"``               : returns 0.0 (pass-through)
    """
    m = (model or "none").lower()
    if m == "none":
        if with_forces:
            n = len(symbols) if symbols is not None else (len(coords) if coords is not None else 0)
            return 0.0, np.zeros((n, 3), dtype=np.float32)
        return 0.0
    if atom_numbers is None:
        from ase.data import atomic_numbers as _Z
        atom_numbers = np.array([_Z[s] for s in symbols], dtype=np.int64)
    if m in ("xtb", "gfn2", "gfn2xtb"):
        return _run_xtb(atom_numbers, coords, param="GFN2xTB", charge=charge, uhf=spin, with_forces=with_forces)
    if m in ("gfn1", "gfn1xtb"):
        return _run_xtb(atom_numbers, coords, param="GFN1xTB", charge=charge, uhf=spin, with_forces=with_forces)
    if m in ("gfn0", "gfn0xtb"):
        return _run_xtb(atom_numbers, coords, param="GFN0xTB", charge=charge, uhf=spin, with_forces=with_forces)
    if is_hf_sto3g_baseline(m):
        # Kept explicit for the historical fixed-STO-3G behavior.
        mol = _build_mol(symbols, coords, basis="sto-3g", charge=charge, spin=spin, unit="Angstrom")
        mf = _run_hf(mol, device=device)
        if not with_forces:
            return float(mf.e_tot) * HARTREE_TO_EV
        return scf_baseline_from_mf(mf)

    spec = resolve_scf_baseline(
        m, feature_basis=basis_hf, baseline_basis=baseline_basis, baseline_xc=baseline_xc,
    )
    if spec is not None:
        kind, basis, xc = spec
        mol = _build_mol(symbols, coords, basis=basis, charge=charge, spin=spin, unit="Angstrom")
        mf = _run_hf(mol, device=device) if kind == "hf" else _run_dft(mol, xc=xc, device=device)
        if not with_forces:
            return float(mf.e_tot) * HARTREE_TO_EV
        return scf_baseline_from_mf(mf)
    raise ValueError(f"Unknown baseline_model: {model}")


def _warm_loc_guess(C_canon, C_prev, S):
    """Project a previous step's localized orbitals into the current block's
    occupied space and Lowdin-orthonormalize, giving a near-localized starting
    guess for Boys/PM. Both blocks must have the same width (constant electron
    count, which holds along an MD trajectory).
    """
    M = C_canon.T @ S @ C_prev            # (nb, nb): overlap canon vs prev-loc
    C_guess = C_canon @ M                 # prev-localized shape, in current span
    G = C_guess.T @ S @ C_guess           # (nb, nb) metric
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 1e-10, None)
    G_inv_sqrt = (V * (1.0 / np.sqrt(w))) @ V.T
    return C_guess @ G_inv_sqrt


def _localize(mf, method: str = "boys", prev_C=None, prev_occ=None):
    """Localize occupied/SOMO/virtual blocks.

    ``prev_C`` / ``prev_occ`` (a previous step's localized coefficients and
    occupations) warm-start the iterative localizer from the projected previous
    solution -- in MD this is the dominant per-step cost, and small geometry
    deltas make the previous localization an excellent starting point. The
    localizer still converges to a valid localized set of the current occupied
    space; warm-start only changes the (non-unique) starting point and tends to
    track a smoother optimum across the trajectory.
    """
    from pyscf import lo
    mol = mf.mol
    mo_coeff = mf.mo_coeff.copy()
    mo_occ = mf.mo_occ.copy()

    if method == "canonical":
        return mf

    idx_occ = np.flatnonzero(mo_occ > 1.5)
    idx_somo = np.flatnonzero((mo_occ > 0.5) & (mo_occ < 1.5))
    idx_virt = np.flatnonzero(mo_occ < 0.5)

    S = mol.intor("int1e_ovlp") if prev_C is not None and prev_occ is not None else None

    def _prev_block(mask):
        if S is None:
            return None
        pidx = np.flatnonzero(mask(np.asarray(prev_occ)))
        if pidx.size == 0:
            return None
        return np.asarray(prev_C)[:, pidx]

    def localize_block(indices, prev_block=None):
        if len(indices) <= 1:
            return mo_coeff[:, indices]
        C_block = mo_coeff[:, indices]
        if method == "boys":
            loc = lo.Boys(mol, C_block)
        elif method in ("pm", "pipek"):
            loc = lo.PM(mol, C_block)
        else:
            return C_block
        # PySCF's default 'atomic' guess selects lab-frame AO components, so a
        # rotated copy can converge to a different local optimum; starting
        # from the canonical MOs keeps the result covariant with the molecule.
        loc.init_guess = None
        # Warm-start from projected previous localized orbitals when shapes match.
        if prev_block is not None and prev_block.shape[1] == C_block.shape[1]:
            try:
                guess = _warm_loc_guess(C_block, prev_block, S)
                return loc.kernel(mo_coeff=guess)
            except Exception:
                pass
        return loc.kernel()

    C_occ = localize_block(idx_occ, _prev_block(lambda o: o > 1.5))
    C_somo = localize_block(idx_somo, _prev_block(lambda o: (o > 0.5) & (o < 1.5)))
    C_virt = localize_block(idx_virt, _prev_block(lambda o: o < 0.5))

    mf.mo_coeff = np.hstack([C_occ, C_somo, C_virt])
    mf.mo_occ = np.hstack([mo_occ[idx_occ], mo_occ[idx_somo], mo_occ[idx_virt]])

    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    dm = mf.make_rdm1()
    if isinstance(dm, (tuple, list)):
        dm = dm[0] + dm[1]
    vhf = mf.get_veff(mol, dm)
    fock = hcore + vhf
    mf.mo_energy = np.diag(mf.mo_coeff.T @ fock @ mf.mo_coeff)

    return mf


def geometry_to_orbital_data(
    symbols=None,
    coords=None,
    mol=None,
    mf=None,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    localizer: str = "boys",
    r_cut: float = 6.0,
    energy: float = None,
    forces=None,
    stress=None,
    cell=None,
    pbc=None,
    unit: str = "Angstrom",
    device: str = None,
    orbital_window: Optional[int] = None,
    dm0=None,
    return_dm: bool = False,
    prev_loc=None,
    return_state: bool = False,
):
    """Full pipeline: geometry/mol/mf -> OrbitalData.

    Provide one of:
      - (symbols, coords): build mol and run HF
      - mol: run HF on existing mol
      - mf: use existing mean-field object directly

    Set ``device="cuda"`` only to opt into gpu4pyscf; otherwise use CPU PySCF.

    ``dm0`` warm-starts the SCF from a previous converged density matrix; in MD
    (constant atoms, small position deltas) this cuts SCF iterations sharply and
    is exact. ``prev_loc`` (a ``{"C", "occ"}`` dict from a previous step) warm-
    starts the iterative localizer, which is the dominant MD cost.

    Return shape:
      - ``return_state=True``: ``(orb_data, state)`` where ``state`` carries the
        canonical density matrix and localized coefficients for the next step.
      - ``return_dm=True`` (legacy): ``(orb_data, dm)``.
      - otherwise: ``orb_data``.
    """
    if mf is None:
        if mol is None:
            if symbols is None or coords is None:
                raise ValueError("Provide (symbols, coords), mol, or mf.")
            mol = _build_mol(symbols, coords, basis, charge, spin, unit)
        mf = _run_hf(mol, device=device, dm0=dm0)
    else:
        mol = mf.mol

    # Canonical density matrix (invariant to the subsequent localization).
    converged_dm = mf.make_rdm1() if (return_dm or return_state) else None

    prev_C = prev_loc.get("C") if isinstance(prev_loc, dict) else None
    prev_occ = prev_loc.get("occ") if isinstance(prev_loc, dict) else None
    mf = _localize(mf, localizer, prev_C=prev_C, prev_occ=prev_occ)

    orb_data = extract_all_orbital_data(mol, mf, localizer)
    graph = build_orbital_graph(
        orb_data, r_cut=r_cut, energy=energy, forces=forces,
        stress=stress, cell=cell, pbc=pbc, orbital_window=orbital_window,
    )
    if return_state:
        state = {
            "dm": converged_dm,
            "C": np.asarray(mf.mo_coeff).copy(),
            "occ": np.asarray(mf.mo_occ).copy(),
        }
        return graph, state
    if return_dm:
        return graph, converged_dm
    return graph
