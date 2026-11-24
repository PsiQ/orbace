# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Compute orbital-level features from PySCF mean-field objects.

Given a converged HF/DFT mean-field object with (optionally localized) MO
coefficients, this module computes:
  - Orbital centroids (expectation value of position operator)
  - Mulliken population weights (orbital-to-atom mapping)
  - Per-orbital scalar features (10 total):
      energy, occupancy, h_ii, g_iiii, <r^2>, Mulliken partition energy,
      orbital spread, 3 quadrupole eigenvalues
  - Orbital species encoding (dominant atom type + angular momentum + occupancy)
  - Fock matrix in MO basis (off-diagonal = orbital coupling)
  - Core Hamiltonian in MO basis (separates 1e from 2e effects)
  - Coulomb/exchange integrals in MO basis
  - MO coefficient norms per angular momentum channel

UNIT CONVENTION:
  Positions: Angstrom (converted from PySCF's Bohr)
  Orbital features (energies/integrals): Hartree (atomic units, for stability)
  Forces labels: eV/Angstrom
"""

import re
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


AO_SHELL_TYPES = [
    "1s", "2s", "3s", "4s", "5s",
    "2p", "3p", "4p", "5p",
    "3d", "4d", "5d",
    "4f", "5f",
    "5g",
]
N_SHELL_TYPES = len(AO_SHELL_TYPES)

L_MAP = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

# ── Orbital-species encoding ────────────────────────────────────────────
# An orbital is encoded as a single integer:
#     species = dominant_Z * SPECIES_STRIDE + dominant_l * N_OCC_CLASSES + occ
# with dominant_l in [0, N_L_CHANNELS) and occ in [0, N_OCC_CLASSES).
# SPECIES_STRIDE must be >= N_L_CHANNELS * N_OCC_CLASSES so distinct
# (Z, l, occ) triples never collide.
N_L_CHANNELS = 5          # s, p, d, f, g
N_OCC_CLASSES = 3         # virtual / singly / doubly occupied
SPECIES_STRIDE = 15       # N_L_CHANNELS * N_OCC_CLASSES
MAX_ATOMIC_NUMBER = 118   # full periodic table (Z up to Og)


def num_species_for_z(max_z: int = MAX_ATOMIC_NUMBER) -> int:
    """Smallest ``num_species`` that represents every (Z<=max_z, l, occ) triple.

    Use ``num_species_for_z()`` (default, full periodic table) when sizing a
    foundational model so any downstream element-set is representable without
    clamping/collapsing species.
    """
    return max_z * SPECIES_STRIDE + (N_L_CHANNELS - 1) * N_OCC_CLASSES + (N_OCC_CLASSES - 1) + 1


# Fixed table size that covers the whole periodic table (1785). Recommended
# for foundational models so checkpoints stay compatible across datasets.
FULL_TABLE_NUM_SPECIES = num_species_for_z(MAX_ATOMIC_NUMBER)


def compute_orbital_centroids(
    mol, mo_coeff: NDArray, dipole_ints: NDArray = None
) -> NDArray:
    """Compute orbital centroids r_i = <psi_i|r|psi_i> in Bohr."""
    n_mo = mo_coeff.shape[1]
    if dipole_ints is None:
        dipole_ints = mol.intor("int1e_r", comp=3)
    centroids = np.zeros((n_mo, 3))
    for k in range(3):
        centroids[:, k] = np.einsum("pi,pq,qi->i", mo_coeff, dipole_ints[k], mo_coeff)
    return centroids


def compute_mulliken_weights(
    mol, mo_coeff: NDArray, S: NDArray = None
) -> NDArray:
    """Compute Mulliken population of each orbital on each atom.

    Returns:
        weights: (n_mo, n_atoms) where weights[i, A] is the fractional
                 Mulliken population of orbital i on atom A. Rows sum to 1;
                 individual entries may be negative.
    """
    if S is None:
        S = mol.intor("int1e_ovlp")
    n_ao, n_mo = mo_coeff.shape
    n_atoms = mol.natm

    ao_to_atom = np.empty(n_ao, dtype=int)
    for ish in range(mol.nbas):
        atm_idx = mol.bas_atom(ish)
        ao_slice = slice(mol.ao_loc[ish], mol.ao_loc[ish + 1])
        ao_to_atom[ao_slice] = atm_idx

    SC = S @ mo_coeff
    pop_ao = mo_coeff * SC

    weights = np.zeros((n_mo, n_atoms))
    for A in range(n_atoms):
        mask = ao_to_atom == A
        weights[:, A] = np.sum(pop_ao[mask, :], axis=0)

    # Normalize by the SIGNED row sum. Gross populations of a normalized orbital
    # already sum to C^T S C = 1, and they can be negative in small bases, so
    # dividing by sum|w| instead would leave rows summing to less than one.
    # The orbital centroid sum_A w_iA R_A is then not an affine combination of
    # nuclear positions and fails to translate with the molecule: under
    # R -> R + t it moves by (sum_A w_iA) t, not t.
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) > 1e-12, row_sums, 1.0)
    weights = weights / row_sums
    return weights


_AO_SHELL_RE = re.compile(r"^(\d+)([spdfg])")


def _ao_angular_labels(mol) -> List[str]:
    """Return the shell label for each AO, e.g. '2p' for 2px, '3d' for 3dz^2.

    Only the principal quantum number and the angular character identify the
    shell; everything after it names the magnetic component.

    Keeping every digit instead, as this did previously, folded that component
    label into the key: '3dz^2' became '3d2' and '4f-3' became '4f3', neither
    of which is in AO_SHELL_TYPES, so the bit was never set. Two of the five d
    components and every f and g component were silently dropped, leaving the
    4f/5f/5g columns permanently zero. Minimal bases are s/p only and so were
    unaffected, which is why this survived; def2-SVP and larger are not.
    """
    out = []
    for lbl in mol.ao_labels():
        parts = lbl.split()
        m = _AO_SHELL_RE.match(parts[-1] if parts else "")
        out.append(m.group(1) + m.group(2) if m else "")
    return out


def compute_ao_shell_encoding(mol, mo_coeff: NDArray, threshold: float = 1e-8) -> NDArray:
    """Binary encoding of which AO shell types contribute to each MO."""
    n_ao, n_mo = mo_coeff.shape
    ao_labels = _ao_angular_labels(mol)
    encoding = np.zeros((n_mo, N_SHELL_TYPES), dtype=np.float32)
    for i in range(n_mo):
        present = {ao_labels[mu] for mu in range(n_ao) if abs(mo_coeff[mu, i]) > threshold}
        for j, shell_type in enumerate(AO_SHELL_TYPES):
            if shell_type in present:
                encoding[i, j] = 1.0
    return encoding


def compute_dominant_l(mol, mo_coeff: NDArray) -> NDArray:
    """Determine dominant angular momentum character of each orbital."""
    n_ao, n_mo = mo_coeff.shape
    n_l = 5
    l_weights = np.zeros((n_mo, n_l))

    for ish in range(mol.nbas):
        l_val = mol.bas_angular(ish)
        if l_val >= n_l:
            l_val = n_l - 1
        ao_slice = slice(mol.ao_loc[ish], mol.ao_loc[ish + 1])
        l_weights[:, l_val] += np.sum(mo_coeff[ao_slice, :] ** 2, axis=0)

    return np.argmax(l_weights, axis=1).astype(np.int64)


def encode_orbital_species(
    mol,
    mo_coeff: NDArray,
    mo_occ: NDArray,
    mulliken_weights: NDArray = None,
    S: NDArray = None,
) -> NDArray:
    """Encode each orbital as an integer species index.

    Species = dominant_Z * 15 + dominant_l * 3 + occ_class
    where occ_class: 0=virtual, 1=singly occupied, 2=doubly occupied.
    """
    if mulliken_weights is None:
        mulliken_weights = compute_mulliken_weights(mol, mo_coeff, S)

    dominant_atom = np.argmax(np.abs(mulliken_weights), axis=1)
    dominant_Z = np.array([mol.atom_charge(int(a)) for a in dominant_atom], dtype=np.int64)

    dominant_l = compute_dominant_l(mol, mo_coeff)

    occ_class = np.zeros(len(mo_occ), dtype=np.int64)
    occ_class[mo_occ > 1.5] = 2
    occ_class[(mo_occ > 0.5) & (mo_occ < 1.5)] = 1

    species = dominant_Z * SPECIES_STRIDE + dominant_l * N_OCC_CLASSES + occ_class
    return species


def r2_ints_about_center(mol) -> NDArray:
    """``int1e_r2`` taken about the molecular centre of geometry.

    PySCF places the origin of ``int1e_r2`` at the lab frame, which makes
    <psi|r^2|psi> depend on where the molecule happens to sit: under
    R -> R + t it picks up 2 t.<r> + |t|^2. That is not a property of the
    electronic structure, and feeding it to the network makes the prediction
    frame-dependent -- measurably so, by ~0.5 eV for a 0.5 A displacement.

    Referring the operator to the centre of geometry removes the dependence
    while keeping the information the channel is there to carry. The centre
    translates with the molecule, so the second moment about it is invariant,
    and it is not the orbital's own centroid, so the result still distinguishes
    a diffuse orbital from one that merely sits far from the molecular centre.
    The centred spread sigma of ``compute_orbital_shape_invariants`` is
    unaffected either way, being a difference of two quantities sharing an
    origin.
    """
    origin = mol.atom_coords().mean(axis=0)
    with mol.with_common_orig(origin):
        return mol.intor("int1e_r2")


def compute_orbital_shape_invariants(
    mol, mo_coeff: NDArray, centroids_bohr: NDArray,
    r2_ints: NDArray = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Orbital shape descriptors from the second-moment tensor.

    For each orbital i, compute the quadrupole-like tensor:
        Q_ab = <psi_i|r_a r_b|psi_i> - <psi_i|r_a|psi_i><psi_i|r_b|psi_i>
    The 3 eigenvalues of Q (sorted descending) are rotation invariants that
    encode orbital shape: spherical (all equal), elongated (one large),
    disk-like (one small).

    Also returns the orbital spread sigma = sqrt(Tr(Q)) = sqrt(<r^2> - |<r>|^2)
    and the full traceless quadrupole tensor (an equivariant l=2 quantity whose
    orientation encodes the directional shape the eigenvalues discard).

    Returns:
        spread: (n_mo,) in Bohr
        q_eigenvalues: (n_mo, 3) sorted descending, in Bohr^2
        q_traceless: (n_mo, 3, 3) traceless symmetric quadrupole, in Bohr^2
    """
    n_mo = mo_coeff.shape[1]

    rr_ints = mol.intor("int1e_rr", comp=9).reshape(3, 3, mol.nao, mol.nao)

    second_moment = np.zeros((n_mo, 3, 3))
    for a in range(3):
        for b in range(3):
            second_moment[:, a, b] = np.einsum(
                "pi,pq,qi->i", mo_coeff, rr_ints[a, b], mo_coeff
            )

    centroid_outer = centroids_bohr[:, :, None] * centroids_bohr[:, None, :]
    Q = second_moment - centroid_outer

    eigvals = np.linalg.eigvalsh(Q)
    eigvals = eigvals[:, ::-1].copy()

    trace = np.trace(Q, axis1=1, axis2=2)
    spread = np.sqrt(np.maximum(trace, 0.0))

    # Traceless part: removes the (isotropic) spread already kept as a scalar,
    # leaving a pure l=2 tensor that transforms equivariantly under rotation.
    q_traceless = Q - (trace[:, None, None] / 3.0) * np.eye(3)[None, :, :]

    return (
        spread.astype(np.float32),
        eigvals.astype(np.float32),
        q_traceless.astype(np.float32),
    )


def compute_orbital_features(
    mol,
    mo_coeff: NDArray,
    mo_occ: NDArray,
    mo_energy: NDArray = None,
    hcore: NDArray = None,
    eri_mo: NDArray = None,
    dipole_ints: NDArray = None,
    r2_ints: NDArray = None,
    centroids_bohr: NDArray = None,
) -> NDArray:
    """Compute per-orbital scalar features (kept in atomic units for stability).

    Features per orbital (10 total):
      0: orbital energy F_ii [Hartree] (Fock diagonal in current MO basis)
      1: occupancy (0, 1, or 2) [dimensionless]
      2: one-electron integral diagonal h_ii [Hartree]
      3: two-electron integral diagonal g_iiii [Hartree]
      4: spatial extent <r^2> [Bohr^2]
      5: Mulliken partition energy n_i*(h_ii + F_ii)/2 [Hartree]
      6: orbital spread sigma_i [Bohr]
      7: quadrupole eigenvalue lambda_1 [Bohr^2]
      8: quadrupole eigenvalue lambda_2 [Bohr^2]
      9: quadrupole eigenvalue lambda_3 [Bohr^2]
    """
    N_FEATURES = 10
    n_mo = mo_coeff.shape[1]
    features = np.zeros((n_mo, N_FEATURES), dtype=np.float32)

    if mo_energy is not None:
        features[:, 0] = mo_energy
    elif hcore is not None:
        features[:, 0] = np.einsum("pi,pq,qi->i", mo_coeff, hcore, mo_coeff)

    features[:, 1] = mo_occ

    if hcore is None:
        hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h_ii = np.einsum("pi,pq,qi->i", mo_coeff, hcore, mo_coeff)
    features[:, 2] = h_ii

    if eri_mo is not None:
        if eri_mo.ndim == 4:
            features[:, 3] = eri_mo[range(n_mo), range(n_mo), range(n_mo), range(n_mo)]
        else:
            from pyscf.ao2mo import restore
            eri4 = restore(1, eri_mo, n_mo)
            features[:, 3] = eri4[range(n_mo), range(n_mo), range(n_mo), range(n_mo)]

    if r2_ints is None:
        r2_ints = r2_ints_about_center(mol)
    features[:, 4] = np.einsum("pi,pq,qi->i", mo_coeff, r2_ints, mo_coeff)

    # Mulliken partition energy: exact decomposition of HF electronic energy.
    # sum_i n_i*(h_ii + F_ii)/2 = E_electronic (no double counting).
    F_ii = features[:, 0]  # mo_energy = diag(C^T F C) after localization
    features[:, 5] = mo_occ * (h_ii + F_ii) / 2.0

    if centroids_bohr is not None:
        spread, q_eigs, _q_tensor = compute_orbital_shape_invariants(
            mol, mo_coeff, centroids_bohr, r2_ints=r2_ints,
        )
        features[:, 6] = spread
        features[:, 7:10] = q_eigs

    return features


def compute_fock_matrix_mo(mol, mf, mo_coeff: NDArray) -> NDArray:
    """Compute the Fock matrix in the MO basis: F_pq = C^T F C."""
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    dm = mf.make_rdm1()
    if isinstance(dm, (tuple, list)):
        dm = dm[0] + dm[1]
    vhf = mf.get_veff(mol, dm)
    fock_ao = hcore + vhf
    return mo_coeff.T @ fock_ao @ mo_coeff


def compute_overlap_mo(mol, mo_coeff: NDArray) -> NDArray:
    """Compute overlap matrix in MO basis: S_pq = C^T S C.

    For canonical MOs this is the identity; for localized MOs the
    off-diagonal elements measure non-orthogonality due to the
    localization procedure, which is informative for the network.
    """
    S_ao = mol.intor("int1e_ovlp")
    return mo_coeff.T @ S_ao @ mo_coeff


def compute_coulomb_exchange_mo(
    mol, mo_coeff: NDArray, eri_full: NDArray = None,
) -> Tuple[NDArray, NDArray]:
    """Compute Coulomb J_ij = (ii|jj) and exchange K_ij = (ij|ji) integrals.

    If *eri_full* (n_mo, n_mo, n_mo, n_mo) is provided, reuses it to avoid a
    redundant ao2mo transformation.
    """
    n_mo = mo_coeff.shape[1]
    if eri_full is None:
        from pyscf import ao2mo
        eri_full = ao2mo.kernel(mol, mo_coeff, compact=False).reshape(
            n_mo, n_mo, n_mo, n_mo
        )
    idx = np.arange(n_mo)
    J = eri_full[idx[:, None], idx[:, None], idx[None, :], idx[None, :]].astype(np.float32)
    K = eri_full[idx[:, None], idx[None, :], idx[None, :], idx[:, None]].astype(np.float32)
    return J, K


def compute_coeff_norms_per_l(mol, mo_coeff: NDArray, n_l: int = 5) -> NDArray:
    """Coefficient norms per angular momentum channel for each MO."""
    n_ao, n_mo = mo_coeff.shape
    norms = np.zeros((n_mo, n_l), dtype=np.float32)

    for ish in range(mol.nbas):
        l_val = mol.bas_angular(ish)
        if l_val >= n_l:
            l_val = n_l - 1
        ao_slice = slice(mol.ao_loc[ish], mol.ao_loc[ish + 1])
        norms[:, l_val] += np.sum(mo_coeff[ao_slice, :] ** 2, axis=0)

    return norms


def extract_all_orbital_data(
    mol, mf, localizer: str = "canonical"
) -> Dict[str, NDArray]:
    """Extract all orbital data for OrbACE from a PySCF mean-field object.

    Positions in Angstrom, orbital features/integrals in atomic units (Hartree)
    for numerical stability.

    Returns dict with keys:
        atom_positions [Å], atom_numbers, mo_coeff, mo_occ, mo_energy [Ha],
        orbital_centroids [Å], mulliken_weights, orbital_species,
        orbital_features [a.u.] (10 columns), ao_shell_encoding,
        fock_matrix_mo [Ha], hcore_mo [Ha], overlap_mo [dimensionless],
        coulomb_mo [Ha], exchange_mo [Ha],
        coeff_norms_per_l [dimensionless],
        orbital_quadrupole [Bohr^2] (n_mo, 3, 3) traceless symmetric tensor,
        orbital_polarization [Bohr] (n_mo, 3) centroid minus dominant nucleus
    """
    mo_coeff = np.asarray(mf.mo_coeff, dtype=np.float64)
    mo_occ = np.asarray(mf.mo_occ, dtype=np.float64)
    mo_energy = getattr(mf, "mo_energy", None)
    if mo_energy is not None:
        mo_energy = np.asarray(mo_energy, dtype=np.float64)
    n_mo = mo_coeff.shape[1]

    atom_positions = mol.atom_coords() * BOHR_TO_ANG
    atom_numbers = np.array([mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int64)

    S = mol.intor("int1e_ovlp")
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    dipole_ints = mol.intor("int1e_r", comp=3)
    r2_ints = r2_ints_about_center(mol)

    centroids_bohr = compute_orbital_centroids(mol, mo_coeff, dipole_ints)
    centroids_ang = centroids_bohr * BOHR_TO_ANG
    mulliken_w = compute_mulliken_weights(mol, mo_coeff, S)
    species = encode_orbital_species(mol, mo_coeff, mo_occ, mulliken_w, S)
    ao_encoding = compute_ao_shell_encoding(mol, mo_coeff)
    coeff_norms = compute_coeff_norms_per_l(mol, mo_coeff)

    # Equivariant l=2 orbital shape tensor (orientation the eigenvalues discard)
    _spread, _q_eigs, q_traceless = compute_orbital_shape_invariants(
        mol, mo_coeff, centroids_bohr, r2_ints=r2_ints,
    )

    # Polarization vector (l=1): centroid displacement from the dominant
    # nucleus, in Bohr. Frozen HF quantity in the lab frame; encodes lone-pair
    # / bond-polarization direction.
    atom_coords_bohr = mol.atom_coords()
    if mulliken_w.shape[1] > 0:
        dominant_atom = np.argmax(np.abs(mulliken_w), axis=1)
        orbital_polarization = centroids_bohr - atom_coords_bohr[dominant_atom]
    else:
        orbital_polarization = np.zeros((n_mo, 3), dtype=np.float64)

    # Overlap matrix in MO basis (off-diagonals = localized-orbital non-orthogonality)
    overlap_mo = mo_coeff.T @ S @ mo_coeff

    # Core Hamiltonian in MO basis (separate 1e from 2e effects)
    hcore_mo = mo_coeff.T @ hcore @ mo_coeff

    # Fock matrix in MO basis — reuse hcore to avoid recomputing it
    dm = mf.make_rdm1()
    if isinstance(dm, (tuple, list)):
        dm = dm[0] + dm[1]
    vhf = mf.get_veff(mol, dm)
    fock_ao = hcore + vhf
    fock_mo = mo_coeff.T @ fock_ao @ mo_coeff

    # Single ERI transformation — the dominant O(N^5) cost.
    from pyscf import ao2mo
    eri_full = ao2mo.kernel(mol, mo_coeff, compact=False).reshape(
        n_mo, n_mo, n_mo, n_mo
    )
    J_mo, K_mo = compute_coulomb_exchange_mo(mol, mo_coeff, eri_full=eri_full)
    features = compute_orbital_features(
        mol, mo_coeff, mo_occ, mo_energy, hcore, eri_mo=eri_full,
        r2_ints=r2_ints, centroids_bohr=centroids_bohr,
    )

    return {
        "atom_positions": atom_positions.astype(np.float32),
        "atom_numbers": atom_numbers,
        "mo_coeff": mo_coeff.astype(np.float32),
        "mo_occ": mo_occ.astype(np.float32),
        "mo_energy": mo_energy.astype(np.float32) if mo_energy is not None else np.zeros(n_mo, dtype=np.float32),
        "orbital_centroids": centroids_ang.astype(np.float32),
        "mulliken_weights": mulliken_w.astype(np.float32),
        "orbital_species": species,
        "orbital_features": features,
        "ao_shell_encoding": ao_encoding,
        "fock_matrix_mo": np.asarray(fock_mo, dtype=np.float32),
        "hcore_mo": np.asarray(hcore_mo, dtype=np.float32),
        "overlap_mo": np.asarray(overlap_mo, dtype=np.float32),
        "coulomb_mo": np.asarray(J_mo, dtype=np.float32),
        "exchange_mo": np.asarray(K_mo, dtype=np.float32),
        "coeff_norms_per_l": coeff_norms,
        "orbital_quadrupole": q_traceless.astype(np.float32),
        "orbital_polarization": orbital_polarization.astype(np.float32),
    }
