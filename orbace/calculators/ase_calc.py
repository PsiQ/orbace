# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""ASE Calculator for OrbACE.

Chains: geometry -> HF -> localization -> orbital features -> OrbACE -> E, F, stress.

The model outputs energy in eV and forces in eV/Ang directly.
No unit conversion needed at the output stage.

Supports:
  - Molecular and periodic systems
  - Stress tensor for PBC
  - Orbital caching for efficiency in MD
  - Loading deployed model from file
"""

import hashlib
from typing import List, Optional  # noqa: F401

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ..modules.models import model_is_multifidelity


class OrbACECalculator(Calculator):
    """ASE Calculator wrapping OrbACE for energy, forces, and stress.

    At each call to calculate():
      1. Build PySCF mol from ASE Atoms
      2. Run HF and localize orbitals (cached if geometry unchanged)
      3. Build orbital graph
      4. Run OrbACE forward pass
      5. Return energy (eV), forces (eV/Ang), and optionally stress (eV/Ang^3)

    ``device`` is the PyTorch device for the model only; Hartree–Fock uses CPU PySCF.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model=None,
        model_path: str = None,
        basis: str = "sto-3g",
        localizer: str = "boys",
        r_cut: float = 6.0,
        charge: int = 0,
        spin: int = 0,
        device: str = "cpu",
        compute_stress: bool = False,
        cache_orbitals: bool = True,
        e0s: dict = None,
        baseline_model: str = "auto",
        baseline_basis: str = "auto",
        baseline_xc: str = "auto",
        orbital_window: Optional[int] = None,
        scf_device: str = "cpu",
        warm_start_scf: bool = False,
        warm_start_loc: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        import os
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        else:
            raise ValueError("Provide model or model_path.")

        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.basis = basis
        self.localizer = localizer
        self.r_cut = r_cut
        self.charge = charge
        self.spin = spin
        self.compute_stress_flag = compute_stress
        self.cache_orbitals = cache_orbitals
        # Reference isolated-atom energies for recovering absolute energies.
        # Priority: explicit arg > E0s persisted on the model at train time >
        # empty (delta/E0-subtracted energies only). Keys are atomic numbers.
        if e0s:
            self.e0s = {int(k): float(v) for k, v in e0s.items()}
        else:
            model_e0s = getattr(self.model, "e0s", None)
            self.e0s = {int(k): float(v) for k, v in model_e0s.items()} if model_e0s else {}

        # Resolve delta-learning baseline model: "auto" reads the attribute
        # stored on the model at training time, otherwise uses the explicit
        # string. When not "none", we compute the baseline energy on every
        # call and add it to the NN prediction.
        if baseline_model == "auto":
            baseline_model = getattr(self.model, "baseline_model", "none") or "none"
        # Multi-fidelity models predict the cheap level with a learned head, so
        # the physics baseline must NOT be recomputed/added at inference.
        if model_is_multifidelity(self.model):
            baseline_model = "none"
        self.baseline_model = baseline_model
        # Basis / functional for HF or DFT baselines. "auto" reads what was
        # recorded on the model at training time so inference recomputes the
        # exact same baseline the residual targets were built against.
        if baseline_basis == "auto":
            baseline_basis = getattr(self.model, "baseline_basis", None)
        if baseline_xc == "auto":
            baseline_xc = getattr(self.model, "baseline_xc", None)
        self.baseline_basis = baseline_basis
        self.baseline_xc = baseline_xc
        self.orbital_window = orbital_window
        # SCF backend device: "cpu" (PySCF) or "cuda" (gpu4pyscf, with CPU
        # fallback). Distinct from ``device``, which is the NN device.
        self.scf_device = scf_device
        # Warm-start flags -- BOTH DEFAULT OFF, based on measurement.
        #
        # Measured (ethanol/STO-3G/PM, H200; slurm/validate_all_*.out):
        #   cold vs cold (same geom)  : 0.00       (cold pipeline is deterministic)
        #   warm SCF only  vs cold    : 2.33       (features shift!)
        #   warm SCF + loc vs cold    : 2.37       (features shift, ~2.4x faster)
        #
        # Warm-starting the SCF from the previous step's density gives the same
        # energy but a slightly different (~1e-9) converged density; that tiny
        # difference is amplified through near-degenerate canonical-orbital
        # diagonalization and the localizer into a large orbital-feature shift.
        # A model trained on cold-prepared features then sees out-of-distribution
        # inputs. So warm-start (SCF and/or localization) is only safe if the
        # TRAINING data was regenerated with the identical warm-start scheme.
        # Localization is ~60% of a step and is the real MD bottleneck, but it
        # cannot be safely accelerated for a cold-trained model.
        self.warm_start_scf = warm_start_scf
        self.warm_start_loc = warm_start_loc

        self._orbital_cache = {}
        self._last_geometry_hash = None
        self._prev_dm = None
        self._prev_loc = None
        self._prev_natoms = None

    def _geometry_hash(self, atoms: Atoms) -> str:
        data = (
            atoms.get_chemical_symbols(),
            atoms.get_positions().round(8).tobytes(),
            atoms.get_cell().array.round(8).tobytes() if atoms.pbc.any() else b"",
        )
        return hashlib.md5(str(data).encode()).hexdigest()

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = None,
        system_changes: List[str] = all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        from orbace.data.preprocessing import geometry_to_orbital_data
        from orbace.data.dataset import collate_orbital_data
        from orbace.tools.isolated_atoms import subtract_e0

        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        has_pbc = atoms.pbc.any()
        cell = atoms.get_cell().array if has_pbc else None
        pbc = atoms.pbc.astype(bool) if has_pbc else None

        geom_hash = self._geometry_hash(atoms) if self.cache_orbitals else None

        # HF/localization run on self.scf_device ("cpu" PySCF by default, or
        # "cuda" for gpu4pyscf with CPU fallback). self.device is the NN device.
        scf_device = self.scf_device

        if self.cache_orbitals and geom_hash == self._last_geometry_hash:
            orb_data = self._orbital_cache["orb_data"]
        else:
            # Warm-start SCF *and* localization from the previous step when the
            # system size is unchanged (same atoms => same basis dimension).
            # Localization is the dominant MD cost, so its warm-start matters
            # most; the SCF warm-start is a smaller, exact bonus.
            n_atoms_now = len(symbols)
            same_system = self._prev_natoms == n_atoms_now
            dm0 = self._prev_dm if (self.warm_start_scf and same_system) else None
            prev_loc = self._prev_loc if (self.warm_start_loc and same_system) else None
            orb_data, state = geometry_to_orbital_data(
                symbols=symbols,
                coords=positions,
                basis=self.basis,
                charge=self.charge,
                spin=self.spin,
                localizer=self.localizer,
                r_cut=self.r_cut,
                unit="Angstrom",
                cell=cell,
                pbc=pbc,
                device=scf_device,
                orbital_window=self.orbital_window,
                dm0=dm0,
                prev_loc=prev_loc,
                return_state=True,
            )
            if self.warm_start_scf or self.warm_start_loc:
                self._prev_dm = state.get("dm")
                self._prev_loc = {"C": state.get("C"), "occ": state.get("occ")}
                self._prev_natoms = n_atoms_now
            if self.cache_orbitals:
                self._orbital_cache["orb_data"] = orb_data
                self._last_geometry_hash = geom_hash

        batch = collate_orbital_data([orb_data])

        batch_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_device[k] = v.to(self.device)
            else:
                batch_device[k] = v

        do_stress = self.compute_stress_flag and "stress" in properties and has_pbc

        with torch.enable_grad():
            output = self.model(
                batch_device, training=False,
                compute_force=True,
                compute_stress=do_stress,
            )

        raw_energy = output["energy"].item()

        baseline_e = 0.0
        baseline_f = None
        if self.baseline_model and self.baseline_model != "none":
            from orbace.data.preprocessing import compute_baseline_energy
            baseline_e, baseline_f = compute_baseline_energy(
                model=self.baseline_model,
                symbols=symbols,
                coords=positions,
                charge=self.charge,
                spin=self.spin,
                with_forces=True,
                basis_hf=self.basis,  # 'hf'/'hf_self' baseline tracks feature basis
                baseline_basis=self.baseline_basis,
                baseline_xc=self.baseline_xc,
                device=self.scf_device,
            )

        if self.e0s:
            e0_sum = sum(
                self.e0s.get(int(z), 0.0)
                for z in atoms.get_atomic_numbers()
            )
            self.results["energy"] = raw_energy + e0_sum + baseline_e
        else:
            self.results["energy"] = raw_energy + baseline_e

        forces = output["forces"]
        if forces is not None:
            f_np = forces.detach().cpu().numpy()
            if baseline_f is not None:
                f_np = f_np + baseline_f
            self.results["forces"] = f_np
        else:
            self.results["forces"] = np.zeros((len(atoms), 3))

        if do_stress and output.get("stress") is not None:
            stress_3x3 = output["stress"].detach().cpu().numpy()[0]
            self.results["stress"] = self._full_to_voigt(stress_3x3)
        else:
            self.results["stress"] = np.zeros(6)

    @staticmethod
    def _full_to_voigt(stress_3x3):
        """Convert (3,3) stress to Voigt notation (6,)."""
        return np.array([
            stress_3x3[0, 0], stress_3x3[1, 1], stress_3x3[2, 2],
            stress_3x3[1, 2], stress_3x3[0, 2], stress_3x3[0, 1],
        ])
