# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""ASE calculator for the AtomOrbACE (atom-centered) model.

Matching training, the model builds cheap differentiable extended-Hückel
features from the geometry in-forward, so ``F = -dE/dR`` via autograd is the
EXACT gradient of the predicted energy -- conservative, no SCF, safe for MD /
geometry optimization.

Delta-learning inference: if the trained model carries a ``baseline_model``
attribute (set at training time), the corresponding baseline energy/forces are
recomputed per structure and added back so the calculator returns absolute
quantities. Reference isolated-atom energies (``model.e0s``) are added too, so
the reported energy is the absolute total energy in the model's label
convention.
"""

from typing import List, Optional

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.data import chemical_symbols

from ..modules.models import model_is_multifidelity


class AtomOrbACECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model=None,
        model_path: Optional[str] = None,
        basis: str = "sto-3g",
        r_cut: float = 5.0,
        charge: int = 0,
        spin: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        add_e0: bool = True,
        baseline_model: str = "auto",
        baseline_basis: Optional[str] = None,
        baseline_xc: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        import os
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        if hasattr(torch, "serialization"):
            torch.serialization.add_safe_globals([slice])
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        else:
            raise ValueError("Provide model or model_path.")

        self.device = torch.device(device)
        self.dtype = dtype
        self.model = self.model.to(self.device)
        if dtype == torch.float64:
            self.model = self.model.double()
        self.model.eval()

        self.basis = basis
        self.r_cut = r_cut
        self.charge = charge
        self.spin = spin
        self.add_e0 = add_e0
        self.e0s = {int(k): float(v) for k, v in getattr(self.model, "e0s", {}).items()}

        # MF models carry a learned cheap-fidelity head, so the physics baseline
        # is never recomputed here (the network already accounts for it).
        self.is_mf = model_is_multifidelity(self.model)

        # Resolve the delta-learning baseline: 'auto' reads it from the model.
        if baseline_model == "auto":
            baseline_model = getattr(self.model, "baseline_model", "none") or "none"
        if self.is_mf:
            baseline_model = "none"
        self.baseline_model = baseline_model
        self.baseline_basis = baseline_basis or getattr(self.model, "baseline_basis", None)
        self.baseline_xc = baseline_xc or getattr(self.model, "baseline_xc", None)

    def _build_batch(self, coords_t, symbols):
        from orbace.data.invariant_graph import PAIR_KEYS, _neighbor_list_isolated
        Z = torch.tensor([chemical_symbols.index(s) for s in symbols], dtype=torch.int64)
        pos_np = coords_t.detach().cpu().numpy()
        s, r, _ = _neighbor_list_isolated(pos_np, self.r_cut)
        edge_index = (torch.tensor(np.stack([s, r], 0), dtype=torch.int64)
                      if len(s) else torch.zeros((2, 0), dtype=torch.int64))
        natm = len(symbols)
        # The model computes differentiable extended-Hückel features from
        # geometry, so we skip any SCF entirely and pass placeholder features
        # (ignored by the model).
        node_scalars = torch.zeros((natm, self.model.n_node_scalars), dtype=coords_t.dtype)
        edge_features = torch.zeros((edge_index.shape[1], len(PAIR_KEYS)), dtype=coords_t.dtype)
        batch = {
            "num_graphs": 1,
            "atom_positions": coords_t,
            "batch_atom": torch.zeros(natm, dtype=torch.int64),
            "edge_index": edge_index,
            "atom_numbers": Z,
            "node_scalars": node_scalars,
            "edge_features": edge_features,
            "has_pbc": False,
            "cell": None,
        }
        if self.charge:
            # The features are built in-forward, so without this the
            # calculator's charge would reach PySCF and nothing else.
            batch["charge"] = torch.tensor([int(self.charge)], dtype=torch.int64)
        return batch

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[List[str]] = None,
        system_changes: List[str] = all_changes,
    ):
        super().calculate(atoms, properties or self.implemented_properties, system_changes)
        symbols = atoms.get_chemical_symbols()
        coords_np = atoms.get_positions()
        Z = atoms.get_atomic_numbers()

        coords_t = torch.tensor(coords_np, dtype=self.dtype, device=self.device,
                                requires_grad=True)
        batch = self._build_batch(coords_t, symbols)
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.enable_grad():
            out = self.model(batch, training=False, compute_force=False)
            # For MF models out["energy"] is already the TOTAL (base + delta)
            # prediction, so no separate base term is added here.
            E = out["energy"].squeeze()
            (grad,) = torch.autograd.grad(E, coords_t)
        e = float(E.detach().cpu())
        f = (-grad).detach().cpu().numpy()

        if self.baseline_model and self.baseline_model != "none":
            from orbace.data.preprocessing import compute_baseline_energy
            be, bf = compute_baseline_energy(
                model=self.baseline_model, symbols=symbols, coords=coords_np,
                charge=self.charge, spin=self.spin, basis_hf=self.basis,
                baseline_basis=self.baseline_basis, baseline_xc=self.baseline_xc,
                with_forces=True)
            e += float(be)
            if bf is not None:
                f = f + np.asarray(bf)

        if self.add_e0 and self.e0s:
            e += float(sum(self.e0s.get(int(z), 0.0) for z in Z))

        self.results["energy"] = e
        self.results["forces"] = f
