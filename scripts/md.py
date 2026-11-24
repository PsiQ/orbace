#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Short molecular-dynamics driver for trained OrbACE models.

Loads a trained model, wraps it in the matching ASE calculator, and runs a
brief MD trajectory. Primarily a stability / energy-conservation smoke test:

* NVE (VelocityVerlet) reports the total-energy drift -- this is the key
  diagnostic for the atom model, which is conservative by construction and so
  should conserve energy well.
* Langevin (NVT) is a gentler stability check at a target temperature.

Examples:
  python scripts/md.py --method atom --model ck/best_model.pt --xyz test.xyz \
      --ensemble nve --nsteps 20 --dt 0.5 --temperature 300
  python scripts/md.py --method orbital --model ck/best_model.pt --xyz test.xyz \
      --ensemble langevin --nsteps 20 --basis sto-3g --localizer pm
"""
import argparse
import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("md")


def _first_molecule(xyz_path, min_atoms=2):
    """Return the first frame with >= min_atoms atoms (skips IsolatedAtom)."""
    from ase.io import read
    frames = read(xyz_path, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    for at in frames:
        if len(at) >= min_atoms:
            return at
    raise SystemExit(f"No frame with >= {min_atoms} atoms found in {xyz_path}")


def _build_calculator(args, model):
    method = args.method
    if method == "orbital":
        from orbace.calculators import OrbACECalculator
        return OrbACECalculator(
            model=model, basis=args.basis, localizer=args.localizer,
            r_cut=args.r_cut, device=args.device)
    from orbace.calculators import AtomOrbACECalculator
    import torch
    return AtomOrbACECalculator(
        model=model, basis=args.basis, r_cut=args.r_cut,
        device=args.device, dtype=torch.float64)


def main():
    ap = argparse.ArgumentParser(description="Short OrbACE MD stability test")
    ap.add_argument("--method", required=True, choices=["orbital", "atom"])
    ap.add_argument("--model", required=True, help="Trained model checkpoint (.pt)")
    ap.add_argument("--xyz", required=True, help="Start geometry (first molecule used)")
    ap.add_argument("--ensemble", default="nve", choices=["nve", "langevin"])
    ap.add_argument("--nsteps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.5, help="Timestep [fs]")
    ap.add_argument("--temperature", type=float, default=300.0, help="Init/target T [K]")
    ap.add_argument("--friction", type=float, default=0.02, help="Langevin friction [1/fs]")
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--localizer", default="pm")
    ap.add_argument("--r_cut", type=float, default=5.0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_traj", default=None, help="Optional .xyz trajectory output")
    args = ap.parse_args()

    import torch
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase.md.langevin import Langevin
    from ase.io import write

    from orbace.modules.models import model_fidelity_mode

    # Resolve "auto" here: the ASE calculators call torch.device(args.device)
    # directly, which rejects the literal string "auto".
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    atoms = _first_molecule(args.xyz)
    logger.info("Start geometry: %s (%d atoms)",
                atoms.get_chemical_formula(), len(atoms))

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    fidelity = model_fidelity_mode(model)
    baseline = getattr(model, "baseline_model", "none")
    logger.info("Model: method=%s fidelity_method=%s baseline=%s",
                args.method, fidelity, baseline)

    atoms.calc = _build_calculator(args, model)

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    if args.ensemble == "nve":
        dyn = VelocityVerlet(atoms, timestep=args.dt * units.fs)
    else:
        dyn = Langevin(atoms, timestep=args.dt * units.fs,
                       temperature_K=args.temperature, friction=args.friction / units.fs)

    traj = []
    log = {"e_pot": [], "e_kin": [], "e_tot": []}

    def _record():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        if not (np.isfinite(epot) and np.isfinite(ekin)):
            raise FloatingPointError("Non-finite energy encountered during MD")
        log["e_pot"].append(epot); log["e_kin"].append(ekin); log["e_tot"].append(etot)
        if args.out_traj:
            traj.append(atoms.copy())
        n = len(log["e_tot"])
        logger.info("step %3d | E_pot=%14.5f  E_kin=%10.5f  E_tot=%14.5f eV",
                    n - 1, epot, ekin, etot)

    dyn.attach(_record, interval=1)

    logger.info("Running %s MD: %d steps, dt=%.3f fs, T=%.0f K",
                args.ensemble.upper(), args.nsteps, args.dt, args.temperature)
    import os as _os
    from orbace.tools.timing import record_timing
    _mctx = _os.path.basename(_os.path.dirname(_os.path.abspath(args.model)))
    _t_md = time.perf_counter()
    try:
        _record()  # initial state
        dyn.run(args.nsteps)
    except (FloatingPointError, RuntimeError, ValueError) as e:
        record_timing(args.method, "md", "run", time.perf_counter() - _t_md,
                      _mctx, status="unstable")
        logger.error("MD became UNSTABLE: %s", e)
        sys.exit(2)
    _md_elapsed = time.perf_counter() - _t_md
    record_timing(args.method, "md", "run", _md_elapsed, _mctx)
    n_frames = max(len(log["e_tot"]), 1)
    record_timing(args.method, "md", "per_step", _md_elapsed / n_frames, _mctx)

    e_tot = np.array(log["e_tot"])
    n_atoms = len(atoms)
    drift = float(np.max(np.abs(e_tot - e_tot[0])))
    drift_per_atom = drift / n_atoms
    logger.info("=" * 60)
    logger.info("MD complete: %d frames", len(e_tot))
    logger.info("  E_tot range      : [%.5f, %.5f] eV", e_tot.min(), e_tot.max())
    logger.info("  Max |dE_tot|     : %.6f eV  (%.6f eV/atom)", drift, drift_per_atom)
    if args.ensemble == "nve":
        note = ("conservative model -> small drift expected"
                if args.method == "atom"
                else "non-conservative orbital forces -> drift is expected/normal")
        logger.info("  NVE energy drift : %s", note)
    logger.info("=" * 60)

    if args.out_traj and traj:
        write(args.out_traj, traj)
        logger.info("Wrote trajectory: %s (%d frames)", args.out_traj, len(traj))


if __name__ == "__main__":
    main()
