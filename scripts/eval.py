#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Unified OrbACE evaluation -- one entry point for every model variant.

Method dispatch:

  # Orbital-space OrbACE (localized-orbital features):
  python eval.py --method orbital --model checkpoints/best_model.pt --npz_data prepared/
  python eval.py --method orbital --model checkpoints/best_model.pt --xyz test.xyz --output_xyz preds.xyz

  # AtomOrbACE (atom-centered, conservative extended-Hückel forces, no SCF):
  python eval.py --method atom --model checkpoints_inv/best_model.pt --npz_data idata_valid/
  python eval.py --method atom --model checkpoints_inv/best_model.pt --xyz valid.xyz --fd 3
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from orbace.modules.models import model_fidelity_mode, model_is_multifidelity
from orbace.tools.timing import record_timing, timed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_Z2SYM = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar",
}


# ============================================================================
#  Shared helpers
# ============================================================================
def _resolve_device(choice: str) -> torch.device:
    if choice in ("auto", "cuda"):
        try:
            if torch.cuda.is_available():
                return torch.device("cuda")
        except Exception:
            pass
        return torch.device("cpu")
    return torch.device(choice)


def _load_model(path, device, double=False):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if hasattr(torch, "serialization"):
        torch.serialization.add_safe_globals([slice])
    model = torch.load(path, map_location="cpu", weights_only=False)
    model = model.to(device)
    if double:
        model = model.double()
    return model.eval()


def _stat(x):
    x = np.asarray(x, float)
    return (np.mean(x), np.median(x), np.max(x)) if x.size else (float("nan"),) * 3


_R_CUT_FALLBACK = 5.0


def _resolve_r_cut(model, requested):
    """Graph cutoff for the atom path, preferring the value baked into the model.

    r_cut is not metadata: it rebuilds the neighbour list every time a frame is
    loaded, so a cutoff that disagrees with the one the model trained at feeds
    it connectivity it has never seen and quietly degrades every metric without
    raising anything. The model carries its training r_max as a buffer, so trust
    that unless the caller explicitly asked for something else.
    """
    stored = getattr(model, "r_max", None)
    stored = float(stored) if stored is not None else None
    if requested is None:
        if stored is None:
            logger.warning(
                "model carries no r_max; falling back to r_cut=%.3f A",
                _R_CUT_FALLBACK)
            return _R_CUT_FALLBACK
        return stored
    if stored is not None and abs(stored - requested) > 1e-6:
        logger.warning(
            "--r_cut %.3f A disagrees with the model's trained r_max %.3f A; "
            "honouring the request, but the reported metrics will not reflect "
            "the graph this model was trained on", requested, stored)
    return requested


def _rmse(sq, weights=None):
    """RMSE from per-frame MEAN squared errors, optionally count-weighted.

    RMSE cannot be recovered from the per-frame MAE lists, since mean|x| and
    sqrt(mean x^2) differ within a frame, so callers pass squared errors that
    were accumulated alongside the absolute ones. ``weights`` is the number of
    residuals each frame contributed (3*n_atoms for forces), which keeps frames
    of different size correctly weighted; pass None when every frame carries a
    single residual, as for total energy.
    """
    sq = np.asarray(sq, float)
    if sq.size == 0:
        return float("nan")
    if weights is None:
        return float(np.sqrt(np.mean(sq)))
    w = np.asarray(weights, float)
    return float(np.sqrt(np.sum(sq * w) / np.sum(w)))


def _atom_print_metrics(label, n_frames, eE, eF, eE_tot, eF_sq=None,
                        f_counts=None):
    mE, mF, mEt = _stat(eE), _stat(eF), _stat(eE_tot)
    e_rmse = _rmse(np.asarray(eE_tot, float) ** 2)
    e_rmse_at = _rmse(np.asarray(eE, float) ** 2)
    print(f"{label} {n_frames} frames")
    print(f"  Energy MAE: {mEt[0]:.6f} eV   ({mEt[0]*23.06:.3f} kcal/mol)")
    print(f"  Energy RMSE: {e_rmse:.6f} eV   ({e_rmse*23.06:.3f} kcal/mol)")
    print(f"  Energy MAE/atom (absolute): {mE[0]*1000:.3f} meV  ({mE[0]*23.06:.4f} kcal/mol/atom)")
    print(f"  Energy RMSE/atom (absolute): {e_rmse_at*1000:.3f} meV  ({e_rmse_at*23.06:.4f} kcal/mol/atom)")
    print(f"  Forces MAE      : {mF[0]:.4f} eV/A (median {mF[1]:.4f}, max {mF[2]:.4f})")
    if eF_sq is not None and len(eF_sq):
        print(f"  Forces RMSE     : {_rmse(eF_sq, f_counts):.4f} eV/A")


# ============================================================================
#  Orbital-space OrbACE evaluation
# ============================================================================
def _orbital_prepare_from_xyz(input_path, basis, localizer, charge, spin, r_max,
                              save_dir=None, baseline_model: str = "none",
                              orbital_window: Optional[int] = None,
                              collect_meta: bool = False,
                              e0s_json: Optional[str] = None,
                              baseline_basis: Optional[str] = None,
                              baseline_xc: Optional[str] = None):
    """Read xyz/extxyz, run HF + orbital feature extraction, return OrbitalData."""
    from orbace.data.extxyz import read_extxyz
    from orbace.data.preprocessing import (
        _build_mol, _run_hf, _localize, compute_baseline_energy,
        hf_baseline_from_mf, is_hf_sto3g_baseline, is_hf_self_baseline,
        is_sto3g_basis,
    )
    from orbace.data.orbital_features import extract_all_orbital_data
    from orbace.data.orbital_graph import build_orbital_graph
    from orbace.tools.isolated_atoms import (
        compute_baseline_atomic_e0s, compute_e0_from_extxyz, subtract_e0,
    )

    all_frames = read_extxyz(input_path)
    frames = [f for f in all_frames if f.get("config_type") != "IsolatedAtom"]
    logger.info(f"Read {len(frames)} structures from {input_path}")

    if e0s_json:
        logger.info(f"Loading isolated-atom E0s from {e0s_json}")
        with open(e0s_json) as _f:
            e0s = {int(k): float(v) for k, v in json.load(_f).items()}
    else:
        e0s = compute_e0_from_extxyz(all_frames, basis=basis, device="cpu")
    for Z, e0 in sorted(e0s.items()):
        logger.info(f"  E0_ref(Z={Z}): {e0:.6f} eV")

    use_baseline = baseline_model and baseline_model != "none"
    if use_baseline:
        logger.info(f"Delta-learning baseline: {baseline_model}")
        all_symbols = set()
        for f in frames:
            all_symbols.update(f["symbols"])
        baseline_atom_e0s = compute_baseline_atomic_e0s(
            sorted(all_symbols), baseline_model, basis=basis,
            baseline_basis=baseline_basis, baseline_xc=baseline_xc,
        )
        for Z in e0s:
            e0s[Z] = e0s[Z] - baseline_atom_e0s.get(Z, 0.0)
        logger.info("Using delta E0s (E0_ref - E0_baseline) for delta-learning eval")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "e0s.json"), "w") as f:
            json.dump({str(k): v for k, v in e0s.items()}, f, indent=2)

    data_list = []
    frame_meta = [] if collect_meta else None
    t0 = time.time()
    for idx, frame in enumerate(frames):
        try:
            mol = _build_mol(frame["symbols"], frame["coords"],
                             basis=basis, charge=charge, spin=spin, unit="Angstrom")
            mf = _run_hf(mol, device="cpu")

            baseline_e = 0.0
            baseline_f = None
            if use_baseline:
                _baseline_at_feature_basis = baseline_basis is None or (
                    (baseline_basis or "").lower().replace("-", "").replace("_", "")
                    == (basis or "").lower().replace("-", "").replace("_", "")
                )
                if (is_hf_self_baseline(baseline_model) and _baseline_at_feature_basis) or (
                    is_hf_sto3g_baseline(baseline_model) and is_sto3g_basis(basis)
                ):
                    baseline_e, baseline_f = hf_baseline_from_mf(mf)
                else:
                    baseline_e, baseline_f = compute_baseline_energy(
                        model=baseline_model, symbols=frame["symbols"],
                        coords=frame["coords"], charge=charge, spin=spin,
                        basis_hf=basis, baseline_basis=baseline_basis,
                        baseline_xc=baseline_xc, with_forces=True,
                    )

            mf = _localize(mf, localizer)
            orb_data_dict = extract_all_orbital_data(mol, mf, localizer)

            atom_numbers = np.array(
                [mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int64)
            energy = None
            if frame["energy"] is not None:
                energy = float(subtract_e0(frame["energy"], atom_numbers, e0s) - baseline_e)

            forces = frame.get("forces")
            if forces is not None and baseline_f is not None:
                forces = forces.astype(np.float32) - baseline_f
            cell = frame.get("lattice")
            pbc = frame.get("pbc")

            orbital_data_obj = build_orbital_graph(
                orb_data_dict, r_cut=r_max, energy=energy, forces=forces,
                cell=cell, pbc=pbc,
                baseline_energy=(baseline_e if use_baseline else None),
                orbital_window=orbital_window,
            )
            data_list.append(orbital_data_obj)

            if collect_meta:
                e0_sum = float(sum(e0s.get(int(z), 0.0) for z in atom_numbers))
                frame_meta.append({
                    "symbols": list(frame["symbols"]),
                    "coords": np.asarray(frame["coords"], dtype=np.float64),
                    "ref_energy": (None if frame["energy"] is None else float(frame["energy"])),
                    "ref_forces": (None if frame.get("forces") is None
                                   else np.asarray(frame["forces"], dtype=np.float64)),
                    "lattice": cell, "pbc": pbc, "e0_sum": e0_sum,
                    "baseline_used": use_baseline, "baseline_e": float(baseline_e),
                    "baseline_f": (None if baseline_f is None
                                   else np.asarray(baseline_f, dtype=np.float64)),
                    "forces_known": True,
                })

            if save_dir:
                save_dict = {**orb_data_dict}
                if energy is not None:
                    save_dict["energy"] = np.float32(energy)
                if forces is not None:
                    save_dict["forces"] = forces.astype(np.float32)
                if use_baseline:
                    save_dict["baseline_energy"] = np.float32(baseline_e)
                    if baseline_f is not None:
                        save_dict["baseline_forces"] = baseline_f.astype(np.float32)
                np.savez(os.path.join(save_dir, f"frame_{idx:06d}.npz"), **save_dict)

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Frame {idx} failed: {e}")

        if (idx + 1) % 25 == 0 or idx == len(frames) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            logger.info(f"  {idx+1}/{len(frames)} ({elapsed:.0f}s, {rate:.1f} frames/s)")

    if save_dir:
        metadata = {
            "basis": basis, "localizer": localizer, "charge": charge, "spin": spin,
            "n_frames": len(data_list), "input_file": os.path.basename(input_path),
            "e0s": {str(k): v for k, v in e0s.items()},
            "baseline_model": baseline_model, "baseline_basis": baseline_basis,
            "baseline_xc": baseline_xc,
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved prepared data to {save_dir}/ ({len(data_list)} frames)")

    if collect_meta:
        return data_list, frame_meta
    return data_list


@torch.no_grad()
def _orbital_evaluate(model, data_list, device, batch_size, dataset=None):
    from orbace.data.dataset import OrbitalDataset, collate_orbital_data
    # MF models predict the ABSOLUTE (residual + baseline) target directly, so
    # the physics baseline must NOT be re-added to the prediction -- it is only
    # used to reconstruct the reference for comparison.
    is_mf = model_is_multifidelity(model)
    # `dataset` (possibly lazy) takes precedence so huge test sets stream from
    # disk without materialising every graph in memory.
    if dataset is None:
        dataset = OrbitalDataset(data_list=data_list)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_orbital_data)
    m_e0s = None
    _me0 = getattr(model, "e0s", None)
    if _me0:
        m_e0s = {int(k): float(v) for k, v in _me0.items()}
    e0_sum_list = []

    e_refs, e_preds, n_atoms_list = [], [], []
    f_ref_all, f_pred_all, f_pred_per_frame = [], [], []
    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        with torch.enable_grad():
            out = model(batch_dev, training=False, compute_force=True)
        e_pred = out["energy"].detach().cpu().numpy()
        e_ref = batch["energy"].numpy() if "energy" in batch else None
        if out.get("forces") is not None:
            f_pred_np = out["forces"].detach().cpu().numpy()
            offset = 0
            for n_at in batch["n_atoms"].numpy().tolist():
                f_pred_per_frame.append(f_pred_np[offset:offset + n_at]); offset += n_at
        if "baseline_energy" in batch:
            bsl = batch["baseline_energy"].numpy()
            if not is_mf:
                e_pred = e_pred + bsl
            if e_ref is not None:
                e_ref = e_ref + bsl
        e_preds.extend(e_pred.tolist())
        _nat = batch["n_atoms"].numpy().tolist()
        n_atoms_list.extend(_nat)
        if m_e0s is not None and "atom_numbers" in batch:
            an = batch["atom_numbers"].numpy()
            off = 0
            for n_at in _nat:
                zs = an[off:off + n_at]; off += n_at
                e0_sum_list.append(float(sum(m_e0s.get(int(z), 0.0) for z in zs)))
        if e_ref is not None:
            e_refs.extend(e_ref.tolist())
        if "forces" in batch and out.get("forces") is not None:
            f_pred_all.append(out["forces"].detach().cpu().numpy())
            f_ref = batch["forces"].numpy()
            if is_mf and "baseline_forces" in batch:
                f_ref = f_ref + batch["baseline_forces"].numpy()
            f_ref_all.append(f_ref)

    results = {"e_preds": np.array(e_preds), "n_atoms": np.array(n_atoms_list)}
    if e_refs:
        results["e_refs"] = np.array(e_refs)
    if f_ref_all:
        results["f_refs"] = np.concatenate(f_ref_all, axis=0)
        results["f_preds"] = np.concatenate(f_pred_all, axis=0)
    if f_pred_per_frame:
        results["f_preds_per_frame"] = f_pred_per_frame

    results["energy_scale"] = "e0_subtracted"
    if m_e0s is not None and len(e0_sum_list) == len(results["e_preds"]):
        e0_sum = np.array(e0_sum_list)
        results["e_preds"] = results["e_preds"] + e0_sum
        if "e_refs" in results:
            results["e_refs"] = results["e_refs"] + e0_sum
        results["energy_scale"] = "absolute_total"
    return results


def _orbital_print_results(results):
    n_samples = len(results["e_preds"])
    print("\n" + "=" * 60)
    print("OrbACE Evaluation Results (orbital-space)")
    print("=" * 60)
    print(f"  Samples: {n_samples}")
    EV_PER_KCALMOL = 0.0433641153
    if "e_refs" in results:
        e_err = np.abs(results["e_refs"] - results["e_preds"])
        n_at = results["n_atoms"]
        e_mae = float(np.mean(e_err))
        e_rmse = float(np.sqrt(np.mean(e_err ** 2)))
        print(f"  Energy MAE:       {e_mae:.6f} eV   ({e_mae/EV_PER_KCALMOL:.3f} kcal/mol)")
        print(f"  Energy MAE/atom:  {np.mean(e_err / n_at):.6f} eV/atom")
        print(f"  Energy RMSE:      {e_rmse:.6f} eV   ({e_rmse/EV_PER_KCALMOL:.3f} kcal/mol)")
        within = float(np.mean((e_err / EV_PER_KCALMOL) < 1.0)) * 100.0
        print(f"  Within 1 kcal/mol (total): {within:.1f}% of structures")
    if "f_refs" in results:
        f_err = np.abs(results["f_refs"] - results["f_preds"])
        print(f"  Forces MAE:       {np.mean(f_err):.6f} eV/Ang")
        print(f"  Forces RMSE:      {np.sqrt(np.mean(f_err ** 2)):.6f} eV/Ang")
    e_p = results["e_preds"]
    _scale = results.get("energy_scale", "e0_subtracted")
    _label = "absolute total" if _scale == "absolute_total" else "e0-subtracted"
    print(f"\n  Predicted energy range ({_label}): [{e_p.min():.4f}, {e_p.max():.4f}] eV")
    if "e_refs" in results and n_samples <= 30:
        e_r = results["e_refs"]; e_err = np.abs(e_r - e_p)
        print(f"\n  {'#':<4} {'E_ref':>12} {'E_pred':>12} {'|err|':>10}")
        print(f"  {'-' * 42}")
        for i in range(n_samples):
            print(f"  {i:<4} {e_r[i]:>12.6f} {e_p[i]:>12.6f} {e_err[i]:>10.6f}")
    print("=" * 60)


def _orbital_build_meta_from_npz(data_list, e0s, baseline_used):
    e0s = {int(k): float(v) for k, v in e0s.items()}
    meta = []
    for d in data_list:
        z = d.atom_numbers.cpu().numpy().astype(int)
        coords = d.atom_positions.cpu().numpy().astype(np.float64)
        e0_sum = float(sum(e0s.get(int(zi), 0.0) for zi in z))
        baseline_e = (float(d.baseline_energy) if getattr(d, "baseline_energy", None)
                      is not None else 0.0)
        baseline_f = (d.baseline_forces.cpu().numpy().astype(np.float64)
                      if getattr(d, "baseline_forces", None) is not None else None)
        ref_energy = None
        if getattr(d, "energy", None) is not None:
            ref_energy = float(d.energy) + baseline_e + e0_sum
        forces_known = (not baseline_used) or (baseline_f is not None)
        ref_forces = None
        if forces_known and getattr(d, "forces", None) is not None:
            ref_forces = d.forces.cpu().numpy().astype(np.float64)
            if baseline_f is not None:
                ref_forces = ref_forces + baseline_f
        meta.append({
            "symbols": [_Z2SYM.get(int(zi), "X") for zi in z], "coords": coords,
            "ref_energy": ref_energy, "ref_forces": ref_forces,
            "lattice": (d.cell.cpu().numpy() if getattr(d, "cell", None) is not None else None),
            "pbc": None, "e0_sum": e0_sum, "baseline_used": baseline_used,
            "baseline_e": baseline_e, "baseline_f": baseline_f, "forces_known": forces_known,
        })
    return meta


def _orbital_write_predictions_xyz(out_path, frame_meta, e_preds, f_preds_per_frame):
    n = min(len(frame_meta), len(e_preds))
    have_fpred = f_preds_per_frame is not None and len(f_preds_per_frame) >= n
    written = 0
    with open(out_path, "w") as fout:
        for i in range(n):
            m = frame_meta[i]; syms = m["symbols"]; coords = m["coords"]; natoms = len(syms)
            e_pred_abs = float(e_preds[i]) + float(m["e0_sum"])
            ref_energy = m.get("ref_energy")
            baseline_used = m.get("baseline_used", False)
            baseline_e = float(m.get("baseline_e", 0.0))
            forces_known = m.get("forces_known", True)
            baseline_f = m.get("baseline_f")
            baseline_f = (np.asarray(baseline_f, dtype=np.float64)
                          if baseline_f is not None else None)
            f_pred_abs = None
            if have_fpred and forces_known and f_preds_per_frame[i] is not None:
                f_pred_abs = np.asarray(f_preds_per_frame[i], dtype=np.float64)
                if baseline_f is not None:
                    f_pred_abs = f_pred_abs + baseline_f
            ref_forces = m.get("ref_forces") if forces_known else None
            bsl_forces = baseline_f if (baseline_used and forces_known) else None
            props = ["species:S:1", "pos:R:3"]
            if ref_forces is not None:
                props.append("REF_forces:R:3")
            if bsl_forces is not None:
                props.append("baseline_forces:R:3")
            if f_pred_abs is not None:
                props.append("pred_forces:R:3")
            comment_parts = []
            lat = m.get("lattice")
            if lat is not None:
                lat = np.asarray(lat).reshape(-1)
                comment_parts.append('Lattice="' + " ".join(f"{v:.8f}" for v in lat) + '"')
            comment_parts.append("Properties=" + ":".join(props))
            comment_parts.append(f"pred_energy={e_pred_abs:.10f}")
            if ref_energy is not None:
                comment_parts.append(f"REF_energy={float(ref_energy):.10f}")
            if baseline_used:
                comment_parts.append(f"baseline_energy={baseline_e:.10f}")
            pbc = m.get("pbc")
            if pbc is not None:
                pbc_s = " ".join("T" if bool(p) else "F" for p in np.asarray(pbc).reshape(-1))
                comment_parts.append(f'pbc="{pbc_s}"')
            fout.write(f"{natoms}\n")
            fout.write(" ".join(comment_parts) + "\n")
            for a in range(natoms):
                row = [syms[a], f"{coords[a][0]:.8f}", f"{coords[a][1]:.8f}", f"{coords[a][2]:.8f}"]
                if ref_forces is not None:
                    row += [f"{ref_forces[a][0]:.8f}", f"{ref_forces[a][1]:.8f}", f"{ref_forces[a][2]:.8f}"]
                if bsl_forces is not None:
                    row += [f"{bsl_forces[a][0]:.8f}", f"{bsl_forces[a][1]:.8f}", f"{bsl_forces[a][2]:.8f}"]
                if f_pred_abs is not None:
                    row += [f"{f_pred_abs[a][0]:.8f}", f"{f_pred_abs[a][1]:.8f}", f"{f_pred_abs[a][2]:.8f}"]
                fout.write(" ".join(row) + "\n")
            written += 1
    logger.info("Wrote predictions to %s (%d frames)", out_path, written)


def run_orbital(args):
    from orbace.data.dataset import OrbitalDataset
    from orbace.tools.utils import set_seeds
    set_seeds(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Loading model: {args.model}")
    model = _load_model(args.model, device)
    logger.info(f"Model on {device}")

    baseline_model = args.baseline_model
    baseline_basis = None if args.baseline_basis == "auto" else args.baseline_basis
    baseline_xc = None if args.baseline_xc == "auto" else args.baseline_xc
    if args.baseline_basis == "auto":
        baseline_basis = getattr(model, "baseline_basis", None)
    if args.baseline_xc == "auto":
        baseline_xc = getattr(model, "baseline_xc", None)
    if baseline_model == "auto":
        baseline_model = getattr(model, "baseline_model", "none") or "none"
        _meta = None
        meta_path = os.path.join(args.npz_data, "metadata.json") if args.npz_data else None
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path) as _f:
                    _meta = json.load(_f)
            except Exception:
                _meta = None
        if _meta is not None:
            if baseline_model == "none":
                baseline_model = _meta.get("baseline_model", "none") or "none"
            if args.baseline_basis == "auto" and baseline_basis is None:
                baseline_basis = _meta.get("baseline_basis")
            if args.baseline_xc == "auto" and baseline_xc is None:
                baseline_xc = _meta.get("baseline_xc")
    if baseline_model and baseline_model != "none":
        logger.info(f"Delta-learning baseline: {baseline_model}"
                    + (f" (basis={baseline_basis})" if baseline_basis else "")
                    + (f" (xc={baseline_xc})" if baseline_xc else ""))

    orb_win = args.orbital_window if args.orbital_window and args.orbital_window > 0 else None
    if orb_win:
        logger.info(f"Orbital window pruning: keeping HOMO-{orb_win}..LUMO+{orb_win}")
    want_xyz = args.output_xyz is not None
    baseline_used = bool(baseline_model and baseline_model != "none")
    frame_meta = None
    _ectx = os.path.basename(os.path.dirname(os.path.abspath(args.model)))

    eval_dataset = None
    data_list = None
    if args.npz_data:
        logger.info(f"Loading prepared data: {args.npz_data}")
        with timed("orbital", "eval", "load", _ectx):
            if want_xyz:
                # need per-frame metadata -> materialise the graphs
                dataset = OrbitalDataset(data_dir=args.npz_data, r_cut=args.r_max,
                                         orbital_window=orb_win,
                                         max_frames=args.n_frames)
                data_list = dataset.data
                eval_dataset = dataset
            else:
                # stream from disk (flat memory) -- required for the ~442k test set
                eval_dataset = OrbitalDataset(data_dir=args.npz_data, r_cut=args.r_max,
                                              orbital_window=orb_win, lazy=True,
                                              max_frames=args.n_frames)
        if want_xyz:
            e0s = {}
            meta_path = os.path.join(args.npz_data, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as _f:
                    e0s = json.load(_f).get("e0s", {})
            frame_meta = _orbital_build_meta_from_npz(data_list, e0s, baseline_used)
    else:
        logger.info(f"Preparing from xyz: {args.xyz}")
        with timed("orbital", "eval", "prepare", _ectx):
            prep = _orbital_prepare_from_xyz(
                args.xyz, args.basis, args.localizer, args.charge, args.spin, args.r_max,
                save_dir=args.save_npz_dir, baseline_model=baseline_model,
                orbital_window=orb_win, collect_meta=want_xyz, e0s_json=args.e0s_json,
                baseline_basis=baseline_basis, baseline_xc=baseline_xc)
        data_list, frame_meta = prep if want_xyz else (prep, None)

    n_struct = len(eval_dataset) if eval_dataset is not None else len(data_list)
    logger.info(f"Evaluating {n_struct} structures...")
    with timed("orbital", "eval", "forward", _ectx):
        results = _orbital_evaluate(model, data_list, device, args.batch_size,
                                    dataset=eval_dataset)
    _orbital_print_results(results)
    if want_xyz:
        _orbital_write_predictions_xyz(
            args.output_xyz, frame_meta, results["e_preds"], results.get("f_preds_per_frame"))


# ============================================================================
#  AtomOrbACE evaluation (atom-centered, conservative)
# ============================================================================
def _atom_forward(model, data, device):
    from orbace.data.invariant_graph import collate_invariant_data
    batch = collate_invariant_data([data])
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.enable_grad():
        out = model(batch, training=False, compute_force=True)
    return out


def _atom_predict(model, data, device):
    out = _atom_forward(model, data, device)
    return float(out["energy"].detach().cpu().item()), out["forces"].detach().cpu().numpy()


def _atom_predict_mf(model, data, device):
    """MF model prediction. The model already outputs the TOTAL (base + delta)
    energy/forces (E0-subtracted), so we just return them -- no physics baseline
    recompute, and no separate base term to add back."""
    out = _atom_forward(model, data, device)
    e = float(out["energy"].detach().cpu().item())
    f = out["forces"].detach().cpu().numpy()
    return e, f


def _atom_eval_npz(model, ds, device, batch_size, e0s, is_mf, bucket,
                   num_workers=0):
    """Per-frame energy/force errors over a prepared npz set, batched.

    One frame per forward pass leaves the GPU almost idle: the 442k-frame test
    set cost 5-8h that way, more than the training stage it was evaluating.
    Batching is sound here because frames never interact -- the graph has no
    cross-frame edges, so the batch energy is a plain sum and autograd's
    dE_total/dR_i is exactly frame i's own force, the same property training
    relies on. Errors are still accumulated per frame, so mean/median/max keep
    the meaning they had before.

    Returns (per-frame energy MAE/atom, per-frame force MAE, per-frame |dE|,
    per-frame mean squared force error, per-frame residual count for forces).
    """
    from orbace.data.invariant_graph import (
        BucketedBatchSampler, collate_invariant_data)

    lk = dict(collate_fn=collate_invariant_data, num_workers=num_workers,
              pin_memory=(device.type == "cuda"))
    if num_workers > 0:
        lk["persistent_workers"] = True
        lk["prefetch_factor"] = 4

    sampler = None
    if bucket:
        try:
            # shuffle=False sorts globally, which is both optimal for padding and
            # reproducible; eval metrics do not depend on frame order.
            sampler = BucketedBatchSampler(
                ds.tb_ao_counts(), batch_size, shuffle=False)
        except (AttributeError, RuntimeError, KeyError, OSError) as e:
            logger.warning(f"Eval bucketing disabled ({e}); using sequential batches")
    loader = (torch.utils.data.DataLoader(ds, batch_sampler=sampler, **lk)
              if sampler is not None else
              torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                          shuffle=False, **lk))

    # Z -> isolated-atom E0 as a lookup array, so the per-frame sums vectorize.
    zmax = max([int(k) for k in e0s] or [0])
    e0_arr = np.zeros(zmax + 2)
    for k, v in e0s.items():
        e0_arr[int(k)] = float(v)

    eE, eF, eE_tot, eF_sq, f_counts = [], [], [], [], []
    for batch in loader:
        dev_batch = {k: (v.to(device, non_blocking=True)
                         if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
        with torch.enable_grad():
            out = model(dev_batch, training=False, compute_force=True)

        nat = batch["n_atoms"].numpy()
        offs = np.concatenate(([0], np.cumsum(nat)[:-1]))
        Z = batch["atom_numbers"].numpy()
        e0_sum = np.add.reduceat(e0_arr[np.clip(Z, 0, zmax + 1)], offs)

        e_pred = out["energy"].detach().cpu().numpy().astype(np.float64)
        base_e = (batch["baseline_energy"].numpy().astype(np.float64)
                  if "baseline_energy" in batch else np.zeros_like(e_pred))

        if "energy" in batch:
            e_ref = batch["energy"].numpy().astype(np.float64)
            # Same arithmetic as the per-frame path: put prediction and reference
            # on the absolute scale and subtract there. For MF the prediction
            # already contains the learned base, so only the reference gains the
            # stored physics baseline.
            if is_mf:
                d_e = np.abs((e_pred + e0_sum) - (e_ref + base_e + e0_sum))
            else:
                d_e = np.abs((e_pred + e0_sum + base_e)
                             - (e_ref + e0_sum + base_e))
            eE_tot.extend(d_e.tolist())
            eE.extend((d_e / np.maximum(nat, 1)).tolist())

        f_pred = out.get("forces")
        if f_pred is not None and "forces" in batch:
            f_pred = f_pred.detach().cpu().numpy().astype(np.float64)
            f_ref = batch["forces"].numpy().astype(np.float64)
            if is_mf and "baseline_forces" in batch:
                f_ref = f_ref + batch["baseline_forces"].numpy().astype(np.float64)
            # mean|df| over each frame's atoms AND xyz components
            df = f_pred - f_ref
            per_atom = np.abs(df).sum(axis=1)
            eF.extend((np.add.reduceat(per_atom, offs) / (nat * 3)).tolist())
            per_atom_sq = (df ** 2).sum(axis=1)
            eF_sq.extend((np.add.reduceat(per_atom_sq, offs) / (nat * 3)).tolist())
            f_counts.extend((nat * 3).tolist())

    return eE, eF, eE_tot, eF_sq, f_counts


def _atom_fresh(symbols, coords, r_cut):
    """Fresh graph for AtomOrbACE: NO HF SCF. Node/edge features are filled
    with zeros (the model recomputes differentiable extended-Hückel features
    from geometry); only positions/atomic-numbers/edges are needed."""
    import numpy as np
    from ase.data import chemical_symbols
    from orbace.data.tb_features import N_NODE_SCALARS, PAIR_KEYS_DEFAULT
    from orbace.data.invariant_graph import build_invariant_graph
    coords = np.asarray(coords, float)
    n = len(symbols)
    Z = np.array([chemical_symbols.index(s) for s in symbols], dtype=np.int64)
    feats = {"atom_positions": coords.astype(np.float32), "atom_numbers": Z,
             "node_scalars": np.zeros((n, N_NODE_SCALARS), np.float32)}
    for k in PAIR_KEYS_DEFAULT:
        feats[k] = np.zeros((n, n), np.float32)
    return build_invariant_graph(feats, r_cut=r_cut)


def _run_atom_direct(args):
    """AtomOrbACE: conservative autograd forces through in-forward TB features."""
    from ase.data import chemical_symbols
    from orbace.data.extxyz import read_extxyz
    from orbace.data.preprocessing import compute_baseline_energy
    from orbace.data.invariant_graph import InvariantDataset

    device = _resolve_device(args.device)
    model = _load_model(args.model, device)
    _ectx = os.path.basename(os.path.dirname(os.path.abspath(args.model)))
    e0s = {int(k): float(v) for k, v in getattr(model, "e0s", {}).items()}
    baseline_model = getattr(model, "baseline_model", "none")
    fidelity = model_fidelity_mode(model)
    is_mf = fidelity in ("mfdh", "mfah")
    args.r_cut = _resolve_r_cut(model, args.r_cut)
    print(f"[eval atom] model={args.model} device={device} baseline={baseline_model}"
          f" r_cut={args.r_cut:.3f}"
          f"{f' fidelity_method={fidelity} (learned cheap head)' if is_mf else ''}"
          f" features=Hückel (no SCF)")

    if args.npz_data:
        with timed("atom", "eval", "load", _ectx):
            # stream from disk (flat memory) -- required for the ~442k test set
            ds = InvariantDataset(data_dir=args.npz_data, r_cut=args.r_cut, lazy=True,
                                  max_frames=args.n_frames)
        _t_fwd = time.perf_counter()
        # Report ABSOLUTE total energy: E0 and the (delta) baseline that
        # prepare.py subtracted are added back to BOTH prediction and reference,
        # putting direct and delta models on one scale. The per-frame MAE is
        # unchanged, since those terms cancel, but the metric is unambiguous.
        eE, eF, eE_tot, eF_sq, f_counts = _atom_eval_npz(
            model, ds, device, args.batch_size, e0s, is_mf,
            bucket=args.eval_bucket, num_workers=args.num_workers)
        record_timing("atom", "eval", "forward", time.perf_counter() - _t_fwd, _ectx)
        _atom_print_metrics("[npz]", len(ds), eE, eF, eE_tot, eF_sq, f_counts)

    if args.xyz:
        frames = [f for f in read_extxyz(args.xyz) if f.get("config_type") != "IsolatedAtom"]
        if args.n_frames > 0:
            frames = frames[:args.n_frames]
        eE, eF, eE_tot, eF_sq, f_counts = [], [], [], [], []
        fd_done = 0
        _t_xyz = time.perf_counter()
        _t_fd_acc = 0.0
        for fi, fr in enumerate(frames):
            sym, coords = fr["symbols"], np.asarray(fr["coords"], float)
            Z = np.array([chemical_symbols.index(s) for s in sym])
            d = _atom_fresh(sym, coords, args.r_cut)
            e0_sum = sum(e0s.get(int(z), 0.0) for z in Z)
            if is_mf:
                # Learned base head replaces the physics baseline: no xTB/HF
                # baseline SCF recompute here, and no SCF for the features.
                e_pred, f_pred = _atom_predict_mf(model, d, device)
                e_abs = e_pred + e0_sum
                f_abs = f_pred
            else:
                e_pred, f_pred = _atom_predict(model, d, device)
                base_e, base_f = 0.0, None
                if baseline_model and baseline_model != "none":
                    try:
                        base_e, base_f = compute_baseline_energy(
                            model=baseline_model, symbols=sym, coords=coords, basis_hf=args.basis,
                            baseline_basis=getattr(model, "baseline_basis", None),
                            baseline_xc=getattr(model, "baseline_xc", None), with_forces=True)
                    except Exception as _e:  # e.g. xTB/HF SCF non-convergence on a frame
                        print(f"  [warn] baseline ({baseline_model}) failed on frame {fi}: "
                              f"{type(_e).__name__}: {_e}; skipping frame")
                        continue
                e_abs = e_pred + e0_sum + base_e
                f_abs = f_pred + (base_f if base_f is not None else 0.0)
            if fr["energy"] is not None:
                eE.append(abs(e_abs - fr["energy"]) / max(len(sym), 1))
                eE_tot.append(abs(e_abs - fr["energy"]))
            if fr["forces"] is not None:
                df = f_abs - fr["forces"]
                eF.append(float(np.mean(np.abs(df))))
                eF_sq.append(float(np.mean(df ** 2)))
                f_counts.append(len(sym) * 3)
            if fd_done < args.fd:
                _t_fd0 = time.perf_counter()
                n = len(sym); F_fd = np.zeros((n, 3))
                for a in range(n):
                    for k in range(3):
                        cp = coords.copy(); cp[a, k] += args.fd_h
                        cm = coords.copy(); cm[a, k] -= args.fd_h
                        _pred = _atom_predict_mf if is_mf else _atom_predict
                        _fresh = lambda s, c: _atom_fresh(s, c, args.r_cut)
                        ep, _ = _pred(model, _fresh(sym, cp), device)
                        em, _ = _pred(model, _fresh(sym, cm), device)
                        F_fd[a, k] = -(ep - em) / (2 * args.fd_h)
                dmax = float(np.max(np.abs(f_pred - F_fd)))
                print(f"  [FD] frame {fi}: max|F_auto - F_fd(own E)| = {dmax:.4f} eV/A "
                      f"({'conservative-ish' if dmax < 1e-2 else 'non-conservative'})")
                fd_done += 1
                _t_fd_acc += time.perf_counter() - _t_fd0
        record_timing("atom", "eval", "predict",
                      (time.perf_counter() - _t_xyz) - _t_fd_acc, _ectx)
        record_timing("atom", "eval", "fd_check", _t_fd_acc, _ectx)
        _atom_print_metrics("[xyz-fresh]", len(frames), eE, eF, eE_tot,
                            eF_sq, f_counts)


def run_atom(args):
    _run_atom_direct(args)


# ============================================================================
#  CLI
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(description="Unified OrbACE evaluation")
    p.add_argument("--method", required=True, choices=["orbital", "atom"],
                   help="orbital = localized-orbital OrbACE; atom = AtomOrbACE")
    p.add_argument("--model", required=True, help="Path to saved .pt model")
    # inputs (shared; --input kept as an alias of --xyz for backward-compat)
    p.add_argument("--xyz", "--input", dest="xyz", default=None,
                   help="xyz/extxyz file (features recomputed fresh)")
    p.add_argument("--npz_data", default=None, help="Directory of prepared .npz files")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    p.add_argument("--seed", type=int, default=42)
    # orbital-specific
    p.add_argument("--localizer", default="boys", choices=["boys", "pm", "canonical"])
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--r_max", type=float, default=6.0, help="cutoff (orbital method)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--save_npz_dir", default=None)
    p.add_argument("--e0s_json", default=None)
    p.add_argument("--output_xyz", default=None)
    p.add_argument("--baseline_model", default="auto")
    p.add_argument("--baseline_basis", default="auto")
    p.add_argument("--baseline_xc", default="auto")
    p.add_argument("--orbital_window", type=int, default=0)
    # atom-specific
    p.add_argument("--r_cut", type=float, default=None,
                   help="(atom) graph cutoff in Angstrom; defaults to the "
                        "r_max stored in the model, which is what you want. "
                        "Passing a different value warns and rebuilds the "
                        "neighbour list at that cutoff instead.")
    p.add_argument("--n_frames", type=int, default=0,
                   help="evaluate only the first N frames (0 = all); applies to "
                        "both --xyz and --npz_data")
    p.add_argument("--num_workers", type=int, default=0,
                   help="(atom) dataloader workers for --npz_data evaluation")
    p.add_argument("--eval_bucket", type=lambda x: str(x).lower() in ("true", "1", "yes"),
                   default=True,
                   help="(atom, TB backend) batch same-size molecules together so "
                        "the padded eigensolve does not dominate; no effect on metrics")
    p.add_argument("--fd", type=int, default=0, help="(atom) #frames for FD conservativeness check")
    p.add_argument("--fd_frames", type=int, default=3, help="(atom) deprecated alias, unused")
    p.add_argument("--fd_h", type=float, default=1e-3, help="(atom) FD step in Angstrom")
    return p


def main():
    args = build_parser().parse_args()
    if args.method == "orbital":
        if not (args.xyz or args.npz_data):
            raise SystemExit("ERROR: --method orbital needs --xyz/--input or --npz_data")
        run_orbital(args)
    else:
        run_atom(args)


if __name__ == "__main__":
    main()
