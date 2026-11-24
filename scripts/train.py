#!/usr/bin/env python3
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Unified OrbACE training -- one entry point for every model variant.

Each method keeps its own hyperparameter defaults; pick one with --method.

  # Orbital-space OrbACE:
  python train.py --method orbital --data_dir ./data --n_epochs 500
  python train.py --method orbital --config train.yaml

  # AtomOrbACE (conservative, cheap extended-Hückel / TB features, no SCF):
  python train.py --method atom --train_dir idata_train --val_dir idata_val
"""

import argparse
import json
import logging
import functools
import math
import os
import sys
from typing import Any, Dict

import numpy as np
import torch
import yaml

if hasattr(torch, "serialization"):
    torch.serialization.add_safe_globals([slice])

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orbace.tools.train import (
    Trainer, WeightedEnergyForcesStressLoss, WeightedHuberEnergyForcesStressLoss,
)
from orbace.data.fidelity import (
    LabelBalancedBatchSampler, auto_hf_per_batch, primary_data_dir)
from orbace.tools.utils import set_seeds, count_parameters, count_used_parameters
from orbace.tools.timing import timed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
#  Per-method hyperparameter defaults (each defines its own CLI surface).
# ============================================================================
ORBITAL_DEFAULTS = {
    "name": "OrbACE",
    "data_dir": None, "train_file": None, "valid_file": None,
    "train_dir": None, "val_dir": None,
    "checkpoint_dir": "./checkpoints", "results_dir": "./results",
    "foundation_model": None,
    "r_max": 5.0, "num_bessel": 8, "num_polynomial_cutoff": 5, "max_ell": 2,
    "num_interactions": 3, "hidden_channels": 64,
    "hidden_irreps": "64x0e + 64x1o + 64x2e", "mlp_channels": 128,
    "correlation": 3, "num_species": 200,
    "n_scalar_features": 10, "n_ao_shell_features": 15, "n_coeff_norm_features": 5,
    "avg_num_neighbors": 10.0,
    "use_long_range": False, "use_multipole_lr": True, "lr_alpha": 0.2, "lr_cutoff": 15.0,
    "use_chem_edge_features": True, "n_chem_edge_features": 6,
    "use_equivariant_orbital_features": True, "n_equiv_channels": 8,
    "pair_repulsion": True, "pair_repulsion_p": 6,
    "baseline_model": "none", "baseline_basis": None, "baseline_xc": None,
    "orbital_window": 0, "atom_head_multi": False, "use_bf16": False,
    "dropout": 0.05, "radial_MLP": "128,128,128",
    "loss": "mse", "huber_delta": 0.01,
    "n_epochs": 2000, "batch_size": 10, "lr": 0.01, "weight_decay": 5e-7,
    "optimizer": "adam", "amsgrad": True,
    "energy_weight": 1.0, "forces_weight": 10.0, "stress_weight": 0.0,
    "patience": 200, "seed": 42, "val_fraction": 0.1, "ema_decay": 0.999,
    "clip_grad": 10.0, "device": "auto", "default_dtype": "float32",
    "use_wandb": False, "wandb_project": "orbace", "wandb_name": None,
    "compute_stress": False, "restart": None, "num_workers": 0,
    "scheduler": "plateau", "scheduler_factor": 0.8, "scheduler_patience": 50,
    "warmup_epochs": 10, "error_table": "PerAtomMAE",
    "keep_checkpoints": True, "save_cpu": True,
    "swa": False, "start_swa": 750, "swa_lr": 0.00025,
    "swa_forces_weight": 100.0, "swa_energy_weight": 50.0,
    "stage_two_start": 1200, "stage_two_energy_weight": 10.0,
    "stage_two_forces_weight": 30.0, "stage_two_stress_weight": 0.0,
    "stage_two_lr": 0.0, "freeze_backbone": False,
    "keep_last_interaction": False, "keep_foundation_e0": False,
    # Multi-fidelity mode, same semantics as the atom method. "regular" =
    # physics-baseline delta (recompute baseline at inference). "mfdh" = learned
    # base head ADDED to the orbital delta. "mfah" = parallel absolute cheap
    # head that stays out of the total. Neither MF mode recomputes a baseline at
    # inference. mf_hf_forces=True supervises the absolute (wB97X) force.
    "fidelity_method": "regular",
    "base_energy_weight": 10.0, "base_forces_weight": 1.0,
    "mf_head_hidden": 64, "mf_hf_forces": True,
    # Labeled frames to force into each batch when only some frames carry a
    # high-fidelity energy. "auto" decides from the label density and does
    # nothing when shuffling already supplies enough; 0 disables it outright.
    "mf_hf_per_batch": "auto",
}

ATOM_DEFAULTS = {
    "name": "AtomOrbACE",
    "data_dir": None, "train_dir": None, "val_dir": None,
    "checkpoint_dir": "./checkpoints_inv", "results_dir": "./results_inv",
    "foundation_model": None,
    "r_max": 5.0, "num_bessel": 8, "num_polynomial_cutoff": 5, "max_ell": 2,
    "num_interactions": 3, "hidden_irreps": "64x0e + 64x1o + 64x2e",
    "mlp_channels": 128, "correlation": 3, "num_species": 100,
    "radial_MLP": "128,128,128", "avg_num_neighbors": 0.0,
    "use_edge_features": True, "pair_repulsion": True, "pair_repulsion_p": 6,
    # Ewald-damped electrostatics between atoms, past the r_max cutoff. Monopole
    # only: the multipole terms need a per-site direction, which the orbital
    # model gets from the frozen HF orbital polarization and an atom has no
    # equivalent of. Off by default, as in ORBITAL_DEFAULTS.
    "use_long_range": False, "lr_alpha": 0.2, "lr_cutoff": 15.0,
    "baseline_model": "none", "baseline_basis": None, "baseline_xc": None,
    "loss": "huber", "huber_delta": 0.01,
    # batch_size/lr below are the pre-rescale reference values; both are raised
    # together (see TB_FAST_BATCH_SIZE), since the TB backend is launch bound.
    "n_epochs": 500, "batch_size": 16, "lr": 0.01, "weight_decay": 5e-7,
    "optimizer": "adam", "amsgrad": True,
    "energy_weight": 10.0, "forces_weight": 100.0, "stress_weight": 0.0,
    "patience": 100, "seed": 42, "val_fraction": 0.1, "ema_decay": 0.999,
    "clip_grad": 10.0, "device": "auto", "num_workers": 0,
    "scheduler": "plateau", "scheduler_factor": 0.8, "scheduler_patience": 25,
    "warmup_epochs": 10, "save_cpu": True, "use_bf16": False,
    "swa": False, "start_swa": 0, "stage_two_start": 0,
    "stage_two_energy_weight": 20.0, "stage_two_forces_weight": 300.0,
    "stage_two_lr": 0.0, "freeze_backbone": False, "restart": None,
    "basis": "sto-3g",
    # Group same-size molecules into a batch (the padded
    # batched eigensolve makes a batch cost batch_size * nao_max^3). Set False to
    # fall back to plain random batches. bucket_pool_batches trades padding
    # savings against how heterogeneous each batch stays: a larger pool cuts
    # padding waste but correlates the compositions inside a batch. Held at 8,
    # the value the reference accuracy (job 526615) was measured with, because
    # raising it buys only ~3% epoch time and is not accuracy-neutral.
    "bucket_by_size": True,
    "bucket_pool_batches": 8,
    # Multi-fidelity mode. "regular" = today's physics-baseline delta (recompute
    # baseline at inference). "mfdh" (multi-fidelity delta head) = learned base
    # head ADDED to a QC-conditioned delta head. "mfah" (multi-fidelity absolute
    # head) = parallel head predicting the cheap level's absolute energy, kept
    # out of the total so the high-fidelity path matches a direct model. Neither
    # MF mode recomputes a baseline at inference.
    "fidelity_method": "regular",
    "base_energy_weight": 10.0, "base_forces_weight": 1.0,
    "mf_head_hidden": 64,
    # mf_hf_forces=True: supervise the high-fidelity total force (max accuracy).
    # False: pure multi-fidelity -- high fidelity is ENERGY-only and its force is
    # predicted purely as the gradient of the total energy (never trained on).
    "mf_hf_forces": True,
    # Labeled frames to force into each batch when only some frames carry a
    # high-fidelity energy. "auto" decides from the label density and does
    # nothing when shuffling already supplies enough; 0 disables it outright.
    "mf_hf_per_batch": "auto",
}

_DEFAULTS = {"orbital": ORBITAL_DEFAULTS, "atom": ATOM_DEFAULTS}

# The TB featurizer is launch-latency bound, not compute bound: a batch is a long
# tail of small kernels plus two eigendecompositions, so most of a batch's cost is
# fixed rather than per-sample. Batch 16 (right for the QC backend, which does no
# such work in-forward) therefore leaves the GPU idle; batch 64 amortizes that
# fixed cost over 4x the samples and runs 2.1x faster per epoch (38.2s vs 81.9s,
# job 535589). At a fixed 1000 epochs this costs ~6% test energy MAE, but at fixed
# wall-clock it is far ahead -- when bs64 finished, bs16 was only at epoch 466 and
# 38% worse on energy. Applies to the TB backend alone; QC keeps batch 16.
TB_FAST_BATCH_SIZE = 64


def _explicit_cli_keys() -> set:
    """Config keys the caller named on the command line (`--k v` or `--k=v`)."""
    return {a[2:].split("=", 1)[0] for a in sys.argv[1:] if a.startswith("--")}


def _migrate_fidelity_alias(cfg: Dict[str, Any], cli_delta_method, explicit: set) -> None:
    """Fold the retired --delta_method / delta_method: spellings into cfg.

    The flag was renamed to --fidelity_method when MF split into mfdh and mfah,
    since mfah has no delta at all. Both the old flag and the old key in a YAML
    config are still honored, and the old value "mf" maps to "mfdh".
    """
    old = cli_delta_method if cli_delta_method is not None else cfg.pop("delta_method", None)
    cfg.pop("delta_method", None)
    if old is None:
        return
    if "fidelity_method" in explicit:
        raise SystemExit(
            "ERROR: pass either --fidelity_method or the retired --delta_method, not both.")
    new = "mfdh" if str(old) == "mf" else str(old)
    cfg["fidelity_method"] = new
    explicit.add("fidelity_method")
    logger.warning(
        "delta_method is deprecated: using --fidelity_method %s. "
        "(MF split into mfdh, this additive delta head, and mfah, parallel "
        "absolute heads.)", new)


def _apply_tb_fast_defaults(cfg: Dict[str, Any], explicit: set) -> None:
    """Raise the TB batch size unless the caller chose one.

    The learning rate moves with it by sqrt(ratio), including when the caller set
    --lr, because the two were validated as a pair: bs64 was measured at exactly
    2x the bs16 rate, and leaving a bs16 rate on a 4x batch is a config nobody has
    run. Pass --batch_size explicitly to opt out of both halves.
    """
    old_bs = int(cfg["batch_size"])
    if "batch_size" in explicit or old_bs >= TB_FAST_BATCH_SIZE:
        return
    scale = math.sqrt(TB_FAST_BATCH_SIZE / old_bs)
    old_lr = float(cfg["lr"])
    cfg["batch_size"] = TB_FAST_BATCH_SIZE
    cfg["lr"] = old_lr * scale
    logger.info(
        f"TB fast defaults: batch_size {old_bs} -> {cfg['batch_size']}, "
        f"lr {old_lr:.6g} -> {cfg['lr']:.6g} (sqrt scaling). "
        f"Pass --batch_size {old_bs} to keep the previous setting.")


def parse_args() -> Dict[str, Any]:
    # Pass 1: figure out the method (selects the defaults / CLI surface).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--method", choices=["orbital", "atom"])
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()
    if known.method is None:
        raise SystemExit("ERROR: --method {orbital,atom} is required "
                         "(e.g. `python train.py --method atom --help`)")
    defaults = _DEFAULTS[known.method]

    # Pass 2: build the full parser from that method's defaults.
    p = argparse.ArgumentParser(description=f"Train OrbACE ({known.method})")
    p.add_argument("--method", choices=["orbital", "atom"], required=True)
    p.add_argument("--config", default=None, help="YAML config (overrides CLI args)")
    for k, v in defaults.items():
        dt = type(v) if v is not None else str
        if dt is bool:
            p.add_argument(f"--{k}", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=v)
        else:
            p.add_argument(f"--{k}", type=dt, default=v)
    # Pre-split spelling, kept working so old scripts and configs still run.
    p.add_argument("--delta_method", default=None, help=argparse.SUPPRESS)
    a = p.parse_args()

    explicit = _explicit_cli_keys()
    if a.config:
        with open(a.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg = {**defaults, **file_cfg}
        explicit |= set(file_cfg)
    else:
        cfg = {k: getattr(a, k) for k in defaults}
    cfg["method"] = known.method
    _migrate_fidelity_alias(cfg, a.delta_method, explicit)
    if known.method == "atom":
        _apply_tb_fast_defaults(cfg, explicit)
    return cfg


# ============================================================================
#  Orbital-space OrbACE
# ============================================================================
def _count_used(build_fn, n_present):
    """count_used_parameters with the builders' own log lines silenced."""
    prev = logger.level
    logger.setLevel(logging.WARNING)
    try:
        return count_used_parameters(build_fn, n_present)
    finally:
        logger.setLevel(prev)


def _log_parameter_count(n_allocated, n_used, n_present, what, n_slots):
    if n_used is None or n_used == n_allocated:
        logger.info(f"Model parameters: {n_allocated:,}")
    else:
        logger.info(f"Model parameters: {n_used:,} for the {n_present} {what} in the "
                    f"training data ({n_allocated:,} allocated over {n_slots} slots)")


def _orbital_build_model(cfg, atom_e0_correction=None, avg_neighbors=None, mean=None, rms=None,
                         base_atom_e0=None):
    from orbace.modules.models import OrbACE
    irreps_str = cfg.get("hidden_irreps", "") or f"{cfg['hidden_channels']}x0e"
    logger.info(f"Hidden irreps: {irreps_str}")
    radial_mlp = [int(x) for x in cfg.get("radial_MLP", "64,64,64").split(",")]
    return OrbACE(
        r_max=cfg["r_max"], num_bessel=cfg["num_bessel"],
        num_polynomial_cutoff=cfg["num_polynomial_cutoff"], max_ell=cfg["max_ell"],
        num_interactions=cfg["num_interactions"], num_species=cfg["num_species"],
        n_scalar_features=cfg["n_scalar_features"], n_ao_shell_features=cfg["n_ao_shell_features"],
        n_coeff_norm_features=cfg.get("n_coeff_norm_features", 5),
        hidden_irreps=irreps_str, MLP_irreps=f"{cfg['mlp_channels']}x0e",
        correlation=cfg["correlation"], radial_MLP=radial_mlp,
        avg_num_neighbors=avg_neighbors or cfg["avg_num_neighbors"],
        atomic_inter_scale=rms or 1.0, atomic_inter_shift=mean or 0.0,
        atom_e0_correction=atom_e0_correction,
        use_chem_edge_features=cfg.get("use_chem_edge_features", True),
        n_chem_edge_features=cfg.get("n_chem_edge_features", 6),
        use_equivariant_orbital_features=cfg.get("use_equivariant_orbital_features", True),
        n_equiv_channels=cfg.get("n_equiv_channels", 8), dropout=cfg.get("dropout", 0.0),
        use_long_range=cfg.get("use_long_range", False),
        use_multipole_lr=cfg.get("use_multipole_lr", True),
        lr_alpha=cfg.get("lr_alpha", 0.2), lr_cutoff=cfg.get("lr_cutoff", 15.0),
        pair_repulsion=cfg.get("pair_repulsion", True), pair_repulsion_p=cfg.get("pair_repulsion_p", 6),
        baseline_model=cfg.get("baseline_model", "none"),
        baseline_basis=cfg.get("baseline_basis", None), baseline_xc=cfg.get("baseline_xc", None),
        atom_head_multi=cfg.get("atom_head_multi", False),
        delta_method=str(cfg.get("fidelity_method", "regular")),
        base_atom_e0=base_atom_e0, mf_head_hidden=int(cfg.get("mf_head_hidden", 64)),
    )


def _build_optimizer(model, cfg):
    name = cfg.get("optimizer", "adamw").lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], amsgrad=cfg.get("amsgrad", True))
    return torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg["weight_decay"], amsgrad=cfg.get("amsgrad", True))


def _build_scheduler(optimizer, cfg):
    name = cfg.get("scheduler", "cosine")
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.get("scheduler_factor", 0.5),
            patience=cfg.get("scheduler_patience", 20), min_lr=1e-6)
    if name == "cosine":
        t_max = max(1, int(cfg["n_epochs"] - cfg.get("warmup_epochs", 0)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.get("scheduler_patience", 50), gamma=cfg.get("scheduler_factor", 0.5))
    return None


def run_orbital(cfg):
    import torch.nn as nn
    from orbace.data.dataset import OrbitalDataset, collate_orbital_data
    from orbace.tools.utils import (
        compute_orbital_e0s, compute_statistics,
    )

    set_seeds(cfg["seed"])
    if cfg.get("default_dtype") == "float64":
        torch.set_default_dtype(torch.float64)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg["device"] == "auto" else torch.device(cfg["device"]))
    logger.info(f"Device: {device}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg.get("results_dir", "./results"), exist_ok=True)
    with open(os.path.join(cfg["checkpoint_dir"], "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    fidelity_method = str(cfg.get("fidelity_method", "regular"))
    mf_flag = fidelity_method in ("mfdh", "mfah")
    if mf_flag:
        logger.info("%s mode: datasets reconstruct absolute targets (residual + baseline); "
                    "no physics baseline is recomputed at inference.",
                    fidelity_method.upper())

    orb_win = cfg.get("orbital_window", None)
    if orb_win is not None and int(orb_win) > 0:
        logger.info(f"Orbital window pruning: keeping HOMO-{orb_win}..LUMO+{orb_win}")
    _tctx = os.path.basename(str(cfg["checkpoint_dir"]).rstrip("/"))
    with timed("orbital", "train", "data_load", _tctx):
        if cfg.get("train_dir") and cfg.get("val_dir"):
            train_set = OrbitalDataset(data_dir=cfg["train_dir"], r_cut=cfg["r_max"],
                                       orbital_window=orb_win, mf=mf_flag)
            val_set = OrbitalDataset(data_dir=cfg["val_dir"], r_cut=cfg["r_max"],
                                     orbital_window=orb_win, mf=mf_flag)
        elif cfg.get("data_dir"):
            dataset = OrbitalDataset(data_dir=cfg["data_dir"], r_cut=cfg["r_max"],
                                     orbital_window=orb_win, mf=mf_flag)
            ds_labeled = dataset.labeled_frames()
            if all(ds_labeled):
                n_val = max(1, int(len(dataset) * cfg["val_fraction"]))
                train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
            else:
                train_set, val_set = _stratified_split(
                    dataset, ds_labeled, cfg["val_fraction"], cfg["seed"])
        else:
            raise ValueError("Provide --data_dir or (--train_dir + --val_dir)")
    logger.info(f"Train: {len(train_set)} | Val: {len(val_set)}")
    train_labeled, n_hf_per_batch = _fidelity_labels(train_set, cfg, mf_flag)

    max_species = 0
    species_present, elements_present = set(), set()
    for _i in range(len(train_set)):
        _item = train_set[_i]
        _sp = _item.orbital_species
        if _sp.numel():
            max_species = max(max_species, int(_sp.max().item()))
            species_present.update(_sp.tolist())
        _z = getattr(_item, "atom_numbers", None)
        if _z is not None:
            elements_present.update(_z.tolist())
    required = max_species + 1
    if required > cfg["num_species"]:
        logger.info(f"Auto-sizing num_species {cfg['num_species']} -> {required} "
                    f"(max orbital species in data = {max_species})")
        cfg["num_species"] = required

    # First of a possibly comma-separated spec: the sidecar e0s.json /
    # metadata.json are verified identical across the directories when the
    # dataset is built, so any one of them answers for all.
    data_dir = primary_data_dir(cfg.get("data_dir") or cfg.get("train_dir"))
    if cfg.get("baseline_model", "none") == "none" and data_dir:
        meta_path = os.path.join(data_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as _f:
                    _meta = json.load(_f)
                _bm = _meta.get("baseline_model", "none")
                if _bm and _bm != "none":
                    cfg["baseline_model"] = _bm
                    if cfg.get("baseline_basis") is None:
                        cfg["baseline_basis"] = _meta.get("baseline_basis")
                    if cfg.get("baseline_xc") is None:
                        cfg["baseline_xc"] = _meta.get("baseline_xc")
                    logger.info(f"Detected delta-learning baseline from metadata: {_bm}")
            except Exception:
                pass

    atom_e0_corr, _e0_base, _e0_slope = compute_orbital_e0s(
        train_set, num_species=cfg["num_species"], data_dir=data_dir)
    avg_neighbors, mean, rms = compute_statistics(
        train_set, atom_e0_correction=atom_e0_corr, num_species=cfg["num_species"])
    logger.info(f"Stats: avg_neighbors={avg_neighbors:.1f}, shift={mean:.6f}, scale={rms:.6f}")

    base_atom_e0 = None
    if mf_flag:
        if cfg.get("baseline_model", "none") in (None, "none"):
            raise SystemExit(
                f"ERROR: --fidelity_method {fidelity_method} requires a cheap-fidelity "
                "baseline (the level the cheap head learns). Prepare the data with "
                "--baseline_model {xtb,gfn2,hf_sto3g,...} first.")
        base_atom_e0 = _atom_fit_base_e0(train_set, cfg["num_species"])
        if base_atom_e0 is None:
            raise SystemExit(
                f"ERROR: --fidelity_method {fidelity_method} needs baseline_energy in every "
                "frame, but none were found in the prepared npz. Re-prepare with a baseline.")
        logger.info(
            "MFDH mode: learned base head (fidelity=%s) ADDED to the orbital delta head"
            if fidelity_method == "mfdh" else
            "MFAH mode: parallel absolute cheap head (fidelity=%s) alongside the orbital readout",
            cfg.get("baseline_model"))

    model = _orbital_build_model(cfg, atom_e0_correction=atom_e0_corr,
                                 avg_neighbors=avg_neighbors, mean=mean, rms=rms,
                                 base_atom_e0=base_atom_e0)
    if data_dir:
        _e0_path = os.path.join(data_dir, "e0s.json")
        if os.path.exists(_e0_path):
            try:
                with open(_e0_path) as _f:
                    model.e0s = {int(k): float(v) for k, v in json.load(_f).items()}
                logger.info(f"Stored {len(model.e0s)} reference E0s on model for inference")
            except Exception as _e:
                logger.warning(f"Could not attach e0s.json to model: {_e}")

    if cfg.get("foundation_model"):
        fpath = cfg["foundation_model"]
        logger.info(f"Loading foundation weights from: {fpath}")
        ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            src_sd = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            src_sd = ckpt
        elif isinstance(ckpt, nn.Module):
            src_sd = ckpt.state_dict()
        else:
            raise ValueError(f"Cannot extract state_dict from {fpath}")
        tgt_sd = model.state_dict()
        loaded, skipped = [], []
        for k, v in src_sd.items():
            if k in tgt_sd and tgt_sd[k].shape == v.shape:
                tgt_sd[k] = v; loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(tgt_sd, strict=False)
        logger.info(f"Foundation weights: loaded {len(loaded)}/{len(src_sd)} keys, skipped {len(skipped)}")
        # Finetune dirs are prepared with --skip_meta (no e0s.json), so inherit the
        # reference E0s from the foundation checkpoint. Without this the finetuned
        # model has empty e0s and eval reports on the e0-subtracted (not absolute)
        # scale.
        if not getattr(model, "e0s", None):
            found_e0s = None
            if isinstance(ckpt, nn.Module):
                found_e0s = getattr(ckpt, "e0s", None)
            elif isinstance(ckpt, dict):
                found_e0s = ckpt.get("e0s")
            if found_e0s:
                model.e0s = {int(k): float(v) for k, v in found_e0s.items()}
                logger.info(f"Inherited {len(model.e0s)} isolated-atom E0s from foundation model")
        if cfg.get("keep_foundation_e0"):
            logger.info("Keeping foundation model's E0, scale/shift (--keep_foundation_e0)")
        else:
            model.scale_shift.shift.fill_(mean)
            model.scale_shift.scale.fill_(rms)
            if hasattr(model, "orbital_energies_fn"):
                model.orbital_energies_fn.e0_base.zero_()
            if hasattr(model, "atom_e0") and atom_e0_corr is not None:
                new_buf = torch.tensor(atom_e0_corr, dtype=model.atom_e0.dtype)
                if new_buf.shape != model.atom_e0.shape:
                    model.atom_e0 = new_buf
                else:
                    model.atom_e0.copy_(new_buf)
            logger.info("Scale/shift and E0s updated for current dataset")

    if cfg.get("freeze_backbone"):
        keep_last = cfg.get("keep_last_interaction", False)
        model.freeze_backbone(keep_last_interaction=keep_last)
        logger.info(f"Backbone frozen (last interaction {'kept' if keep_last else 'frozen'})")

    n_used = None
    if not cfg.get("freeze_backbone"):
        n_used = _count_used(lambda ns: _orbital_build_model({**cfg, "num_species": ns}),
                             len(species_present))
        head_w = getattr(model, "atom_head_out_weight", None)
        if n_used is not None and head_w is not None and elements_present:
            # The multi-element atom head is indexed by atomic number, not by
            # orbital species, so its slices follow the elements present.
            n_used += (len(elements_present) - len(species_present)) * (head_w.shape[1] + 1)
    _log_parameter_count(count_parameters(model), n_used, len(species_present),
                         "orbital species", cfg["num_species"])

    _nw = int(cfg["num_workers"])
    _loader_kw = dict(collate_fn=collate_orbital_data, num_workers=_nw, pin_memory=device.type == "cuda")
    if _nw > 0:
        _loader_kw["persistent_workers"] = True
        _loader_kw["prefetch_factor"] = 4
    if n_hf_per_batch:
        train_loader = DataLoader(
            train_set, batch_sampler=LabelBalancedBatchSampler(
                train_labeled, cfg["batch_size"], n_hf_per_batch,
                seed=int(cfg.get("seed", 0))), **_loader_kw)
    else:
        train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, **_loader_kw)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, **_loader_kw)

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    if mf_flag:
        from orbace.tools.train import MFEnergyForcesLoss
        hf_forces = bool(cfg.get("mf_hf_forces", True))
        loss_fn = MFEnergyForcesLoss(
            energy_weight=cfg["energy_weight"], forces_weight=cfg["forces_weight"],
            base_energy_weight=float(cfg.get("base_energy_weight", 10.0)),
            base_forces_weight=float(cfg.get("base_forces_weight", 1.0)),
            forces_normalizer=rms, huber=(cfg.get("loss", "mse").lower() == "huber"),
            huber_delta=cfg.get("huber_delta", 0.01), hf_forces=hf_forces)
        logger.info("Loss: %s (hf_forces=%s, base_E_w=%.1f, base_F_w=%.1f, huber=%s)",
                    fidelity_method.upper(), hf_forces, cfg.get("base_energy_weight", 10.0),
                    cfg.get("base_forces_weight", 1.0), cfg.get("loss", "mse").lower() == "huber")
        if not hf_forces:
            logger.info("MF energy-only: high-fidelity FORCES are NOT supervised; "
                        "the high-level force is the gradient of the total energy.")
    elif cfg.get("loss", "mse").lower() == "huber":
        loss_fn = WeightedHuberEnergyForcesStressLoss(
            energy_weight=cfg["energy_weight"], forces_weight=cfg["forces_weight"],
            stress_weight=cfg["stress_weight"], huber_delta=cfg.get("huber_delta", 0.01),
            forces_normalizer=rms)
        logger.info("Loss: Huber (delta=%s, force_normalizer=%.6f)", cfg.get("huber_delta", 0.01), rms)
    else:
        loss_fn = WeightedEnergyForcesStressLoss(
            energy_weight=cfg["energy_weight"], forces_weight=cfg["forces_weight"],
            stress_weight=cfg["stress_weight"], forces_normalizer=rms)
        logger.info("Loss: MSE (force_normalizer=%.6f)", rms)

    if cfg["use_wandb"]:
        try:
            import wandb
            wandb.init(project=cfg.get("wandb_project", "orbace"),
                       name=cfg.get("wandb_name") or cfg.get("name", "OrbACE"), config=cfg)
        except ImportError:
            logger.warning("wandb not installed, skipping")
            cfg["use_wandb"] = False

    st_start = int(cfg.get("stage_two_start") or 0) or None
    st_lr = cfg.get("stage_two_lr")
    st_lr = st_lr if (st_lr and st_lr > 0) else None
    st_sw = cfg.get("stage_two_stress_weight")
    st_sw = st_sw if (st_sw and st_sw > 0) else None
    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,
        ema_decay=cfg["ema_decay"], use_swa=cfg.get("swa", False), swa_start=cfg.get("start_swa", 0),
        max_grad_norm=cfg["clip_grad"], warmup_epochs=cfg.get("warmup_epochs", 15), device=device,
        checkpoint_dir=cfg["checkpoint_dir"], use_wandb=cfg["use_wandb"],
        compute_stress=cfg["compute_stress"], save_cpu=cfg.get("save_cpu", True),
        stage_two_start=st_start,
        stage_two_energy_weight=float(cfg.get("stage_two_energy_weight", 1000.0)),
        stage_two_forces_weight=float(cfg.get("stage_two_forces_weight", 100.0)),
        stage_two_stress_weight=st_sw, stage_two_lr=st_lr, use_bf16=cfg.get("use_bf16", False))
    if cfg.get("restart"):
        epoch0 = trainer.load_checkpoint(cfg["restart"])
        logger.info(f"Restarted from epoch {epoch0}")

    logger.info("Starting training...")
    with timed("orbital", "train", "fit", _tctx):
        trainer.fit(train_loader=train_loader, val_loader=val_loader,
                    n_epochs=cfg["n_epochs"], patience=cfg["patience"])
    logger.info(f"Training complete. Checkpoints at {cfg['checkpoint_dir']}/")


# ============================================================================
#  AtomOrbACE (atom-centered)
# ============================================================================
def _dataset_view(subset, method: str):
    """Call ``method()`` on a dataset, seeing through random_split's Subset."""
    if hasattr(subset, "dataset") and hasattr(subset, "indices"):
        values = getattr(subset.dataset, method)()
        return [values[i] for i in subset.indices]
    return getattr(subset, method)()


def _resolve_hf_per_batch(cfg, n_labeled: int, n_total: int, batch_size: int):
    """Interpret --mf_hf_per_batch. Returns (count, chosen_automatically)."""
    raw = str(cfg.get("mf_hf_per_batch", "auto")).strip().lower()
    if raw in ("", "auto"):
        return auto_hf_per_batch(n_labeled, n_total, batch_size), True
    try:
        n = int(raw)
    except ValueError:
        raise SystemExit(
            f"ERROR: --mf_hf_per_batch takes 'auto' or an integer, got {raw!r}.")
    if n < 0:
        raise SystemExit(f"ERROR: --mf_hf_per_batch cannot be negative, got {n}.")
    if n >= batch_size:
        raise SystemExit(
            f"ERROR: --mf_hf_per_batch {n} leaves no room for cheap frames in a "
            f"batch of {batch_size}. Use a value below --batch_size, or 0 to "
            "disable balancing.")
    if n > n_labeled:
        raise SystemExit(
            f"ERROR: --mf_hf_per_batch {n} exceeds the {n_labeled} labeled "
            "training frames available.")
    return n, False


def _fidelity_labels(train_set, cfg, mf_flag: bool):
    """Report high-fidelity label coverage and pick the per-batch label count.

    Returns ``(labeled, n_hf_per_batch)``. ``labeled`` is None when every frame
    carries a label, which is the case every run before multi-fidelity splits
    existed; the caller then takes exactly the path it always took.
    """
    labeled = _dataset_view(train_set, "labeled_frames")
    n_lab, n_tot = sum(labeled), len(labeled)
    if n_lab == n_tot:
        return None, 0
    if n_lab == 0:
        raise SystemExit(
            "ERROR: no training frame carries a high-fidelity energy. Prepared "
            "data stores 'energy' only for frames whose source carried a "
            "reference value, so the input geometries most likely had none.")
    if not mf_flag:
        raise SystemExit(
            f"ERROR: {n_tot - n_lab} of {n_tot} training frames have no energy "
            "label. Only the multi-fidelity arms can learn from unlabeled "
            "frames -- pass --fidelity_method {mfdh,mfah}, or point at a fully "
            "labeled set.")

    bs = int(cfg["batch_size"])
    n_hf, auto = _resolve_hf_per_batch(cfg, n_lab, n_tot, bs)
    density = n_lab / n_tot
    how = "auto" if auto else "requested"
    logger.info("Multi-fidelity labels: %s / %s training frames (%.2f%%) carry a "
                "high-fidelity energy.", f"{n_lab:,}", f"{n_tot:,}", 100 * density)
    if n_hf:
        logger.info("  Batch balancing ON (%s): %d labeled + %d cheap per batch of %d.",
                    how, n_hf, bs - n_hf, bs)
        logger.info("  The labeled set is revisited ~%.1fx per pass over the cheap "
                    "frames; watch the validation curve for overfitting to it.",
                    (n_hf / bs) / density)
    else:
        logger.info("  Batch balancing OFF (%s): shuffling already puts ~%.1f labeled "
                    "frames in a batch of %d.", how, bs * density, bs)
    logger.info("  Set --mf_hf_per_batch N to fix the count yourself, or 0 to disable.")
    return labeled, n_hf


def _stratified_split(dataset, labeled, val_fraction: float, seed: int):
    """Train/val split holding the labeled fraction equal on both sides.

    A plain random split of a sparsely labeled set can leave validation with no
    labeled frames at all, which would make the high-fidelity validation metric
    -- what early stopping and the LR schedule key on -- quietly empty.
    """
    g = torch.Generator().manual_seed(int(seed))
    train_idx, val_idx = [], []
    for group in ([i for i, f in enumerate(labeled) if f],
                  [i for i, f in enumerate(labeled) if not f]):
        if not group:
            continue
        n_val = min(max(1, int(round(len(group) * val_fraction))), len(group) - 1)
        perm = torch.randperm(len(group), generator=g).tolist()
        val_idx.extend(group[i] for i in perm[:n_val])
        train_idx.extend(group[i] for i in perm[n_val:])
    return Subset(dataset, sorted(train_idx)), Subset(dataset, sorted(val_idx))


def _atom_fit_base_e0(dataset, num_species: int):
    """Least-squares per-element E0 fit to the stored (cheap) baseline energies.

    Used only by the MF arms (mfdh, mfah): gives the cheap-level head a
    per-species offset so its MLP only has to learn the small interaction part
    of the baseline. Returns None when no frame carries a baseline_energy.
    """
    n = len(dataset)
    counts = np.zeros((n, num_species))
    b_energies = np.zeros(n)
    has_base = np.zeros(n, dtype=bool)
    for i in range(n):
        d = dataset[i]
        Z = d.atom_numbers.numpy().clip(0, num_species - 1)
        for z in Z:
            counts[i, z] += 1
        if getattr(d, "baseline_energy", None) is not None:
            b_energies[i] = float(d.baseline_energy)
            has_base[i] = True
    if not has_base.any():
        return None
    # Only the frames that carry a baseline enter the fit. Species composition
    # still comes from the whole set, so a species seen only in frames without
    # one gets the least-norm zero rather than a column of fictitious energies.
    present = counts.sum(0) > 0
    base_e0 = np.zeros(num_species, dtype=np.float64)
    if present.any():
        sol, *_ = np.linalg.lstsq(
            counts[np.ix_(has_base, present)], b_energies[has_base], rcond=None)
        base_e0[present] = sol
    return base_e0


def _atom_compute_stats(dataset, num_species: int):
    n = len(dataset)
    counts = np.zeros((n, num_species))
    energies = np.zeros(n)
    labeled = np.zeros(n, dtype=bool)
    n_atoms = np.zeros(n)
    total_edges = total_atoms = 0
    force_sq_sum = 0.0
    force_n = 0
    for i in range(n):
        d = dataset[i]
        Z = d.atom_numbers.numpy().clip(0, num_species - 1)
        for z in Z:
            counts[i, z] += 1
        if d.energy is not None:
            energies[i] = float(d.energy)
            labeled[i] = True
        n_atoms[i] = len(Z)
        total_edges += int(d.edge_index.shape[1])
        total_atoms += len(Z)
        if d.forces is not None:
            f = d.forces.numpy()
            force_sq_sum += float((f ** 2).sum()); force_n += f.size

    # E0 and the shift describe the energies, so they are fit over the frames
    # that have one. Under a multi-fidelity split most frames do not, and
    # letting them in as zeros would drag every E0 toward zero.
    present = counts.sum(0) > 0
    atom_e0 = np.zeros(num_species, dtype=np.float64)
    if present.any() and labeled.any():
        sol, *_ = np.linalg.lstsq(
            counts[np.ix_(labeled, present)], energies[labeled], rcond=None)
        atom_e0[present] = sol
    residual = (energies - counts @ atom_e0)[labeled]
    shift = (float(np.mean(residual / np.clip(n_atoms[labeled], 1, None)))
             if labeled.any() else 0.0)
    # avg_neighbors is pure geometry, so every frame counts toward it.
    rms = float(np.sqrt(force_sq_sum / max(force_n, 1))) if force_n else 1.0
    avg_neighbors = total_edges / max(total_atoms, 1)
    elements_present = set(np.flatnonzero(present).tolist())
    return atom_e0, avg_neighbors, shift, max(rms, 1e-6), elements_present


def run_atom(cfg):
    from orbace.data.invariant_graph import (
        BucketedBatchSampler, InvariantDataset, collate_invariant_data)
    from orbace.data.tb_features import N_NODE_SCALARS, N_EDGE_FEATURES
    from orbace.modules.atom_model import AtomOrbACE

    set_seeds(cfg["seed"])
    device = torch.device(
        "cuda" if (cfg["device"] in ("auto", "cuda") and torch.cuda.is_available())
        else (cfg["device"] if cfg["device"] not in ("auto",) else "cpu"))
    logger.info(f"Device: {device}")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    with open(os.path.join(cfg["checkpoint_dir"], "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    mf_flag = str(cfg.get("fidelity_method", "regular")) in ("mfdh", "mfah")
    if mf_flag:
        logger.info("%s mode: datasets reconstruct absolute targets (residual + baseline).",
                    str(cfg["fidelity_method"]).upper())
    _tctx = os.path.basename(str(cfg["checkpoint_dir"]).rstrip("/"))
    with timed("atom", "train", "data_load", _tctx):
        if cfg.get("train_dir") and cfg.get("val_dir"):
            train_set = InvariantDataset(data_dir=cfg["train_dir"], r_cut=cfg["r_max"],
                                         basis=cfg["basis"], mf=mf_flag)
            val_set = InvariantDataset(data_dir=cfg["val_dir"], r_cut=cfg["r_max"],
                                       basis=cfg["basis"], mf=mf_flag)
        elif cfg.get("data_dir"):
            ds = InvariantDataset(data_dir=cfg["data_dir"], r_cut=cfg["r_max"],
                                  basis=cfg["basis"], mf=mf_flag)
            ds_labeled = ds.labeled_frames()
            if all(ds_labeled):
                n_val = max(1, int(len(ds) * cfg["val_fraction"]))
                train_set, val_set = random_split(ds, [len(ds) - n_val, n_val])
            else:
                train_set, val_set = _stratified_split(
                    ds, ds_labeled, cfg["val_fraction"], cfg["seed"])
        else:
            raise ValueError("Provide --data_dir or (--train_dir + --val_dir)")
    logger.info(f"Train: {len(train_set)} | Val: {len(val_set)}")
    train_labeled, n_hf_per_batch = _fidelity_labels(train_set, cfg, mf_flag)

    stat_src = train_set.dataset if hasattr(train_set, "dataset") else train_set
    atom_e0, avg_neighbors, shift, rms, elements_present = _atom_compute_stats(
        stat_src, cfg["num_species"])
    logger.info(f"Stats: avg_neighbors={avg_neighbors:.1f}, shift={shift:.6f}, force_rms={rms:.6f}")

    # TB features are computed from geometry in-forward, so standardize with
    # statistics of the TB features themselves.
    from orbace.data.tb_features import tb_feature_stats
    node_mean, node_std, edge_mean, edge_std = tb_feature_stats(stat_src)
    logger.info("extended-Hückel features, exact conservative forces, no SCF.")

    # First of a possibly comma-separated spec: the sidecar e0s.json /
    # metadata.json are verified identical across the directories when the
    # dataset is built, so any one of them answers for all.
    data_dir = primary_data_dir(cfg.get("data_dir") or cfg.get("train_dir"))
    if data_dir and cfg.get("baseline_model", "none") == "none":
        meta = os.path.join(data_dir, "metadata.json")
        if os.path.exists(meta):
            try:
                with open(meta) as f:
                    m = json.load(f)
                if m.get("baseline_model", "none") != "none":
                    cfg["baseline_model"] = m["baseline_model"]
                    cfg["baseline_basis"] = cfg["baseline_basis"] or m.get("baseline_basis")
                    cfg["baseline_xc"] = cfg["baseline_xc"] or m.get("baseline_xc")
            except Exception:
                pass

    fidelity_method = str(cfg.get("fidelity_method", "regular"))
    base_atom_e0 = None
    if fidelity_method in ("mfdh", "mfah"):
        if cfg.get("baseline_model", "none") in (None, "none"):
            raise SystemExit(
                f"ERROR: --fidelity_method {fidelity_method} requires a cheap-fidelity "
                "baseline (the level the cheap head learns). Prepare the data with "
                "--baseline_model {xtb,gfn2,hf_sto3g,...} first.")
        base_atom_e0 = _atom_fit_base_e0(stat_src, cfg["num_species"])
        if base_atom_e0 is None:
            raise SystemExit(
                f"ERROR: --fidelity_method {fidelity_method} needs baseline_energy in every "
                "frame, but none were found in the prepared npz. Re-prepare with a baseline.")
        logger.info(
            "MFDH mode: learned base head (fidelity=%s) ADDED to a QC-conditioned delta head"
            if fidelity_method == "mfdh" else
            "MFAH mode: parallel absolute cheap head (fidelity=%s); the high-fidelity "
            "path is left identical to a direct model",
            cfg.get("baseline_model"))

    radial_mlp = [int(x) for x in str(cfg["radial_MLP"]).split(",")]
    model_kw = dict(
        r_max=cfg["r_max"], num_bessel=cfg["num_bessel"],
        num_polynomial_cutoff=cfg["num_polynomial_cutoff"], max_ell=cfg["max_ell"],
        num_interactions=cfg["num_interactions"], num_species=cfg["num_species"],
        n_node_scalars=N_NODE_SCALARS, n_edge_features=N_EDGE_FEATURES,
        hidden_irreps=cfg["hidden_irreps"], MLP_irreps=f"{cfg['mlp_channels']}x0e",
        correlation=cfg["correlation"], radial_MLP=radial_mlp,
        avg_num_neighbors=cfg["avg_num_neighbors"] or avg_neighbors,
        atomic_inter_scale=rms, atomic_inter_shift=shift, atom_e0=atom_e0,
        node_feature_mean=node_mean, node_feature_std=node_std,
        edge_feature_mean=edge_mean, edge_feature_std=edge_std,
        use_edge_features=cfg["use_edge_features"],
        pair_repulsion=cfg["pair_repulsion"], pair_repulsion_p=cfg["pair_repulsion_p"],
        use_long_range=cfg.get("use_long_range", False),
        lr_alpha=cfg.get("lr_alpha", 0.2), lr_cutoff=cfg.get("lr_cutoff", 15.0),
        baseline_model=cfg["baseline_model"], baseline_basis=cfg["baseline_basis"],
        baseline_xc=cfg["baseline_xc"],
        delta_method=fidelity_method, base_atom_e0=base_atom_e0,
        mf_head_hidden=int(cfg.get("mf_head_hidden", 64)))
    model = AtomOrbACE(**model_kw)
    if data_dir:
        e0p = os.path.join(data_dir, "e0s.json")
        if os.path.exists(e0p):
            with open(e0p) as f:
                model.e0s = {int(k): float(v) for k, v in json.load(f).items()}

    if cfg.get("foundation_model"):
        ckpt = torch.load(cfg["foundation_model"], map_location="cpu", weights_only=False)
        src = ckpt.state_dict() if isinstance(ckpt, torch.nn.Module) else ckpt
        tgt = model.state_dict()
        loaded = 0
        for k, v in src.items():
            if k in tgt and tgt[k].shape == v.shape:
                tgt[k] = v; loaded += 1
        model.load_state_dict(tgt, strict=False)
        logger.info(f"Foundation: loaded {loaded}/{len(src)} keys")
        # Inherit isolated-atom E0s from the foundation when the (finetune) data
        # dir did not provide them. Without this, model.e0s stays empty and the
        # absolute-energy (fresh xyz) eval is off by the full E0 reference.
        if not getattr(model, "e0s", None):
            found_e0s = getattr(ckpt, "e0s", None) if isinstance(ckpt, torch.nn.Module) else None
            if found_e0s:
                model.e0s = {int(k): float(v) for k, v in found_e0s.items()}
                logger.info(f"Inherited {len(model.e0s)} isolated-atom E0s from foundation model")
    if cfg.get("freeze_backbone"):
        model.freeze_backbone()

    n_used = None
    if not cfg.get("freeze_backbone"):
        n_used = _count_used(lambda ns: AtomOrbACE(**{**model_kw, "num_species": ns,
                                                      "atom_e0": None, "base_atom_e0": None}),
                             len(elements_present))
    _log_parameter_count(count_parameters(model), n_used, len(elements_present),
                         "elements", cfg["num_species"])

    _nw = int(cfg["num_workers"])
    # The AO bookkeeping is coordinate-independent, so it belongs in a worker
    # rather than the training step, where deriving it forces a device sync.
    collate = functools.partial(collate_invariant_data, tb_pack=True)
    if _nw == 0 and "--num_workers" not in sys.argv:
        _nw = 4
        logger.info("Defaulting to num_workers=4 so the AO bookkeeping "
                    "overlaps with compute (override --num_workers)")
    lk = dict(collate_fn=collate, num_workers=_nw, pin_memory=device.type == "cuda")
    if _nw > 0:
        lk["persistent_workers"] = True
        lk["prefetch_factor"] = 4

    # The TB backend solves a padded batched eigenproblem, so a batch costs
    # batch_size * nao_max^3; mixing a 105-AO molecule with 6-AO ones wastes 4-5x
    # the FLOPs. Bucketing by size recovers it.
    def _ao_counts(s):
        """AO count per frame, seeing through the Subset that random_split makes."""
        if hasattr(s, "dataset") and hasattr(s, "indices"):
            all_counts = s.dataset.tb_ao_counts()
            return [all_counts[i] for i in s.indices]
        return s.tb_ao_counts()

    bucket = bool(cfg.get("bucket_by_size", True))
    if bucket:
        try:
            train_sizes = _ao_counts(train_set)
            val_sizes = _ao_counts(val_set)
        except (AttributeError, RuntimeError, KeyError) as e:
            logger.warning(f"Bucketed batching disabled ({e}); using random batches")
            bucket = False
    pool_batches = int(cfg.get("bucket_pool_batches", 8))
    if n_hf_per_batch:
        # Label balancing owns batch membership, so it subsumes the bucketed
        # sampler; it takes the sizes and does the same pooled sort itself.
        train_loader = DataLoader(
            train_set, batch_sampler=LabelBalancedBatchSampler(
                train_labeled, cfg["batch_size"], n_hf_per_batch,
                sizes=train_sizes if bucket else None,
                pool_batches=pool_batches, seed=int(cfg.get("seed", 0))), **lk)
    elif bucket:
        train_loader = DataLoader(
            train_set, batch_sampler=BucketedBatchSampler(
                train_sizes, cfg["batch_size"], shuffle=True,
                pool_batches=pool_batches, seed=int(cfg.get("seed", 0))), **lk)
    else:
        train_loader = DataLoader(train_set, batch_size=cfg["batch_size"],
                                  shuffle=True, **lk)
    if bucket:
        # Validation takes no gradients, so order is irrelevant and a full sort
        # (shuffle=False) is both optimal and reproducible.
        val_loader = DataLoader(
            val_set, batch_sampler=BucketedBatchSampler(
                val_sizes, cfg["batch_size"], shuffle=False), **lk)
        logger.info(
            "%s (TB backend, pool_batches=%d): train nao %d..%d, %d batches/epoch",
            "Label-balanced batching, size-sorted" if n_hf_per_batch
            else "Bucketed batching ON", pool_batches,
            min(train_sizes), max(train_sizes), len(train_loader))
    else:
        val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, **lk)

    opt = (torch.optim.AdamW if cfg["optimizer"].lower() == "adamw" else torch.optim.Adam)(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], amsgrad=cfg["amsgrad"])
    if cfg["scheduler"] == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg["scheduler_factor"],
            patience=cfg["scheduler_patience"], min_lr=1e-6)
    elif cfg["scheduler"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg["n_epochs"] - cfg["warmup_epochs"]), eta_min=1e-6)
    else:
        sched = None

    if fidelity_method in ("mfdh", "mfah"):
        from orbace.tools.train import MFEnergyForcesLoss
        hf_forces = bool(cfg.get("mf_hf_forces", True))
        loss_fn = MFEnergyForcesLoss(
            energy_weight=cfg["energy_weight"], forces_weight=cfg["forces_weight"],
            base_energy_weight=float(cfg.get("base_energy_weight", 10.0)),
            base_forces_weight=float(cfg.get("base_forces_weight", 1.0)),
            forces_normalizer=rms, huber=(cfg["loss"].lower() == "huber"),
            huber_delta=cfg["huber_delta"], hf_forces=hf_forces)
        logger.info("Loss: %s (hf_forces=%s, base_E_w=%.1f, base_F_w=%.1f, huber=%s)",
                    fidelity_method.upper(), hf_forces, cfg.get("base_energy_weight", 10.0),
                    cfg.get("base_forces_weight", 1.0), cfg["loss"].lower() == "huber")
        if not hf_forces:
            logger.info("MF energy-only: high-fidelity FORCES are NOT supervised; "
                        "the high-level force is predicted as the gradient of the "
                        "total energy (cheap-fidelity forces still anchor the base head).")
    elif cfg["loss"].lower() == "huber":
        loss_fn = WeightedHuberEnergyForcesStressLoss(
            energy_weight=cfg["energy_weight"], forces_weight=cfg["forces_weight"],
            stress_weight=cfg["stress_weight"], huber_delta=cfg["huber_delta"], forces_normalizer=rms)
    else:
        loss_fn = WeightedEnergyForcesStressLoss(
            energy_weight=cfg["energy_weight"], forces_weight=cfg["forces_weight"],
            stress_weight=cfg["stress_weight"], forces_normalizer=rms)

    st_start = int(cfg["stage_two_start"]) or None
    st_lr = cfg["stage_two_lr"] if cfg["stage_two_lr"] and cfg["stage_two_lr"] > 0 else None
    trainer = Trainer(
        model=model, optimizer=opt, scheduler=sched, loss_fn=loss_fn,
        ema_decay=cfg["ema_decay"], use_swa=cfg["swa"], swa_start=cfg["start_swa"],
        max_grad_norm=cfg["clip_grad"], warmup_epochs=cfg["warmup_epochs"], device=device,
        checkpoint_dir=cfg["checkpoint_dir"], save_cpu=cfg["save_cpu"],
        stage_two_start=st_start, stage_two_energy_weight=float(cfg["stage_two_energy_weight"]),
        stage_two_forces_weight=float(cfg["stage_two_forces_weight"]), stage_two_lr=st_lr,
        use_bf16=cfg["use_bf16"])
    if cfg.get("restart"):
        epoch0 = trainer.load_checkpoint(cfg["restart"])
        logger.info(f"Restarted from epoch {epoch0}")

    logger.info("Starting training...")
    with timed("atom", "train", "fit", _tctx):
        trainer.fit(train_loader=train_loader, val_loader=val_loader,
                    n_epochs=cfg["n_epochs"], patience=cfg["patience"])
    logger.info(f"Done. Checkpoints at {cfg['checkpoint_dir']}/")


def main():
    cfg = parse_args()
    if cfg["method"] == "orbital":
        run_orbital(cfg)
    else:
        run_atom(cfg)


if __name__ == "__main__":
    main()
