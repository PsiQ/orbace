# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Training loop for OrbACE."""

import logging
import math
import os
import shutil
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.fidelity import masked_mean
from .utils import AverageMeter, EMAModel, SWAModel

logger = logging.getLogger(__name__)


# ── Loss functions ───────────────────────────────────────────────────────

class WeightedEnergyForcesLoss(nn.Module):
    """Per-atom MSE energy + per-component MSE forces."""

    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 10.0,
        forces_normalizer: float = 1.0,
    ):
        super().__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.forces_normalizer = max(float(forces_normalizer), 1.0e-8)

    def forward(self, pred: Dict, data: Dict) -> Dict[str, torch.Tensor]:
        losses = {}
        if "energy" in data and pred.get("energy") is not None:
            n_atoms = data["n_atoms"].float()
            losses["energy"] = torch.mean(
                ((pred["energy"] - data["energy"]) / n_atoms) ** 2
            )
        if "forces" in data and pred.get("forces") is not None:
            force_diff = (pred["forces"] - data["forces"]) / self.forces_normalizer
            losses["forces"] = torch.mean(force_diff ** 2)

        total = torch.tensor(0.0, device=pred["energy"].device)
        if "energy" in losses:
            total = total + self.energy_weight * losses["energy"]
        if "forces" in losses:
            total = total + self.forces_weight * losses["forces"]
        losses["total"] = total
        return losses


class WeightedEnergyForcesStressLoss(nn.Module):
    """Weighted energy-forces-stress loss."""

    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=10.0,
        stress_weight=1.0,
        forces_normalizer: float = 1.0,
    ):
        super().__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight
        self.forces_normalizer = max(float(forces_normalizer), 1.0e-8)

    def forward(self, pred: Dict, data: Dict) -> Dict[str, torch.Tensor]:
        losses = {}
        if "energy" in data and pred.get("energy") is not None:
            n_atoms = data["n_atoms"].float()
            losses["energy"] = torch.mean(
                ((pred["energy"] - data["energy"]) / n_atoms) ** 2
            )
        if "forces" in data and pred.get("forces") is not None:
            force_diff = (pred["forces"] - data["forces"]) / self.forces_normalizer
            losses["forces"] = torch.mean(force_diff ** 2)
        if "stress" in data and pred.get("stress") is not None:
            losses["stress"] = torch.mean((pred["stress"] - data["stress"]) ** 2)

        total = torch.tensor(0.0, device=pred["energy"].device)
        if "energy" in losses:
            total = total + self.energy_weight * losses["energy"]
        if "forces" in losses:
            total = total + self.forces_weight * losses["forces"]
        if "stress" in losses:
            total = total + self.stress_weight * losses["stress"]
        losses["total"] = total
        return losses


class WeightedHuberEnergyForcesStressLoss(nn.Module):
    """Huber loss variant for energy-forces-stress."""

    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=10.0,
        stress_weight=1.0,
        huber_delta=0.01,
        forces_normalizer: float = 1.0,
    ):
        super().__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight
        self.huber_delta = huber_delta
        self.forces_normalizer = max(float(forces_normalizer), 1.0e-8)

    def forward(self, pred: Dict, data: Dict) -> Dict[str, torch.Tensor]:
        losses = {}
        if "energy" in data and pred.get("energy") is not None:
            n_atoms = data["n_atoms"].float()
            losses["energy"] = torch.nn.functional.huber_loss(
                pred["energy"] / n_atoms, data["energy"] / n_atoms,
                reduction="mean", delta=self.huber_delta,
            )
        if "forces" in data and pred.get("forces") is not None:
            force_delta = self.huber_delta / self.forces_normalizer
            losses["forces"] = torch.nn.functional.huber_loss(
                pred["forces"] / self.forces_normalizer,
                data["forces"] / self.forces_normalizer,
                reduction="mean", delta=force_delta,
            )
        if "stress" in data and pred.get("stress") is not None:
            losses["stress"] = torch.nn.functional.huber_loss(
                pred["stress"], data["stress"],
                reduction="mean", delta=self.huber_delta,
            )

        total = torch.tensor(0.0, device=pred["energy"].device)
        if "energy" in losses:
            total = total + self.energy_weight * losses["energy"]
        if "forces" in losses:
            total = total + self.forces_weight * losses["forces"]
        if "stress" in losses:
            total = total + self.stress_weight * losses["stress"]
        losses["total"] = total
        return losses


class MFEnergyForcesLoss(nn.Module):
    """Multi-fidelity loss, shared by ``fidelity_method`` 'mfdh' and 'mfah'.

    Both modes are trained with the same four terms -- the high-fidelity output
    against the ABSOLUTE reference, and the cheap-level head against the cheap
    fidelity:

    * ``pred['energy']`` / ``pred['forces']`` vs the absolute reference
      ``data['energy']`` / ``data['forces']`` -- these are the primary terms,
      over well-behaved absolute forces (never the inflated residual), and
    * the cheap-level head ``pred['base_energy']`` / ``pred['base_forces']``
      vs the cheap-baseline targets ``data['baseline_energy']`` /
      ``data['baseline_forces']``.

    Only the MODEL differs between the two arms, not this loss. Under mfdh the
    cheap head is an addend of ``pred['energy']``, so the two energy terms fight
    over one shared quantity and the cheap head must be evaluated at inference.
    Under mfah the two heads are parallel and independent, so the cheap terms
    act purely as an auxiliary task that shapes the shared backbone, and the
    cheap head can be dropped after training.

    ``energy_weight`` / ``forces_weight`` weight the high-fidelity terms (so the
    shared Trainer's ``need_forces`` / stage-two logic keep working);
    ``base_*_weight`` weight the cheap-level terms. The cheap force term is kept
    light by default: the baseline forces are a poor target, so pulling the
    shared features hard toward them degrades the high-fidelity force accuracy.
    """

    def __init__(
        self,
        energy_weight: float = 10.0,
        forces_weight: float = 100.0,
        base_energy_weight: float = 10.0,
        base_forces_weight: float = 1.0,
        forces_normalizer: float = 1.0,
        huber: bool = True,
        huber_delta: float = 0.01,
        hf_forces: bool = True,
    ):
        super().__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.base_energy_weight = base_energy_weight
        self.base_forces_weight = base_forces_weight
        self.forces_normalizer = max(float(forces_normalizer), 1.0e-8)
        self.huber = huber
        self.huber_delta = huber_delta
        # hf_forces=False => pure multi-fidelity: the high-fidelity level is
        # supervised on ENERGY only, and the high-level force is left to emerge as
        # the (autograd) gradient of the total energy -- never trained on. The
        # cheap-fidelity force anchor is still applied, so forces are still
        # computed during training (see the ``need_forces`` hook below).
        self.hf_forces = bool(hf_forces)

    @property
    def need_forces(self) -> bool:
        """Trainer hook: compute forces during training when EITHER the
        high-fidelity force term or the cheap base-force anchor is active."""
        return (self.hf_forces and self.forces_weight > 0) or self.base_forces_weight > 0

    def _e_term(self, pred_e, tgt_e, n_atoms, mask=None):
        if self.huber:
            per = torch.nn.functional.huber_loss(
                pred_e / n_atoms, tgt_e / n_atoms, reduction="none", delta=self.huber_delta)
        else:
            per = ((pred_e - tgt_e) / n_atoms) ** 2
        return masked_mean(per, mask)

    def _f_term(self, pred_f, tgt_f, mask=None):
        fn = self.forces_normalizer
        if self.huber:
            per = torch.nn.functional.huber_loss(
                pred_f / fn, tgt_f / fn, reduction="none", delta=self.huber_delta / fn)
        else:
            per = ((pred_f - tgt_f) / fn) ** 2
        return masked_mean(per, mask)

    def forward(self, pred: Dict, data: Dict) -> Dict[str, torch.Tensor]:
        losses = {}
        n_atoms = data["n_atoms"].float()
        # Present only when the batch mixes labeled and unlabeled frames; the
        # zero-filled entries are the ones these keep out of the mean. Absent
        # means every frame carried the label, and the terms reduce as before.
        e_mask = data.get("energy_mask")
        f_mask = data.get("forces_mask")

        if "energy" in data and pred.get("energy") is not None:
            losses["energy"] = self._e_term(
                pred["energy"], data["energy"], n_atoms, e_mask)
        # High-fidelity force term only when hf_forces=True. In energy-only mode
        # the total force is still predicted (for val F_MAE reporting) but not
        # supervised against the high-level forces.
        if self.hf_forces and "forces" in data and pred.get("forces") is not None:
            losses["forces"] = self._f_term(pred["forces"], data["forces"], f_mask)
        if "baseline_energy" in data and pred.get("base_energy") is not None:
            losses["base_energy"] = self._e_term(
                pred["base_energy"], data["baseline_energy"], n_atoms,
                data.get("baseline_energy_mask"))
        if "baseline_forces" in data and pred.get("base_forces") is not None:
            losses["base_forces"] = self._f_term(
                pred["base_forces"], data["baseline_forces"],
                data.get("baseline_forces_mask"))

        total = torch.tensor(0.0, device=pred["energy"].device)
        if "energy" in losses:
            total = total + self.energy_weight * losses["energy"]
        if "forces" in losses:
            total = total + self.forces_weight * losses["forces"]
        if "base_energy" in losses:
            total = total + self.base_energy_weight * losses["base_energy"]
        if "base_forces" in losses:
            total = total + self.base_forces_weight * losses["base_forces"]
        losses["total"] = total
        return losses


class EnergyForceLoss(WeightedEnergyForcesLoss):
    """Same as ``WeightedEnergyForcesLoss``; ``force_weight`` aliases ``forces_weight``.

    Kept for ``train.py`` / legacy CLI. Stage-two Trainer updates
    ``energy_weight`` and ``forces_weight`` on this class as expected.
    """

    def __init__(
        self,
        energy_weight: float = 1.0,
        force_weight: float = 10.0,
        stress_weight: float = 0.0,
    ):
        if stress_weight and stress_weight > 0:
            raise ValueError(
                "EnergyForceLoss does not support stress; use WeightedEnergyForcesStressLoss"
            )
        super().__init__(
            energy_weight=energy_weight,
            forces_weight=force_weight,
            forces_normalizer=1.0,
        )


# ── Trainer ─────────────────────────────────────────────────────────────

class Trainer:
    """Training loop with warmup, SWA, EMA, gradient clipping.

    Stage Two: after ``stage_two_start`` epochs, optionally switch
    loss weights to ``stage_two_energy_weight`` / ``stage_two_forces_weight`` /
    ``stage_two_stress_weight`` and optionally reset the optimizer LR to
    ``stage_two_lr`` (typically used together with SWA / SWALR).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        loss_fn=None,
        ema_decay: float = 0.999,
        use_swa: bool = False,
        swa_start: int = 0,
        max_grad_norm: float = 10.0,
        warmup_epochs: int = 5,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = False,
        compute_stress: bool = False,
        save_cpu: bool = True,
        stage_two_start: Optional[int] = None,
        stage_two_energy_weight: float = 1000.0,
        stage_two_forces_weight: float = 100.0,
        stage_two_stress_weight: Optional[float] = None,
        stage_two_lr: Optional[float] = None,
        use_bf16: bool = False,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or WeightedEnergyForcesLoss()
        self.ema = EMAModel(model, decay=ema_decay, warmup_steps=0)
        self._ema_active = False
        # Overwritten by load_checkpoint; these defaults make fit() behave
        # exactly as before for a run that is not resuming.
        self._resume_epoch = 0
        self._resume_best_val = float("inf")
        self._resume_best_forces = float("inf")
        self.swa = SWAModel(model) if use_swa else None
        self.swa_start = swa_start
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = warmup_epochs
        self._nonfinite_skips = 0
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.compute_stress = compute_stress
        self.save_cpu = save_cpu
        self.stage_two_start = stage_two_start
        self.stage_two_energy_weight = stage_two_energy_weight
        self.stage_two_forces_weight = stage_two_forces_weight
        self.stage_two_stress_weight = stage_two_stress_weight
        self.stage_two_lr = stage_two_lr
        self._stage_two_applied = False
        # bf16 autocast accelerates e3nn/interaction blocks ~1.5-2x with
        # negligible accuracy loss (MACE uses this in production). Disabled
        # on CPU (no bf16 gain without AVX512-BF16).
        self.use_bf16 = bool(use_bf16) and device.type == "cuda"
        os.makedirs(checkpoint_dir, exist_ok=True)

        self._base_lr = optimizer.param_groups[0]["lr"]
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                self.use_wandb = False

    def _to_device(self, data):
        # non_blocking transfers overlap with compute when the source tensors
        # are pinned (DataLoader pin_memory=True) and the device is CUDA.
        non_blocking = self.device.type == "cuda"
        out = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=non_blocking)
            else:
                out[k] = v
        return out

    def _maybe_apply_stage_two(self, epoch: int) -> bool:
        """Stage Two: new energy/forces (and optional stress) loss weights.

        Returns True on the single epoch where the switch happens, so ``fit``
        can reset its best-loss / patience state. The stage-two objective is a
        *different* objective: its total loss is not comparable to stage one's,
        so carrying ``best_val`` across the boundary would freeze model
        selection at the last stage-one epoch and make early stopping fire
        ``patience`` epochs later regardless of progress.
        """
        if self.stage_two_start is None or epoch < self.stage_two_start:
            return False
        if self._stage_two_applied:
            return False
        self._stage_two_applied = True
        self._apply_stage_two_settings(f"Stage Two (epoch {epoch})")
        return True

    def _restore_stage_two_on_resume(self, start_epoch: int):
        """Re-apply stage-two settings for a run resuming past the boundary.

        A fresh ``Trainer`` is built with stage-one weights and a live
        scheduler, so a resume from a stage-two checkpoint has to switch them
        back or it would silently keep training on the stage-one objective.
        It must *not* go through ``_maybe_apply_stage_two``: that reports a
        transition, which makes ``fit`` reset ``best_val`` and overwrite
        ``best_stage_one_model.pt`` with a stage-two model.
        """
        if self.stage_two_start is None or self._stage_two_applied:
            return
        if start_epoch < self.stage_two_start:
            return
        self._stage_two_applied = True
        self._apply_stage_two_settings(
            f"Stage Two (restored on resume at epoch {start_epoch})"
        )

    def _apply_stage_two_settings(self, prefix: str):
        e0 = getattr(self.loss_fn, "energy_weight", None)
        f0 = getattr(self.loss_fn, "forces_weight", None)
        s0 = getattr(self.loss_fn, "stress_weight", None)

        self.loss_fn.energy_weight = self.stage_two_energy_weight
        self.loss_fn.forces_weight = self.stage_two_forces_weight
        if self.stage_two_stress_weight is not None and hasattr(
            self.loss_fn, "stress_weight"
        ):
            self.loss_fn.stress_weight = self.stage_two_stress_weight

        msg = (
            f"{prefix}: loss weights "
            f"energy {e0} -> {self.stage_two_energy_weight}, "
            f"forces {f0} -> {self.stage_two_forces_weight}"
        )
        if s0 is not None and self.stage_two_stress_weight is not None:
            msg += f", stress {s0} -> {self.stage_two_stress_weight}"
        logger.info(msg)

        if self.stage_two_lr is not None:
            self._base_lr = self.stage_two_lr
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.stage_two_lr
            if self.scheduler is not None:
                logger.info(
                    "Stage Two: disabling LR scheduler (fixed LR from stage_two_lr); "
                    "set stage_two_lr=None to keep cosine/plateau scheduling."
                )
                self.scheduler = None

    def _warmup_lr(self, epoch: int):
        if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
            factor = epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = self._base_lr * factor

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        meters = {k: AverageMeter() for k in ("total", "energy", "forces", "stress")}
        skips_before = self._nonfinite_skips
        need_forces = getattr(self.loss_fn, "need_forces",
                               getattr(self.loss_fn, "forces_weight", 0) > 0)
        need_stress = self.compute_stress and getattr(self.loss_fn, "stress_weight", 0) > 0

        for batch in train_loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()

            if self.use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = self.model(
                        batch, training=True,
                        compute_force=need_forces and "forces" in batch,
                        compute_stress=need_stress and "stress" in batch,
                    )
                    losses = self.loss_fn(pred, batch)
            else:
                pred = self.model(
                    batch, training=True,
                    compute_force=need_forces and "forces" in batch,
                    compute_stress=need_stress and "stress" in batch,
                )
                losses = self.loss_fn(pred, batch)

            # Skip non-finite batches so a single exploding gradient cannot
            # poison every parameter (and thus every subsequent epoch) via
            # optimizer.step(). Common with response-aware/second-derivative
            # forces during LR warmup.
            if not torch.isfinite(losses["total"]):
                self._nonfinite_skips += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            losses["total"].backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            if not all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in self.model.parameters()):
                self._nonfinite_skips += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.optimizer.step()
            if self._ema_active:
                self.ema.update(self.model)

            n = batch["num_graphs"]
            for k in meters:
                if k in losses:
                    meters[k].update(losses[k].item(), n)

        skipped = self._nonfinite_skips - skips_before
        if skipped:
            logging.warning(
                "Skipped %d non-finite batch(es) this epoch (LR may be too high "
                "for the response-aware forces).", skipped)
        return {k: m.avg for k, m in meters.items()}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate WITHOUT torch.no_grad() -- forces need autograd."""
        self.model.eval()
        meters = {k: AverageMeter() for k in
                  ("total", "energy", "forces", "stress",
                   "energy_mae", "energy_mae_peratom", "forces_mae",
                   "energy_mse", "energy_mse_peratom", "forces_mse")}
        need_forces_for_loss = getattr(self.loss_fn, "forces_weight", 0) > 0
        need_stress = self.compute_stress and getattr(self.loss_fn, "stress_weight", 0) > 0

        for batch in val_loader:
            batch = self._to_device(batch)
            has_forces = "forces" in batch
            pred = self.model(
                batch, training=False,
                # A multi-fidelity batch can hold no high-fidelity frame at all,
                # and its cheap-level force anchor still has to be evaluated.
                compute_force=has_forces or getattr(self.loss_fn, "need_forces", False),
                compute_stress=need_stress and "stress" in batch,
            )
            losses = self.loss_fn(pred, batch)

            n = batch["num_graphs"]
            for k in ("total", "energy", "forces", "stress"):
                if k in losses:
                    meters[k].update(losses[k].detach().item(), n)

            # Present only when the batch mixes labeled and unlabeled frames.
            # The metrics then average over the labeled ones and are weighted by
            # how many there were, so a sparse batch cannot pull the epoch
            # average toward the zeros that stand in for the missing labels.
            e_mask = batch.get("energy_mask")
            f_mask = batch.get("forces_mask")
            n_e = n if e_mask is None else int(e_mask.sum())
            if "energy" in losses and n_e:
                e_diff = torch.abs(pred["energy"].detach() - batch["energy"])
                n_atoms = batch["n_atoms"].float()
                meters["energy_mae"].update(masked_mean(e_diff, e_mask).item(), n_e)
                meters["energy_mae_peratom"].update(
                    masked_mean(e_diff / n_atoms, e_mask).item(), n_e
                )
                meters["energy_mse"].update(masked_mean(e_diff ** 2, e_mask).item(), n_e)
                meters["energy_mse_peratom"].update(
                    masked_mean((e_diff / n_atoms) ** 2, e_mask).item(), n_e
                )
            n_f = n if f_mask is None else int(f_mask.sum())
            if has_forces and pred.get("forces") is not None and n_f:
                f_diff = pred["forces"].detach() - batch["forces"]
                meters["forces_mae"].update(
                    masked_mean(torch.abs(f_diff), f_mask).item(), n_f)
                meters["forces_mse"].update(masked_mean(f_diff ** 2, f_mask).item(), n_f)

        out = {k: m.avg for k, m in meters.items()}
        # RMSE is the sqrt of the averaged squared error, so it has to be
        # derived here rather than averaged like the MAE meters.
        for mse_key, rmse_key in (("energy_mse", "energy_rmse"),
                                  ("energy_mse_peratom", "energy_rmse_peratom"),
                                  ("forces_mse", "forces_rmse")):
            out[rmse_key] = max(out[mse_key], 0.0) ** 0.5
        return out

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 100,
        log_interval: int = 1,
        patience: int = 50,
        start_epoch: Optional[int] = None,
    ) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": [], "lr": []}
        # Picked up from load_checkpoint so a resumed run continues its schedule
        # instead of restarting it. Counting from 1 again would re-run the LR
        # warmup over already-converged weights and re-date stage two.
        if start_epoch is None:
            start_epoch = self._resume_epoch
        # Seeding these from the checkpoint is what stops the first resumed
        # epoch from overwriting a better best_model.pt purely because its own
        # baseline is infinite.
        best_val = self._resume_best_val
        best_forces = self._resume_best_forces
        wait = 0

        if start_epoch:
            logger.info("Resuming at epoch %d (best val so far %.6f)",
                        start_epoch + 1, best_val)
            self._restore_stage_two_on_resume(start_epoch)

        for epoch in range(start_epoch + 1, n_epochs + 1):
            self._warmup_lr(epoch)
            if not self._ema_active and epoch > self.warmup_epochs:
                self._ema_active = True
                self.ema.shadow = {
                    name: p.data.clone()
                    for name, p in self.model.named_parameters()
                }
                logger.info(
                    "EMA activated (epoch %d): shadow reset to current params", epoch
                )
            if self._maybe_apply_stage_two(epoch):
                # Preserve the stage-one winner, then restart selection under
                # the new objective.
                if os.path.exists(os.path.join(self.checkpoint_dir, "best_model.pt")):
                    shutil.copyfile(
                        os.path.join(self.checkpoint_dir, "best_model.pt"),
                        os.path.join(self.checkpoint_dir, "best_stage_one_model.pt"),
                    )
                logger.info(
                    "Stage Two: kept stage-one best (val=%.6f) as "
                    "best_stage_one_model.pt; resetting best-loss and patience",
                    best_val,
                )
                best_val = float("inf")
                wait = 0
                # ReduceLROnPlateau tracks its own best of the same total loss,
                # so it too would read the weight change as a regression and
                # start cutting the LR immediately.
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if hasattr(self.scheduler, "_reset"):
                        self.scheduler._reset()
                    else:  # pragma: no cover - older torch
                        self.scheduler.best = self.scheduler.mode_worse
                        self.scheduler.num_bad_epochs = 0
                        self.scheduler.cooldown_counter = 0
                    logger.info("Stage Two: reset ReduceLROnPlateau baseline")
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            dt = time.time() - t0

            history["train_loss"].append(train_metrics["total"])
            lr = self.optimizer.param_groups[0]["lr"]
            history["lr"].append(lr)

            log_data = {"epoch": epoch, "train/loss": train_metrics["total"], "lr": lr}

            if val_loader is not None:
                # Validate (and select/save the best model) using the EMA
                # weights once EMA is active. The EMA average is smoother and
                # generalizes better than the raw step weights; saving the raw
                # weights as best_model.pt (the previous behavior) means eval/MD
                # never actually benefit from EMA. Swap the shadow in, validate,
                # do all "best" saves, then restore the raw weights for the next
                # training epoch.
                ema_backup = None
                if self._ema_active:
                    ema_backup = self.ema.get_backup(self.model)
                    self.ema.apply(self.model)

                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["total"])
                log_data.update({
                    "val/loss": val_metrics["total"],
                    "val/E_MAE_peratom": val_metrics.get("energy_mae_peratom", 0),
                    "val/F_MAE": val_metrics.get("forces_mae", 0),
                    "val/E_RMSE_peratom": val_metrics.get("energy_rmse_peratom", 0),
                    "val/F_RMSE": val_metrics.get("forces_rmse", 0),
                })

                if val_metrics["total"] < best_val:
                    best_val = val_metrics["total"]
                    wait = 0
                    # best_model.pt now holds the (EMA) weights that were
                    # actually validated, so eval/MD load the evaluated model.
                    self.save_model("best_model.pt")
                    self.save_checkpoint("best.pt", epoch, best_val, best_forces)
                else:
                    wait += 1

                f_mae = val_metrics.get("forces_mae", float("inf"))
                if f_mae < best_forces:
                    best_forces = f_mae
                    self.save_model("best_forces_model.pt")

                # Restore raw step weights for the next training epoch.
                if ema_backup is not None:
                    self.ema.restore(self.model, ema_backup)

                if epoch == 1 or epoch % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch:4d} | "
                        f"train={train_metrics['total']:.6f} | "
                        f"val={val_metrics['total']:.6f} | "
                        f"E_MAE={val_metrics.get('energy_mae', 0):.4f} eV | "
                        f"E_MAE/atom={val_metrics.get('energy_mae_peratom', 0):.4f} | "
                        f"F_MAE={val_metrics.get('forces_mae', 0):.4f} eV/A | "
                        f"E_RMSE={val_metrics.get('energy_rmse', 0):.4f} eV | "
                        f"F_RMSE={val_metrics.get('forces_rmse', 0):.4f} eV/A | "
                        f"lr={lr:.2e} | {dt:.1f}s"
                    )

                if wait >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                if epoch == 1 or epoch % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch:4d} | train={train_metrics['total']:.6f} | "
                        f"lr={lr:.2e} | {dt:.1f}s"
                    )

            if self._wandb is not None:
                self._wandb.log(log_data)

            # SWA
            if self.swa is not None and epoch >= self.swa_start:
                self.swa.update(self.model)

            # Scheduler step
            if epoch > self.warmup_epochs and self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_metrics["total"] if val_loader else train_metrics["total"]
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

        self.save_checkpoint("last.pt", epoch, best_val, best_forces)
        self.save_model("last_model.pt")

        # Save EMA model
        backup = self.ema.get_backup(self.model)
        self.ema.apply(self.model)
        self.save_model("ema_model.pt")
        self.ema.restore(self.model, backup)

        # Save SWA model
        if self.swa is not None and self.swa.n_averaged > 0:
            swa_backup = self.ema.get_backup(self.model)
            self.swa.apply(self.model)
            self.save_model("swa_model.pt")
            self.ema.restore(self.model, swa_backup)

        return history

    def save_checkpoint(self, filename: str, epoch: int,
                        best_val: float = float("inf"),
                        best_forces: float = float("inf")):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_shadow": self.ema.shadow,
            # Without these a resumed run restarts model selection from an
            # infinite baseline and overwrites a better checkpoint with a
            # worse one on its very first validation.
            "best_val": best_val,
            "best_forces": best_forces,
        }, path)

    def save_model(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        if self.save_cpu:
            model_copy = self.model.to("cpu")
            torch.save(model_copy, path)
            self.model.to(self.device)
        else:
            torch.save(self.model, path)

    def load_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._resume_epoch = int(ckpt.get("epoch", 0))
        self._resume_best_val = float(ckpt.get("best_val", float("inf")))
        self._resume_best_forces = float(ckpt.get("best_forces", float("inf")))
        if "ema_shadow" in ckpt:
            self.ema.shadow = ckpt["ema_shadow"]
            # fit() arms EMA the first epoch past warmup by resetting the
            # shadow to the current weights. On a resume that is already past
            # warmup it would throw away the restored average -- which is what
            # validation and every "best" save actually run on.
            if self._resume_epoch > self.warmup_epochs:
                self._ema_active = True
        return self._resume_epoch
