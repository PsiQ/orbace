# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Resuming must continue a run, not silently restart it.

Long multi-fidelity runs outlive a 72h walltime and have to be chained, so a
resume has to preserve three things that are easy to lose. Each was a real
defect: the epoch counter (restarting it re-runs LR warmup over converged
weights), the EMA active flag (fit arms EMA past warmup by overwriting the
shadow, discarding the restored average that validation actually runs on), and
the model-selection baseline (an infinite best_val overwrites a better
best_model.pt on the first validation).
"""

import torch
import torch.nn as nn

from orbace.tools.train import Trainer


def _trainer(tmp_path, warmup=15, **kwargs):
    model = nn.Linear(4, 1)
    loss_fn = nn.MSELoss()
    loss_fn.energy_weight, loss_fn.forces_weight = 1.0, 10.0
    return Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        scheduler=None,
        loss_fn=loss_fn,
        checkpoint_dir=str(tmp_path),
        warmup_epochs=warmup,
        device=torch.device("cpu"),
        **kwargs,
    )


def _stage_two_trainer(tmp_path, **kwargs):
    return _trainer(
        tmp_path,
        warmup=15,
        stage_two_start=400,
        stage_two_energy_weight=1000.0,
        stage_two_forces_weight=100.0,
        **kwargs,
    )


def test_roundtrip_restores_epoch_and_selection_baseline(tmp_path):
    t = _trainer(tmp_path)
    t.save_checkpoint("best.pt", epoch=300, best_val=0.0123, best_forces=0.45)

    t2 = _trainer(tmp_path)
    assert t2._resume_epoch == 0, "a fresh trainer must not claim to be resuming"
    assert t2.load_checkpoint("best.pt") == 300
    assert t2._resume_epoch == 300
    assert t2._resume_best_val == 0.0123
    assert t2._resume_best_forces == 0.45


def test_resume_past_warmup_keeps_the_restored_ema(tmp_path):
    t = _trainer(tmp_path, warmup=15)
    t.ema.shadow = {"marker": torch.full((2,), 7.0)}
    t.save_checkpoint("best.pt", epoch=300, best_val=0.1)

    t2 = _trainer(tmp_path, warmup=15)
    t2.load_checkpoint("best.pt")
    # Past warmup the shadow is a real average; fit must not reset it.
    assert t2._ema_active is True
    assert torch.equal(t2.ema.shadow["marker"], torch.full((2,), 7.0))


def test_resume_inside_warmup_still_arms_ema_normally(tmp_path):
    t = _trainer(tmp_path, warmup=15)
    t.ema.shadow = {"marker": torch.zeros(2)}
    t.save_checkpoint("best.pt", epoch=5, best_val=0.1)

    t2 = _trainer(tmp_path, warmup=15)
    t2.load_checkpoint("best.pt")
    # Before warmup ends the shadow is not yet meaningful, so leaving EMA
    # inactive lets fit seed it at the usual point.
    assert t2._ema_active is False


def test_missing_best_val_falls_back_to_infinity(tmp_path):
    """Checkpoints written before best_val existed must still load."""
    t = _trainer(tmp_path)
    path = tmp_path / "old.pt"
    torch.save({
        "epoch": 42,
        "model_state_dict": t.model.state_dict(),
        "optimizer_state_dict": t.optimizer.state_dict(),
    }, path)

    t2 = _trainer(tmp_path)
    assert t2.load_checkpoint("old.pt") == 42
    assert t2._resume_best_val == float("inf")


def test_fit_starts_after_the_restored_epoch(tmp_path):
    t = _trainer(tmp_path)
    t.save_checkpoint("best.pt", epoch=300, best_val=0.1)

    t2 = _trainer(tmp_path)
    t2.load_checkpoint("best.pt")

    seen = []
    t2.train_epoch = lambda loader: (seen.append(len(seen)), {"total": 1.0})[1]
    t2.fit(train_loader=[], val_loader=None, n_epochs=303)
    # 301, 302, 303 -- not a fresh 1..303.
    assert len(seen) == 3, f"expected 3 further epochs, ran {len(seen)}"


def test_resume_past_stage_two_reapplies_its_loss_weights(tmp_path):
    """A fresh trainer starts on stage-one weights; a stage-two resume must switch."""
    t = _stage_two_trainer(tmp_path)
    t.save_checkpoint("best.pt", epoch=450, best_val=2.3)

    t2 = _stage_two_trainer(tmp_path)
    t2.load_checkpoint("best.pt")
    t2.train_epoch = lambda loader: {"total": 1.0}
    t2.fit(train_loader=[], val_loader=None, n_epochs=451)

    assert t2.loss_fn.energy_weight == 1000.0
    assert t2.loss_fn.forces_weight == 100.0


def test_resume_past_stage_two_keeps_best_val_and_stage_one_model(tmp_path):
    """Re-firing the switch on resume would discard the stage-two best."""
    t = _stage_two_trainer(tmp_path)
    t.save_checkpoint("best.pt", epoch=450, best_val=2.3)
    (tmp_path / "best_model.pt").write_bytes(b"stage-two-winner")
    (tmp_path / "best_stage_one_model.pt").write_bytes(b"stage-one-winner")

    t2 = _stage_two_trainer(tmp_path)
    t2.load_checkpoint("best.pt")
    t2.train_epoch = lambda loader: {"total": 1.0}
    # Worse than the restored best: nothing may be promoted over it.
    t2.validate = lambda loader: {"total": 9.9, "forces_mae": 9.9}
    t2.fit(train_loader=[], val_loader=[1], n_epochs=451)

    assert (tmp_path / "best_stage_one_model.pt").read_bytes() == b"stage-one-winner"
    assert (tmp_path / "best_model.pt").read_bytes() == b"stage-two-winner"
    assert torch.load(
        tmp_path / "best.pt", map_location="cpu", weights_only=False
    )["best_val"] == 2.3


def test_genuine_stage_two_crossing_still_resets_selection(tmp_path):
    """The resume guard must not disarm the real transition."""
    t = _stage_two_trainer(tmp_path)
    t.save_checkpoint("best.pt", epoch=399, best_val=0.21)
    (tmp_path / "best_model.pt").write_bytes(b"stage-one-winner")

    t2 = _stage_two_trainer(tmp_path)
    t2.load_checkpoint("best.pt")
    t2.train_epoch = lambda loader: {"total": 1.0}
    # Worse than stage one's 0.21, but the objective changed, so it is the
    # new baseline and must be saved.
    t2.validate = lambda loader: {"total": 2.5, "forces_mae": 0.1}
    t2.fit(train_loader=[], val_loader=[1], n_epochs=400)

    assert t2.loss_fn.energy_weight == 1000.0
    assert (tmp_path / "best_stage_one_model.pt").read_bytes() == b"stage-one-winner"
    assert torch.load(
        tmp_path / "best.pt", map_location="cpu", weights_only=False
    )["best_val"] == 2.5


def test_fresh_run_is_unaffected(tmp_path):
    t = _trainer(tmp_path)
    seen = []
    t.train_epoch = lambda loader: (seen.append(len(seen)), {"total": 1.0})[1]
    t.fit(train_loader=[], val_loader=None, n_epochs=3)
    assert len(seen) == 3
