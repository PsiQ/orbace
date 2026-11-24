# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Training on datasets where only some frames carry a high-fidelity label.

A multi-fidelity dataset need not label every frame at the expensive level:
the cheap method runs over everything and the expensive one over whatever the
budget allowed. These check the three things that has to get right -- that
collation keeps the batch aligned and says which frames were labeled, that the
loss reduces over exactly those frames, and that batch composition can be
fixed when labels are too sparse for shuffling to deliver any.

Run either way:  python tests/test_partial_labels.py  |  pytest tests/test_partial_labels.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbace.data.fidelity import (  # noqa: E402
    LabelBalancedBatchSampler,
    auto_hf_per_batch,
    check_prep_compatible,
    list_npz,
    masked_mean,
    split_data_dirs,
)
from orbace.data.invariant_graph import (  # noqa: E402
    InvariantData,
    collate_invariant_data,
)
from orbace.tools.train import MFEnergyForcesLoss  # noqa: E402


@pytest.fixture(autouse=True)
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _frame(n_atoms=3, energy=-10.0, labeled=True, seed=0):
    """One graph, optionally without its high-fidelity labels.

    Every frame carries the cheap baseline, which is what makes the unlabeled
    ones worth training on at all.
    """
    g = torch.Generator().manual_seed(seed)
    dt = torch.get_default_dtype()
    src = torch.arange(n_atoms).repeat_interleave(n_atoms - 1)
    dst = torch.tensor([j for i in range(n_atoms)
                        for j in range(n_atoms) if j != i])
    return InvariantData(
        atom_positions=torch.rand(n_atoms, 3, generator=g, dtype=dt),
        atom_numbers=torch.full((n_atoms,), 8, dtype=torch.int64),
        node_scalars=torch.rand(n_atoms, 4, generator=g, dtype=dt),
        edge_index=torch.stack([src, dst]),
        edge_features=torch.rand(src.numel(), 6, generator=g, dtype=dt),
        energy=torch.tensor(energy, dtype=dt) if labeled else None,
        forces=torch.rand(n_atoms, 3, generator=g, dtype=dt) if labeled else None,
        baseline_energy=torch.tensor(energy - 1.0, dtype=dt),
        baseline_forces=torch.rand(n_atoms, 3, generator=g, dtype=dt),
    )


# ── collation ───────────────────────────────────────────────────────────

def test_fully_labeled_batch_carries_no_masks():
    """The pre-existing case has to stay exactly as it was."""
    batch = collate_invariant_data([_frame(seed=i) for i in range(4)])
    assert "energy_mask" not in batch
    assert "forces_mask" not in batch
    assert batch["energy"].shape == (4,)


def test_unlabeled_batch_omits_the_key_entirely():
    """No labels at all is not a masked batch, it is a batch with no target."""
    batch = collate_invariant_data(
        [_frame(labeled=False, seed=i) for i in range(4)])
    assert "energy" not in batch
    assert "forces" not in batch
    assert "energy_mask" not in batch
    # The cheap level is still there; that is the whole point of the frames.
    assert batch["baseline_energy"].shape == (4,)


def test_mixed_batch_stays_aligned_with_the_graphs():
    labeled = [True, False, True, False, False]
    frames = [_frame(n_atoms=3 + i, labeled=f, seed=i)
              for i, f in enumerate(labeled)]
    batch = collate_invariant_data(frames)

    # One energy per graph, not one per LABELED graph -- the misalignment that
    # dropping the absent ones used to cause.
    assert batch["energy"].shape == (len(frames),)
    assert batch["forces"].shape == (batch["total_n_atoms"], 3)
    assert batch["energy_mask"].tolist() == labeled

    # The per-atom mask has to follow each frame's atom count.
    expected = [f for f, fr in zip(labeled, frames)
                for _ in range(fr.atom_positions.shape[0])]
    assert batch["forces_mask"].tolist() == expected

    # Absent labels read as zero, so anything that ignores the mask sees a
    # number rather than garbage.
    assert torch.equal(batch["energy"][~batch["energy_mask"]],
                       torch.zeros(3, dtype=batch["energy"].dtype))


def test_baseline_is_never_masked_when_every_frame_has_one():
    batch = collate_invariant_data(
        [_frame(labeled=(i == 0), seed=i) for i in range(4)])
    assert "baseline_energy_mask" not in batch
    assert "baseline_forces_mask" not in batch


# ── masked reduction ────────────────────────────────────────────────────

def test_masked_mean_without_a_mask_is_a_plain_mean():
    x = torch.randn(7, 3)
    assert torch.allclose(masked_mean(x, None), x.mean())


def test_masked_mean_ignores_the_unselected():
    x = torch.tensor([1.0, 99.0, 3.0, 99.0])
    mask = torch.tensor([True, False, True, False])
    assert torch.allclose(masked_mean(x, mask), torch.tensor(2.0))


def test_masked_mean_broadcasts_a_row_mask_over_components():
    x = torch.tensor([[1.0, 1.0, 1.0], [99.0, 99.0, 99.0], [3.0, 3.0, 3.0]])
    mask = torch.tensor([True, False, True])
    assert torch.allclose(masked_mean(x, mask), torch.tensor(2.0))


def test_masked_mean_of_nothing_is_zero_but_differentiable():
    x = torch.randn(4, requires_grad=True)
    out = masked_mean(x, torch.zeros(4, dtype=torch.bool))
    assert out.detach().item() == 0.0
    out.backward()
    assert torch.allclose(x.grad, torch.zeros(4))


# ── loss ────────────────────────────────────────────────────────────────

def _pred_for(batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    dt = torch.get_default_dtype()
    n_g, n_a = batch["num_graphs"], batch["total_n_atoms"]
    return {
        "energy": torch.rand(n_g, generator=g, dtype=dt, requires_grad=True),
        "forces": torch.rand(n_a, 3, generator=g, dtype=dt, requires_grad=True),
        "base_energy": torch.rand(n_g, generator=g, dtype=dt, requires_grad=True),
        "base_forces": torch.rand(n_a, 3, generator=g, dtype=dt, requires_grad=True),
    }


def test_loss_on_a_mixed_batch_equals_the_loss_on_its_labeled_frames():
    """The property everything else rests on.

    The high-fidelity terms have to see the same number they would have seen
    had the unlabeled frames never been in the batch.
    """
    loss_fn = MFEnergyForcesLoss(huber=False)
    labeled = [True, False, True, False]
    frames = [_frame(n_atoms=3, labeled=f, seed=i) for i, f in enumerate(labeled)]

    mixed = collate_invariant_data(frames)
    pred = _pred_for(mixed)
    mixed_losses = loss_fn(pred, mixed)

    # The same frames and the same predictions, with the unlabeled ones removed.
    keep = [i for i, f in enumerate(labeled) if f]
    dense = collate_invariant_data([frames[i] for i in keep])
    atom_keep = torch.tensor(
        [i for i, gi in enumerate(mixed["batch_atom"].tolist()) if labeled[gi]])
    dense_pred = {
        "energy": pred["energy"][keep],
        "forces": pred["forces"][atom_keep],
        "base_energy": pred["base_energy"][keep],
        "base_forces": pred["base_forces"][atom_keep],
    }
    dense_losses = loss_fn(dense_pred, dense)

    assert torch.allclose(mixed_losses["energy"], dense_losses["energy"])
    assert torch.allclose(mixed_losses["forces"], dense_losses["forces"])


def test_unlabeled_frames_still_train_the_cheap_head():
    """A batch with no expensive frame is not a wasted step."""
    loss_fn = MFEnergyForcesLoss(huber=False)
    batch = collate_invariant_data(
        [_frame(labeled=False, seed=i) for i in range(4)])
    pred = _pred_for(batch)
    losses = loss_fn(pred, batch)

    assert "energy" not in losses          # nothing to compare against
    assert "base_energy" in losses         # but the cheap level is supervised
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert pred["base_energy"].grad.abs().sum() > 0
    # The high-fidelity head is untouched rather than pushed toward zero.
    assert pred["energy"].grad is None


def test_a_mixed_batch_backpropagates_through_both_heads():
    loss_fn = MFEnergyForcesLoss(huber=False)
    batch = collate_invariant_data(
        [_frame(labeled=(i % 3 == 0), seed=i) for i in range(6)])
    pred = _pred_for(batch)
    loss_fn(pred, batch)["total"].backward()

    # Only the labeled graphs pull on the high-fidelity head.
    e_grad = pred["energy"].grad
    assert (e_grad[batch["energy_mask"]].abs() > 0).all()
    assert torch.allclose(e_grad[~batch["energy_mask"]],
                          torch.zeros_like(e_grad[~batch["energy_mask"]]))
    assert (pred["base_energy"].grad.abs() > 0).all()


def test_huber_and_mse_agree_about_which_frames_count():
    for huber in (True, False):
        loss_fn = MFEnergyForcesLoss(huber=huber, huber_delta=1.0)
        frames = [_frame(labeled=(i < 2), seed=i) for i in range(4)]
        mixed = collate_invariant_data(frames)
        pred = _pred_for(mixed)
        dense = collate_invariant_data(frames[:2])
        dense_pred = {k: v[:2] if k in ("energy", "base_energy") else v[:6]
                      for k, v in pred.items()}
        assert torch.allclose(loss_fn(pred, mixed)["energy"],
                              loss_fn(dense_pred, dense)["energy"]), huber


# ── the auto rule ───────────────────────────────────────────────────────

def test_dense_labels_need_no_balancing():
    """50k cheap / 10k expensive already puts ~13 labeled frames in a batch."""
    assert auto_hf_per_batch(n_labeled=10_000, n_total=50_000, batch_size=64) == 0


def test_sparse_labels_turn_balancing_on():
    """1M cheap / 10k expensive averages 0.64 per batch, and half are empty."""
    assert auto_hf_per_batch(n_labeled=10_000, n_total=1_000_000, batch_size=64) == 8


def test_balancing_never_takes_over_the_batch():
    assert auto_hf_per_batch(n_labeled=50, n_total=100_000, batch_size=8) <= 4


def test_a_fully_labeled_set_is_left_alone():
    assert auto_hf_per_batch(n_labeled=500, n_total=500, batch_size=64) == 0
    assert auto_hf_per_batch(n_labeled=0, n_total=500, batch_size=64) == 0


# ── the sampler ─────────────────────────────────────────────────────────

def _labels(n_total, n_labeled, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_labeled, replace=False)
    flags = [False] * n_total
    for i in idx:
        flags[i] = True
    return flags


def test_every_batch_holds_the_requested_number_of_labels():
    flags = _labels(1000, 20)
    sampler = LabelBalancedBatchSampler(flags, batch_size=16,
                                        n_labeled_per_batch=4, seed=1)
    batches = list(sampler)
    assert batches
    for b in batches:
        assert len(b) == 16
        assert sum(flags[i] for i in b) == 4
        assert len(set(b)) == 16          # no frame twice in one batch


def test_an_epoch_is_one_pass_over_the_cheap_frames():
    flags = _labels(1000, 20)
    sampler = LabelBalancedBatchSampler(flags, batch_size=16,
                                        n_labeled_per_batch=4, seed=1)
    seen = [i for b in sampler for i in b if not flags[i]]
    assert len(seen) == len(set(seen))            # each cheap frame once
    assert len(seen) == 980 // 12 * 12            # up to the last partial batch


def test_the_small_labeled_set_is_reused_rather_than_exhausted():
    flags = _labels(1000, 20)
    sampler = LabelBalancedBatchSampler(flags, batch_size=16,
                                        n_labeled_per_batch=4, seed=1)
    drawn = [i for b in sampler for i in b if flags[i]]
    assert len(drawn) > 20                         # wrapped around
    assert set(drawn) == {i for i, f in enumerate(flags) if f}


def test_batches_differ_between_epochs():
    flags = _labels(400, 40)
    sampler = LabelBalancedBatchSampler(flags, batch_size=10,
                                        n_labeled_per_batch=2, seed=3)
    assert list(sampler)[0] != list(sampler)[0]


def test_size_sorting_groups_similar_molecules():
    """Bucketing has to survive the label constraint, or the TB backend pays."""
    flags = _labels(600, 60, seed=2)
    sizes = list(np.random.default_rng(7).integers(6, 106, size=600))
    kw = dict(batch_size=20, n_labeled_per_batch=4, seed=5)
    spread = {}
    for name, s in (("sorted", sizes), ("random", None)):
        batches = list(LabelBalancedBatchSampler(flags, sizes=s, **kw))
        spread[name] = np.mean([max(sizes[i] for i in b) - min(sizes[i] for i in b)
                                for b in batches])
        for b in batches:
            assert sum(flags[i] for i in b) == 4
    assert spread["sorted"] < 0.5 * spread["random"]


def test_the_sampler_refuses_a_configuration_it_cannot_honor():
    flags = _labels(100, 10)
    with pytest.raises(ValueError):
        LabelBalancedBatchSampler(flags, batch_size=8, n_labeled_per_batch=8)
    with pytest.raises(ValueError):
        LabelBalancedBatchSampler([True] * 10, batch_size=8, n_labeled_per_batch=2)


# ── multi-directory datasets ────────────────────────────────────────────

def test_a_directory_list_is_split_and_trimmed():
    assert split_data_dirs("a, b ,c") == ["a", "b", "c"]
    assert split_data_dirs("solo") == ["solo"]
    assert split_data_dirs(None) == []


def _prep_dir(root, name, n, baseline="xtb", e0=-1.0):
    d = os.path.join(root, name)
    os.makedirs(d)
    for i in range(n):
        np.savez(os.path.join(d, f"{i:04d}.npz"), atom_numbers=np.array([8, 1]))
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump({"baseline_model": baseline, "basis": "sto-3g"}, f)
    with open(os.path.join(d, "e0s.json"), "w") as f:
        json.dump({"1": e0, "8": -75.0}, f)
    return d


def test_two_directories_load_as_one_dataset():
    with tempfile.TemporaryDirectory() as root:
        cheap = _prep_dir(root, "cheap", 5)
        dft = _prep_dir(root, "dft", 2)
        files = list_npz(f"{cheap},{dft}")
        assert len(files) == 7
        assert len(list_npz(cheap)) == 5


def test_incompatible_preps_are_refused_rather_than_averaged():
    with tempfile.TemporaryDirectory() as root:
        a = _prep_dir(root, "a", 2, baseline="xtb")
        b = _prep_dir(root, "b", 2, baseline="hf_sto3g")
        with pytest.raises(SystemExit, match="prepared differently"):
            check_prep_compatible([a, b])


def test_mismatched_atom_references_are_refused():
    with tempfile.TemporaryDirectory() as root:
        a = _prep_dir(root, "a", 2, e0=-1.0)
        b = _prep_dir(root, "b", 2, e0=-2.0)
        with pytest.raises(SystemExit, match="isolated-atom"):
            check_prep_compatible([a, b])


def main():
    import inspect
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction)
           if n.startswith("test_")]
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        for name, fn in fns:
            fn()
            print(f"  ok  {name}")
    finally:
        torch.set_default_dtype(prev)
    print(f"\n{len(fns)} checks passed")


if __name__ == "__main__":
    main()
