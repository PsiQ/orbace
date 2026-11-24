# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Multi-fidelity arm checks: mfdh (additive) vs mfah (parallel absolute heads).

Run either way:  python tests/test_mfah.py   |   pytest tests/test_mfah.py

The checks are named test_* so pytest collects them individually. main() still
drives them in order for the direct-invocation path, and _float64 below keeps
the two paths equivalent: the tolerances here assume float64, which main() used
to set on its own.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbace.data.tb_features import N_NODE_SCALARS  # noqa: E402
from orbace.modules.atom_model import AtomOrbACE  # noqa: E402
from orbace.modules.models import (  # noqa: E402
    model_fidelity_mode,
    model_is_multifidelity,
)

NUM_SPECIES = 10


@pytest.fixture(autouse=True)
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _build(fidelity, seed=0):
    torch.manual_seed(seed)
    return AtomOrbACE(
        r_max=4.0, num_bessel=4, num_polynomial_cutoff=5, max_ell=2,
        num_interactions=2, num_species=NUM_SPECIES,
        n_node_scalars=N_NODE_SCALARS, hidden_irreps="16x0e+16x1o",
        MLP_irreps="8x0e", avg_num_neighbors=6.0, atom_e0=np.zeros(NUM_SPECIES),
        pair_repulsion=False, delta_method=fidelity,
        base_atom_e0=np.linspace(-1.0, -0.1, NUM_SPECIES),
    )


def _water_batch():
    """Two water molecules as one batch, with positions requiring grad."""
    pos = torch.tensor([
        [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
        [3.0, 0.0, 0.0], [3.96, 0.1, 0.0], [2.76, 0.93, 0.0],
    ], dtype=torch.get_default_dtype(), requires_grad=True)
    numbers = torch.tensor([8, 1, 1, 8, 1, 1])
    batch_atom = torch.tensor([0, 0, 0, 1, 1, 1])
    # Fully connected within each molecule.
    src, dst = [], []
    for off in (0, 3):
        for i in range(3):
            for j in range(3):
                if i != j:
                    src.append(off + i); dst.append(off + j)
    edge_index = torch.tensor([src, dst])
    torch.manual_seed(1)
    return {
        "atom_positions": pos,
        "atom_numbers": numbers,
        "batch_atom": batch_atom,
        "edge_index": edge_index,
        "node_scalars": torch.randn(6, N_NODE_SCALARS) * 0.1,
        "ptr": torch.tensor([0, 3, 6]),
        "n_atoms": torch.tensor([3, 3]),
        "num_graphs": 2,
    }


def _run(model, data):
    return model(dict(data), training=True, compute_force=True)


def test_modes_and_heads():
    """Each arm reports the right mode and owns the right heads."""
    reg, dh, ah = _build("regular"), _build("mfdh"), _build("mfah")

    assert model_fidelity_mode(reg) == "regular"
    assert model_fidelity_mode(dh) == "mfdh"
    assert model_fidelity_mode(ah) == "mfah"
    assert not model_is_multifidelity(reg)
    assert model_is_multifidelity(dh) and model_is_multifidelity(ah)

    assert hasattr(dh, "mf_base_head") and hasattr(dh, "mf_qc_delta_head")
    assert not hasattr(dh, "mf_low_head")
    assert hasattr(ah, "mf_low_head")
    assert not hasattr(ah, "mf_base_head") and not hasattr(ah, "mf_qc_delta_head")

    # The legacy value a pre-split checkpoint carries must still resolve.
    dh.delta_method = "mf"
    assert model_fidelity_mode(dh) == "mfdh"
    print("PASS  modes and heads")


def test_mfah_matches_direct():
    """MFAH's high-fidelity output must equal a direct model's.

    This is the defining property of the arm: the low head is parallel, so it
    contributes nothing to `energy`/`forces`, and any accuracy difference vs a
    direct model comes purely from how the auxiliary task shapes the backbone.
    """
    data = _water_batch()
    reg, ah = _build("regular", seed=7), _build("mfah", seed=7)
    # Same seed builds the shared trunk identically; the extra low head is drawn
    # after it, so the backbone weights coincide.
    sd_reg, sd_ah = reg.state_dict(), ah.state_dict()
    shared = [k for k in sd_reg if k in sd_ah]
    for k in shared:
        assert torch.equal(sd_reg[k], sd_ah[k]), k

    out_reg, out_ah = _run(reg, data), _run(ah, data)
    de = (out_reg["energy"] - out_ah["energy"]).abs().max().item()
    df = (out_reg["forces"] - out_ah["forces"]).abs().max().item()
    assert de < 1e-9, f"energy differs by {de}"
    assert df < 1e-9, f"forces differ by {df}"

    # And the low head is live: it predicts something and carries a gradient.
    assert out_ah["base_energy"] is not None
    assert out_ah["base_forces"] is not None
    assert out_ah["base_forces"].abs().max().item() > 0
    print(f"PASS  mfah high-fidelity path identical to direct (dE={de:.2e}, dF={df:.2e})")


def test_mfdh_is_additive():
    """MFDH's total must move with its base head; MFAH's must not."""
    data = _water_batch()
    for mode, head_name, expect_shift in (
        ("mfdh", "mf_base_head", True),
        ("mfah", "mf_low_head", False),
    ):
        model = _build(mode, seed=3)
        e_before = _run(model, data)["energy"].detach().clone()
        with torch.no_grad():
            head = getattr(model, head_name)
            head.mlp[-1].bias += 1.0
        out = _run(model, data)
        d_total = (out["energy"] - e_before).abs().max().item()
        d_base = (out["base_energy"] - (e_before * 0)).abs().max().item()
        assert d_base > 0, f"{mode}: base head produced nothing"
        if expect_shift:
            assert d_total > 1e-6, f"{mode}: total ignored the base head"
        else:
            assert d_total < 1e-9, f"{mode}: total moved by {d_total} with the low head"
        print(f"PASS  {mode} total shift on base-head bias = {d_total:.2e} "
              f"({'additive' if expect_shift else 'parallel'})")


def test_loss_and_backward():
    """The shared MF loss trains both arms: all four terms present, grads flow."""
    from orbace.tools.train import MFEnergyForcesLoss

    data = _water_batch()
    torch.manual_seed(5)
    targets = {
        "energy": torch.tensor([-76.4, -76.3]),
        "forces": torch.randn(6, 3) * 0.05,
        "baseline_energy": torch.tensor([-75.9, -75.8]),
        "baseline_forces": torch.randn(6, 3) * 0.05,
        "n_atoms": data["n_atoms"],
    }
    loss_fn = MFEnergyForcesLoss(huber=True, huber_delta=0.01, forces_normalizer=0.5)
    for mode, head_name in (("mfdh", "mf_base_head"), ("mfah", "mf_low_head")):
        model = _build(mode, seed=11)
        losses = loss_fn(_run(model, data), targets)
        for term in ("energy", "forces", "base_energy", "base_forces", "total"):
            assert term in losses, f"{mode}: missing loss term {term}"
            assert torch.isfinite(losses[term]), f"{mode}: {term} not finite"
        losses["total"].backward()
        head = getattr(model, head_name)
        g_head = max(p.grad.abs().max().item() for p in head.parameters()
                     if p.grad is not None)
        g_trunk = model.interactions[0].linear.weight.grad
        assert g_head > 0, f"{mode}: cheap head got no gradient"
        assert g_trunk is not None and g_trunk.abs().max().item() > 0, \
            f"{mode}: backbone got no gradient"
        print(f"PASS  {mode} loss={losses['total'].item():.4f} "
              f"cheap-head grad={g_head:.2e} backbone grad={g_trunk.abs().max().item():.2e}")


def test_freeze_backbone():
    """Finetuning a frozen backbone must leave each arm's MF heads trainable."""
    for mode, head_name in (("mfdh", "mf_base_head"), ("mfah", "mf_low_head")):
        model = _build(mode)
        model.freeze_backbone()
        head = getattr(model, head_name)
        assert all(p.requires_grad for p in head.parameters()), mode
        assert not model.interactions[0].linear.weight.requires_grad, mode
        print(f"PASS  {mode} freeze_backbone keeps {head_name} trainable")


def main():
    torch.set_default_dtype(torch.float64)
    test_modes_and_heads()
    test_mfah_matches_direct()
    test_mfdh_is_additive()
    test_loss_and_backward()
    test_freeze_backbone()
    print("\nAll MFAH/MFDH checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
