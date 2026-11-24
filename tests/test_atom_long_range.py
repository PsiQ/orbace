# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Atom-model long-range electrostatics: off by default, conservative when on.

Run directly:  python tests/test_atom_long_range.py
"""
import functools
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orbace.data.invariant_graph import InvariantDataset, collate_invariant_data
from orbace.data.tb_features import N_NODE_SCALARS, N_EDGE_FEATURES
from orbace.modules.atom_model import AtomOrbACE

R_CUT = 5.0

ETHANOL = (["C", "C", "O", "H", "H", "H", "H", "H", "H"], np.array([
    [-0.748, -0.015, 0.024], [0.558, 0.420, -0.317], [1.488, -0.584, 0.075],
    [-1.546, 0.648, -0.322], [-0.906, -1.020, -0.374], [-0.822, -0.095, 1.112],
    [0.665, 1.425, 0.104], [0.663, 0.483, -1.405], [2.359, -0.215, -0.134],
]))


def _write_extxyz(path, n_frames=3):
    rng = np.random.default_rng(0)
    lines = []
    for sym, e in (("H", -13.6), ("C", -1027.0), ("O", -2035.0)):
        lines += ["1", f'Properties=species:S:1:pos:R:3:REF_forces:R:3 '
                  f'config_type=IsolatedAtom REF_energy={e} pbc="F F F"',
                  f"{sym} 0.0 0.0 0.0 0.0 0.0 0.0"]
    species, pos0 = ETHANOL
    for _ in range(n_frames):
        pos = pos0 + rng.normal(scale=0.03, size=pos0.shape)
        forces = rng.normal(scale=0.1, size=pos0.shape)
        lines += [str(len(species)),
                  f'Properties=species:S:1:pos:R:3:REF_forces:R:3 '
                  f'REF_energy={-4200.0 + rng.normal(scale=0.1):.6f} pbc="F F F"']
        lines += [f"{s} " + " ".join(f"{v:.6f}" for v in (*p, *f))
                  for s, p, f in zip(species, pos, forces)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


@functools.lru_cache(maxsize=None)
def prep_dir():
    """Prepare a few perturbed ethanol frames with the atom family (no SCF)."""
    tmp = tempfile.mkdtemp(prefix="orbace_lr_")
    xyz = os.path.join(tmp, "ethanol.xyz")
    out = os.path.join(tmp, "prep")
    _write_extxyz(xyz)
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "prepare.py"),
         "--method", "atom", "--input", xyz, "--output_dir", out,
         "--basis", "sto-3g", "--num_workers", "1", "--baseline_model", "none"],
        check=True, capture_output=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ROOT})
    return out


def build(use_long_range):
    torch.manual_seed(0)
    return AtomOrbACE(
        r_max=R_CUT, num_interactions=2,
        hidden_irreps="32x0e + 32x1o", MLP_irreps="16x0e",
        n_node_scalars=N_NODE_SCALARS, n_edge_features=N_EDGE_FEATURES,
        avg_num_neighbors=10.0,
        use_long_range=use_long_range,
    ).double()


def batch(n=2, pack=True):
    ds = InvariantDataset(data_dir=prep_dir(), r_cut=R_CUT, basis="sto-3g", mf=False)
    b = collate_invariant_data([ds[i] for i in range(n)], tb_pack=pack)
    for k, v in b.items():
        if torch.is_tensor(v) and v.is_floating_point():
            b[k] = v.double()
    return b


def test_default_is_off():
    m = build(False)
    assert not m.use_long_range
    assert not hasattr(m, "long_range")
    out = m(batch(), training=True)
    assert out["lr_energy"] is None
    print("default off: no long_range submodule, lr_energy is None")


def test_on_adds_params_and_starts_small():
    off, on = build(False), build(True)
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on > n_off
    out = on(batch(), training=True)
    lr = out["lr_energy"]
    assert lr is not None and torch.isfinite(lr).all()
    # Small enough to perturb nothing (references are ~1e3 eV), but NOT zero --
    # see test_gradient_is_alive_at_init for why zero would be fatal.
    assert lr.abs().max() < 1e-2, lr
    print(f"on: +{n_on - n_off} params, |lr_energy| at init = "
          f"{lr.abs().max().item():.2e} eV ({n_off} -> {n_on})")


def test_gradient_is_alive_at_init():
    """The charge readout must not be initialized to exactly zero.

    E_qq is quadratic in the charges, so dE/dq vanishes identically at q = 0. If
    the readout weight is zero too, dE/dh = W^T dE/dq is also zero, so no
    gradient reaches any layer of charge_net and the term can never leave zero --
    it is dead for the entire run rather than growing from data. This guards the
    fix: every parameter of charge_net must see a nonzero gradient at init.
    """
    m = build(True)
    out = m(batch(), training=True, compute_force=True)
    (out["energy"].pow(2).mean() + out["forces"].pow(2).mean()).backward()

    grads = {f"charge_net.{i}": p.grad.abs().max().item()
             for i, layer in enumerate(m.long_range.charge_net)
             for p in layer.parameters()
             if p.grad is not None}
    assert grads, "charge_net received no gradients at all"
    for name, g in grads.items():
        assert g > 0.0, f"{name} has zero gradient at init -- readout is dead"
    print("  gradient alive at init: " +
          ", ".join(f"{k}={v:.2e}" for k, v in grads.items()))


def test_term_trains_off_its_initial_value():
    """A few optimizer steps must actually move the charge readout."""
    m = build(True)
    w = m.long_range.charge_net[-1].weight
    before = w.detach().abs().max().item()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    b = batch()
    for _ in range(10):
        opt.zero_grad()
        out = m(b, training=True, compute_force=True)
        (out["energy"].pow(2).mean() + out["forces"].pow(2).mean()).backward()
        opt.step()
    after = w.detach().abs().max().item()
    assert after != before, "readout did not move in 10 steps"
    print(f"  readout |W| moved {before:.3e} -> {after:.3e} over 10 steps")


def test_forces_stay_conservative():
    """Finite-difference check with the long-range term made active.

    The readout starts deliberately small, so the bias is raised first to put the
    term at a magnitude where the check can actually resolve it.
    """
    m = build(True)
    with torch.no_grad():
        m.long_range.charge_net[-1].bias.fill_(0.05)

    b = batch(n=1)
    out = m(b, training=True, compute_force=True)
    assert out["lr_energy"].abs().sum() > 0, "long-range term still inert"
    f_analytic = out["forces"]

    pos = b["atom_positions"]
    h = 1e-5
    worst = 0.0
    for i in range(pos.shape[0]):
        for d in range(3):
            for sign in (+1, -1):
                shifted = pos.clone()
                shifted[i, d] += sign * h
                bb = dict(b)
                bb["atom_positions"] = shifted
                e = m(bb, training=False, compute_force=False)["energy"].sum()
                if sign > 0:
                    e_plus = e
                else:
                    e_minus = e
            fd = -(e_plus - e_minus) / (2 * h)
            worst = max(worst, abs(fd.item() - f_analytic[i, d].item()))
    print(f"max |analytic - finite difference| force error: {worst:.2e} eV/A")
    assert worst < 1e-4, worst


if __name__ == "__main__":
    test_default_is_off()
    test_on_adds_params_and_starts_small()
    test_gradient_is_alive_at_init()
    test_term_trains_off_its_initial_value()
    test_forces_stay_conservative()
    print("\nall long-range checks passed")
