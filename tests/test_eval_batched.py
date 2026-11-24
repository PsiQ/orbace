# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Batched npz evaluation must reproduce the per-frame metrics exactly.

Evaluation is where the reported numbers come from, so batching it is only
acceptable if it changes nothing but the runtime. This pins the new batched path
against a straight reimplementation of the per-frame loop it replaced, on the
real trained model and test data.

Three things are checked, because they can fail independently:
  - sequential batching reproduces the per-frame errors elementwise, in order;
  - several batch sizes agree, so nothing depends on where batch edges land;
  - bucketed batching (which reorders frames) yields the same multiset of
    per-frame errors, and therefore the same mean/median/max.

Point it at a trained atom-family checkpoint and its prepared test set:

    ORBACE_EVAL_MODEL=ck/best_model.pt ORBACE_EVAL_DATA=adata_test \
        python tests/test_eval_batched.py

Skips cleanly if either is unset or absent.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

MODEL = os.environ.get("ORBACE_EVAL_MODEL", "")
DATA = os.environ.get("ORBACE_EVAL_DATA", "")
N_FRAMES = 192


def per_frame_reference(model, ds, device, e0s, is_mf):
    """The pre-batching loop: collate one frame, forward, accumulate."""
    from orbace.data.invariant_graph import collate_invariant_data

    eE, eF, eE_tot = [], [], []
    for i in range(len(ds)):
        d = ds[i]
        batch = collate_invariant_data([d])
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.enable_grad():
            out = model(batch, training=False, compute_force=True)
        e_pred = float(out["energy"].detach().cpu().item())
        f_pred = out["forces"].detach().cpu().numpy()

        zs = d.atom_numbers.tolist()
        e0_sum = sum(e0s.get(int(z), 0.0) for z in zs)
        base_e = (float(d.baseline_energy)
                  if getattr(d, "baseline_energy", None) is not None else 0.0)
        base_f = (d.baseline_forces.numpy()
                  if getattr(d, "baseline_forces", None) is not None else None)

        if is_mf:
            e_pred_abs, e_ref_abs = e_pred + e0_sum, float(d.energy) + base_e + e0_sum
            f_ref = d.forces.numpy() + (base_f if base_f is not None else 0.0)
        else:
            e_pred_abs = e_pred + e0_sum + base_e
            e_ref_abs = float(d.energy) + e0_sum + base_e
            f_ref = d.forces.numpy()

        eE.append(abs(e_pred_abs - e_ref_abs) / max(len(zs), 1))
        eE_tot.append(abs(e_pred_abs - e_ref_abs))
        eF.append(float(np.mean(np.abs(f_pred - f_ref))))
    return eE, eF, eE_tot


# The network runs in float32, so grouping frames into a batch reorders the
# scatter-reductions behind the energy and changes the last bits of the result.
# Observed effect is ~1e-6 relative, six orders below anything reported. A real
# defect here -- frames misaligned against their references, a bad batch offset --
# would instead show up at the scale of the errors themselves, so this tolerance
# still catches those by a wide margin.
FLOAT32_TOL = 1e-4


def compare(name, a, b, sort=False, tol=FLOAT32_TOL):
    ok = True
    for label, x, y in zip(("E/atom", "F", "E_tot"), a, b):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.shape != y.shape:
            print(f"  FAIL {name} {label}: {x.shape} vs {y.shape}")
            ok = False
            continue
        if sort:
            x, y = np.sort(x), np.sort(y)
        scale = max(float(np.abs(y).max()), 1e-12)
        err = float(np.abs(x - y).max())
        flag = "OK  " if err <= tol * max(scale, 1.0) else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  {flag} {name} {label}: max|diff| = {err:.3e} (scale {scale:.3e})")
    return ok


def main():
    if not os.path.exists(MODEL) or not os.path.isdir(DATA):
        print(f"SKIP: need {MODEL} and {DATA}")
        return 0

    from eval import _atom_eval_npz, _load_model, _resolve_device  # noqa: E402
    from orbace.data.invariant_graph import InvariantDataset  # noqa: E402
    from orbace.modules.models import model_is_multifidelity  # noqa: E402

    device = _resolve_device("auto")
    model = _load_model(MODEL, device)
    e0s = {int(k): float(v) for k, v in getattr(model, "e0s", {}).items()}
    is_mf = model_is_multifidelity(model)
    print(f"device={device}  frames={N_FRAMES}")

    ds = InvariantDataset(data_dir=DATA, r_cut=5.0, lazy=True, max_frames=N_FRAMES)
    ref = per_frame_reference(model, ds, device, e0s, is_mf)
    print(f"reference: {len(ref[0])} frames, "
          f"F MAE {np.mean(ref[1]):.6f} eV/A")

    ok = True
    # Sequential batching: frame order is preserved, so compare elementwise.
    for bs in (1, 7, 32):
        got = _atom_eval_npz(model, ds, device, bs, e0s, is_mf, bucket=False)
        ok &= compare(f"seq bs={bs:<2d}", got, ref)

    # Bucketing reorders frames; the multiset of per-frame errors must survive.
    got = _atom_eval_npz(model, ds, device, 32, e0s, is_mf, bucket=True)
    ok &= compare("bucketed  ", got, ref, sort=True)
    # The aggregates are what actually gets reported, so check them directly.
    stats_ok = True
    for label, x, y in zip(("E/atom", "F", "E_tot"), got, ref):
        for stat, fn in (("mean", np.mean), ("median", np.median), ("max", np.max)):
            dx, dy = float(fn(x)), float(fn(y))
            if abs(dx - dy) > FLOAT32_TOL * max(abs(dy), 1e-12):
                print(f"  FAIL bucketed {label} {stat}: {dx:.9f} vs {dy:.9f}")
                stats_ok = False
    if stats_ok:
        print("  OK   bucketed mean/median/max match the reference")
    ok &= stats_ok

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


def test_main():
    """pytest entry point.

    main() returns 0 both on success and on the missing-data path, so the
    precondition is checked here instead -- otherwise an absent checkpoint
    would report as a pass rather than a skip.
    """
    import pytest
    if not os.path.exists(MODEL) or not os.path.isdir(DATA):
        pytest.skip("set ORBACE_EVAL_MODEL and ORBACE_EVAL_DATA to run")
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
