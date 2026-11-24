# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Equivalence test: batched TB featurizer vs the per-molecule reference path.

The batched path in ``tb_features`` pads every molecule in a batch to the batch
maximum and issues two batched ``eigh`` calls instead of 2*batch_size serial
ones. That is only a legitimate optimization if it reproduces the reference
path's features AND its gradients, so this test pins both.

Run directly:  python tests/test_tb_batched.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbace.data.tb_features import (  # noqa: E402
    _molecule_features, tb_features,
)

torch.manual_seed(0)

# Deliberately mixed sizes and elements so padding is actually exercised: if the
# batch were uniform, the padded block would be empty and the test would pass
# without proving anything.
MOLECULES = [
    # (atomic numbers, coordinates in Angstrom)
    ([8, 1, 1], [[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]]),
    ([6, 1, 1, 1, 1], [[0.0, 0.0, 0.0], [0.629, 0.629, 0.629],
                       [-0.629, -0.629, 0.629], [-0.629, 0.629, -0.629],
                       [0.629, -0.629, -0.629]]),
    ([7, 1, 1, 1], [[0.0, 0.0, 0.120], [0.0, 0.939, -0.280],
                    [0.813, -0.469, -0.280], [-0.813, -0.469, -0.280]]),
    ([6, 6, 8, 1, 1, 1, 1], [[0.0, 0.0, 0.0], [1.50, 0.05, 0.10],
                             [2.10, 1.20, -0.15], [-0.50, 0.95, 0.20],
                             [-0.45, -0.90, 0.55], [-0.30, -0.15, -1.00],
                             [1.95, -0.85, 0.35]]),
]


def build_batch(dtype=torch.float64, device="cpu"):
    """Return (positions, numbers, edge_index, batch_atom, num_graphs).

    Edges are fully connected within each molecule and never cross molecules,
    matching what the real graph collate produces.
    """
    pos, num, batch, edges = [], [], [], []
    off = 0
    for g, (zs, xyz) in enumerate(MOLECULES):
        n = len(zs)
        pos.extend(xyz)
        num.extend(zs)
        batch.extend([g] * n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append((off + i, off + j))
        off += n
    positions = torch.tensor(pos, dtype=dtype, device=device, requires_grad=True)
    numbers = torch.tensor(num, dtype=torch.long, device=device)
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).T
    batch_atom = torch.tensor(batch, dtype=torch.long, device=device)
    return positions, numbers, edge_index, batch_atom, len(MOLECULES)


def reference_features(positions, numbers, edge_index, batch_atom, num_graphs):
    """The pre-batching implementation: one molecule at a time."""
    n_atoms = positions.shape[0]
    node_out = positions.new_zeros((n_atoms, 12))
    edge_out = positions.new_zeros((edge_index.shape[1], 6))

    counts = torch.bincount(batch_atom, minlength=num_graphs)
    offsets = torch.zeros_like(counts)
    offsets[1:] = torch.cumsum(counts, 0)[:-1]
    local_idx = (torch.arange(n_atoms, device=positions.device)
                 - offsets[batch_atom])
    src, dst = edge_index[0], edge_index[1]
    edge_mol = batch_atom[src]

    for g in range(num_graphs):
        gidx = torch.nonzero(batch_atom == g, as_tuple=False).flatten()
        zs = [int(z) for z in numbers[gidx].tolist()]
        node_g, pair_g = _molecule_features(positions[gidx], zs)
        node_out = node_out.index_put((gidx,), node_g)
        emask = edge_mol == g
        if bool(emask.any()):
            vals = pair_g[local_idx[src[emask]], local_idx[dst[emask]]]
            edge_out = edge_out.index_put((torch.nonzero(emask).flatten(),), vals)
    return node_out, edge_out


def check_device(device):
    """Run the full equivalence check on one device.

    Devices are checked separately and both must pass: an earlier version of the
    padding collided with the real overlap spectrum, and whether eigh then mixed
    eigenvectors across the padded boundary depended on the LAPACK build, so the
    gradient check passed on one machine and failed on another.
    """
    print(f"\n--- device: {device} ---")
    ok = True

    # ---- features ----------------------------------------------------------
    pos_a, num, ei, ba, ng = build_batch(device=device)
    node_b, edge_b = tb_features(pos_a, num, ei, ba, ng)

    pos_b, _, _, _, _ = build_batch(device=device)
    node_r, edge_r = reference_features(pos_b, num, ei, ba, ng)

    dn = (node_b - node_r).abs().max().item()
    de = (edge_b - edge_r).abs().max().item()
    sn = node_r.abs().max().item()
    se = edge_r.abs().max().item()
    print(f"node: max|batched-ref| = {dn:.3e}  (scale {sn:.3e})")
    print(f"edge: max|batched-ref| = {de:.3e}  (scale {se:.3e})")
    if dn > 1e-9 * max(sn, 1.0) or de > 1e-9 * max(se, 1.0):
        print("FAIL: features differ beyond float64 round-off")
        ok = False

    # ---- first derivatives (this is what becomes the force) ----------------
    # A random projection rather than a plain sum, so cancellation cannot hide a
    # sign or index error in individual feature columns.
    wn = torch.randn_like(node_b)
    we = torch.randn_like(edge_b)
    gb, = torch.autograd.grad((node_b * wn).sum() + (edge_b * we).sum(), pos_a)
    gr, = torch.autograd.grad((node_r * wn).sum() + (edge_r * we).sum(), pos_b)
    dg = (gb - gr).abs().max().item()
    print(f"grad: max|batched-ref| = {dg:.3e}  (scale {gr.abs().max():.3e})")
    if dg > 1e-8 * max(gr.abs().max().item(), 1.0):
        print("FAIL: gradients differ beyond float64 round-off")
        ok = False

    # ---- double backward (needed for the conservative force loss) ----------
    pos_c, num_c, ei_c, ba_c, ng_c = build_batch(device=device)
    node_c, edge_c = tb_features(pos_c, num_c, ei_c, ba_c, ng_c)
    e_pred = (node_c ** 2).sum() + (edge_c ** 2).sum()
    forces, = torch.autograd.grad(e_pred, pos_c, create_graph=True)
    loss = (forces ** 2).sum()
    g2, = torch.autograd.grad(loss, pos_c)
    finite = bool(torch.isfinite(g2).all())
    print(f"double backward finite: {finite}  max|d2| = {g2.abs().max():.3e}")
    if not finite:
        print("FAIL: double backward produced non-finite values")
        ok = False

    # ---- single molecule (num_graphs=1, no padding at all) -----------------
    pos1 = torch.tensor(MOLECULES[0][1], dtype=torch.float64, device=device,
                        requires_grad=True)
    num1 = torch.tensor(MOLECULES[0][0], dtype=torch.long, device=device)
    ei1 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device=device)
    ba1 = torch.zeros(3, dtype=torch.long, device=device)
    n1, e1 = tb_features(pos1, num1, ei1, ba1, 1)
    if not (torch.isfinite(n1).all() and torch.isfinite(e1).all()):
        print("FAIL: single-molecule path produced non-finite features")
        ok = False
    else:
        print("single-molecule path: finite")

    return ok


def main():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    results = {d: check_device(d) for d in devices}
    for d, r in results.items():
        print(f"{d}: {'PASS' if r else 'FAIL'}")
    ok = all(results.values())
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


def test_main():
    """pytest entry point; main() returns 0 on success, 1 on failure."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
