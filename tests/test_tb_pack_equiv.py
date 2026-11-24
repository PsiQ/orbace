# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""The packed AO bookkeeping must change nothing but where the work happens.

``build_tb_pack`` moves the padded AO tables out of the training step and into
the collate function. That is a pure relocation, so features and gradients are
required to match the in-step path EXACTLY -- not to a tolerance. Any nonzero
difference means the pack and the derived tables disagree somewhere, which would
silently train a different model.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbace.data.tb_features import TB_PACK_KEYS, build_tb_pack, tb_features


def make_batch():
    """Three molecules of deliberately different AO counts, so the pack has to
    pad, plus a water repeated to catch index-vs-composition mixups."""
    mols = [
        ([8, 1, 1], [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]),
        ([6, 1, 1, 1, 1], [[0.0, 0.0, 0.0], [0.629, 0.629, 0.629],
                           [-0.629, -0.629, 0.629], [-0.629, 0.629, -0.629],
                           [0.629, -0.629, -0.629]]),
        ([7, 1, 1, 1], [[0.0, 0.0, 0.116], [0.0, 0.939, -0.271],
                        [0.813, -0.469, -0.271], [-0.813, -0.469, -0.271]]),
    ]
    pos, Z, batch_atom = [], [], []
    for g, (zs, xyz) in enumerate(mols):
        pos += xyz
        Z += zs
        batch_atom += [g] * len(zs)
    pos = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    Z = torch.tensor(Z, dtype=torch.long)
    batch_atom = torch.tensor(batch_atom, dtype=torch.long)

    # Dense intra-molecular edges.
    src, dst = [], []
    off = 0
    for zs, _ in mols:
        n = len(zs)
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(off + i)
                    dst.append(off + j)
        off += n
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    z_lists = [tuple(zs) for zs, _ in mols]
    return pos, Z, edge_index, batch_atom, len(mols), z_lists


def run(pos, Z, edge_index, batch_atom, num_graphs, pack):
    node, edge = tb_features(pos, Z, edge_index, batch_atom, num_graphs, pack=pack)
    loss = node.square().sum() + edge.square().sum()
    grad, = torch.autograd.grad(loss, pos, create_graph=True)
    return node, edge, grad


def main():
    pos, Z, edge_index, batch_atom, num_graphs, z_lists = make_batch()

    pack = build_tb_pack(z_lists)
    missing = [k for k in TB_PACK_KEYS if k not in pack]
    assert not missing, f"build_tb_pack missing keys: {missing}"

    ref = run(pos, Z, edge_index, batch_atom, num_graphs, None)
    got = run(pos, Z, edge_index, batch_atom, num_graphs, pack)

    ok = True
    for name, a, b in zip(("node", "edge", "grad"), ref, got):
        d = (a - b).abs().max().item()
        exact = torch.equal(a, b)
        print(f"  {name:5} max|pack-derived| = {d:.3e}  bit-identical={exact}"
              f"  (scale {a.abs().max().item():.3e})")
        ok &= exact

    # The pack must also survive a second backward, since the force loss needs it.
    g2, = torch.autograd.grad(got[2].square().sum(), pos, retain_graph=True)
    finite = bool(torch.isfinite(g2).all())
    print(f"  double backward finite = {finite}")
    ok &= finite

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """pytest entry point; main() returns 0 on success, 1 on failure."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
