# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""BucketedBatchSampler: does it cover the data, and does it actually save work?

Two things have to hold at once. It must be a legitimate sampler -- every frame
seen exactly once per epoch, batch sizes as requested, membership varying between
epochs -- and it must genuinely remove the padding waste it exists to remove,
which is measured here as the O(nao^3) FLOP ratio against an unpadded solve.

Run directly:  python tests/test_bucketed_sampler.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbace.data.invariant_graph import BucketedBatchSampler  # noqa: E402

# Roughly the real ANI-1x AO spread: median ~36, long tail to ~105.
rng = np.random.default_rng(0)
SIZES = np.clip(rng.lognormal(3.55, 0.35, size=5000).astype(int), 6, 105).tolist()
BATCH = 16


def waste(batches, sizes):
    """Padded FLOPs / unpadded FLOPs, the quantity bucketing is meant to cut."""
    r = []
    for b in batches:
        n = np.array([sizes[i] for i in b], dtype=float)
        r.append(len(n) * n.max() ** 3 / (n ** 3).sum())
    return float(np.mean(r))


def random_baseline():
    idx = rng.permutation(len(SIZES))
    return [idx[i:i + BATCH].tolist() for i in range(0, len(idx), BATCH)]


def check_coverage(name, batches, expect_n):
    flat = [i for b in batches for i in b]
    ok = True
    if sorted(flat) != list(range(expect_n)):
        dup = len(flat) - len(set(flat))
        print(f"  FAIL {name}: coverage broken "
              f"({len(flat)} emitted, {len(set(flat))} unique, {dup} duplicated)")
        ok = False
    else:
        print(f"  OK   {name}: all {expect_n} frames exactly once")
    return ok


def main():
    ok = True
    s = BucketedBatchSampler(SIZES, BATCH, shuffle=True, seed=0)

    e0 = list(iter(s))
    e1 = list(iter(s))
    ok &= check_coverage("epoch 0", e0, len(SIZES))
    ok &= check_coverage("epoch 1", e1, len(SIZES))

    if len(s) != len(e0):
        print(f"  FAIL __len__={len(s)} but emitted {len(e0)} batches")
        ok = False
    else:
        print(f"  OK   __len__ matches emitted batch count ({len(s)})")

    bad = [len(b) for b in e0 if len(b) != BATCH]
    # Only the final batch of each pool may be short, and only when the pool does
    # not divide evenly.
    if any(n > BATCH for n in bad):
        print(f"  FAIL over-full batches: {bad[:5]}")
        ok = False
    else:
        print(f"  OK   no batch exceeds batch_size ({len(bad)} short batches)")

    if e0 == e1:
        print("  FAIL batch membership identical across epochs (no shuffling)")
        ok = False
    else:
        same = sum(1 for a, b in zip(e0, e1) if a == b)
        print(f"  OK   membership varies across epochs ({same}/{len(e0)} identical)")

    w_rand = waste(random_baseline(), SIZES)
    w_buck = waste(e0, SIZES)
    print(f"\n  padding waste: random {w_rand:.2f}x -> bucketed {w_buck:.2f}x "
          f"at the default pool ({w_rand / w_buck:.1f}x less wasted work)")
    if w_buck > 0.6 * w_rand:
        print("  FAIL bucketing barely reduced the padding waste")
        ok = False

    # pool_batches is the speed/heterogeneity dial, so pin that it actually
    # dials: more batches per pool must mean strictly less padding waste.
    curve = [(pb, waste(list(iter(BucketedBatchSampler(
        SIZES, BATCH, shuffle=True, pool_batches=pb, seed=0))), SIZES))
        for pb in (2, 8, 50)]
    print("  waste vs pool_batches: "
          + ", ".join(f"{pb}->{w:.2f}x" for pb, w in curve))
    if not all(a[1] > b[1] for a, b in zip(curve, curve[1:])):
        print("  FAIL padding waste is not monotonically decreasing in pool_batches")
        ok = False
    if curve[-1][1] > 1.15:
        print("  FAIL a large pool should remove nearly all the padding waste")
        ok = False

    # shuffle=False is used for validation, where a full sort is optimal.
    sv = BucketedBatchSampler(SIZES, BATCH, shuffle=False)
    v0, v1 = list(iter(sv)), list(iter(sv))
    ok &= check_coverage("shuffle=False", v0, len(SIZES))
    if v0 != v1:
        print("  FAIL shuffle=False is not deterministic")
        ok = False
    else:
        print(f"  OK   shuffle=False deterministic, waste {waste(v0, SIZES):.3f}x")

    sd = BucketedBatchSampler(SIZES, BATCH, shuffle=True, drop_last=True, seed=0)
    bd = list(iter(sd))
    if len(sd) != len(bd) or any(len(b) != BATCH for b in bd):
        print(f"  FAIL drop_last: __len__={len(sd)} emitted={len(bd)}, "
              f"sizes={{{set(len(b) for b in bd)}}}")
        ok = False
    else:
        print(f"  OK   drop_last: {len(bd)} full batches, __len__ agrees")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


def test_main():
    """pytest entry point; main() returns 0 on success, 1 on failure."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
