# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Support for multi-fidelity datasets where only some frames carry a label.

The multi-fidelity arms (``mfdh`` and ``mfah``) learn the expensive level as a
correction to, or alongside, a cheap level. Nothing forces the two levels to
cover the same frames: the natural way to build such a dataset is to run the
cheap method (xTB, GFN2, HF/STO-3G) over everything and the expensive one over
whatever subset the compute budget allowed, which can be a few percent.

Every frame therefore carries ``baseline_energy`` / ``baseline_forces``, but
only the expensive subset carries ``energy`` / ``forces``. Two things follow:

* Collation has to keep the batch index-aligned. The packers below zero-fill
  the absent labels and return a boolean mask; the loss reduces over the mask
  rather than over the whole batch. A batch where every frame is labeled gets
  a mask of ``None`` and is byte-identical to what collation produced before.
* When labels are sparse enough, plain shuffling leaves many batches with no
  expensive frame at all and the high-fidelity gradient arrives only in
  bursts. ``LabelBalancedBatchSampler`` fixes the per-batch count instead, and
  ``auto_hf_per_batch`` decides when that is worth doing.
"""

import json
import os
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Sampler

# Target number of labeled frames per batch when balancing is on "auto".
# The high-fidelity term is a mean over the labeled frames in a batch, so this
# is the effective batch size of that term: below ~8 its gradient is noisy
# enough to show up in the optimizer's moment estimates, and above it the
# returns fall off while the labeled set gets revisited more often.
AUTO_HF_TARGET = 8


# ── collation ───────────────────────────────────────────────────────────

def pack_frame_scalars(
    values: Sequence[Optional[torch.Tensor]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Concatenate one scalar per frame, zero-filling the frames without one.

    Returns ``(packed, mask)``. ``mask`` is None when every frame carried the
    label (the packed tensor is then exactly what the caller would have built
    itself), and ``(None, None)`` when no frame did, which is how a dataset
    with no labels of this kind at all stays indistinguishable from before.
    """
    present = [v is not None for v in values]
    if not any(present):
        return None, None
    ref = next(v for v in values if v is not None)
    packed = torch.cat(
        [(ref.new_zeros(1) if v is None else v.reshape(-1)) for v in values], 0)
    if all(present):
        return packed, None
    return packed, torch.tensor(present, dtype=torch.bool)


def pack_frame_rows(
    values: Sequence[Optional[torch.Tensor]],
    n_rows: Sequence[int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """As ``pack_frame_scalars`` but for per-atom quantities (forces).

    ``n_rows`` gives each frame's atom count, which is what the zero-fill needs
    in order to keep the concatenated rows aligned with ``batch_atom``. The
    mask is per-ROW, so the loss can reduce over it directly.
    """
    present = [v is not None for v in values]
    if not any(present):
        return None, None
    ref = next(v for v in values if v is not None)
    packed = torch.cat(
        [(ref.new_zeros((n, ref.shape[1])) if v is None else v)
         for v, n in zip(values, n_rows)], 0)
    if all(present):
        return packed, None
    mask = torch.cat(
        [torch.full((n,), bool(p), dtype=torch.bool)
         for p, n in zip(present, n_rows)], 0)
    return packed, mask


def masked_mean(per_element: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean of ``per_element`` over the entries ``mask`` selects.

    ``mask=None`` means "everything", so this reduces to a plain mean and the
    fully-labeled path is unchanged. A batch with nothing selected contributes
    exactly zero, still attached to the graph so the parameters feeding it keep
    receiving a (zero) gradient rather than dropping out of the step.
    """
    if mask is None:
        return per_element.mean()
    m = mask.to(per_element.dtype)
    while m.dim() < per_element.dim():
        m = m.unsqueeze(-1)
    m = m.expand_as(per_element)
    denom = m.sum()
    if float(denom) == 0.0:
        return per_element.sum() * 0.0
    return (per_element * m).sum() / denom


# ── dataset assembly ────────────────────────────────────────────────────

def split_data_dirs(spec) -> List[str]:
    """Split a comma-separated ``--data_dir`` / ``--train_dir`` into paths.

    Cheap and expensive frames are usually prepared into separate directories,
    and the cheap one can hold a million files, so physically merging them to
    train on both is not reasonable. Passing them as a list is.
    """
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        parts = [str(s) for s in spec]
    else:
        parts = str(spec).split(",")
    return [p.strip() for p in parts if p.strip()]


def primary_data_dir(spec) -> Optional[str]:
    """First directory of a possibly multi-directory spec.

    Where the sidecar ``metadata.json`` / ``e0s.json`` are read from.
    ``check_prep_compatible`` has already established that the rest agree, so
    any of them would do.
    """
    dirs = split_data_dirs(spec)
    return dirs[0] if dirs else None


def list_npz(spec) -> List[str]:
    """Every ``.npz`` under one directory, or under a comma-separated list.

    Sorted within each directory and concatenated in the order given, so the
    file order is reproducible across runs.
    """
    dirs = split_data_dirs(spec)
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(f"data directory not found: {', '.join(missing)}")
    check_prep_compatible(dirs)
    files: List[str] = []
    for d in dirs:
        files.extend(sorted(os.path.join(d, f)
                            for f in os.listdir(d) if f.endswith(".npz")))
    return files


def check_prep_compatible(dirs: Sequence[str]) -> None:
    """Fail if directories trained together were prepared incompatibly.

    Mixing preps that disagree on the cheap baseline, or on the isolated-atom
    references the stored energies were shifted by, produces energies that are
    quietly on two different scales -- the model trains, the loss falls, and
    every number afterwards is wrong. Cheaper to refuse here.
    """
    if len(dirs) < 2:
        return
    ref_meta = ref_e0 = None
    ref_dir = None
    for d in dirs:
        meta = _read_json(os.path.join(d, "metadata.json"))
        e0s = _read_json(os.path.join(d, "e0s.json"))
        keys = ("baseline_model", "baseline_basis", "baseline_xc", "basis")
        meta = None if meta is None else {k: meta.get(k) for k in keys}
        if ref_meta is None and meta is not None:
            ref_meta, ref_dir = meta, d
        elif meta is not None and meta != ref_meta:
            raise SystemExit(
                f"ERROR: {d} and {ref_dir} were prepared differently and cannot be "
                f"trained together:\n  {ref_dir}: {ref_meta}\n  {d}: {meta}")
        if e0s is not None:
            if ref_e0 is None:
                ref_e0 = e0s
            elif not _e0s_close(e0s, ref_e0):
                raise SystemExit(
                    f"ERROR: {d} and {ref_dir} use different isolated-atom "
                    "references (e0s.json), so their stored energies are on "
                    "different scales. Re-prepare them with a shared --e0s.")


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _e0s_close(a, b, tol: float = 1e-6) -> bool:
    ka, kb = set(a), set(b)
    if ka != kb:
        return False
    return all(abs(float(a[k]) - float(b[k])) <= tol for k in ka)


# ── batch balancing ─────────────────────────────────────────────────────

def auto_hf_per_batch(n_labeled: int, n_total: int, batch_size: int,
                      target: int = AUTO_HF_TARGET) -> int:
    """How many labeled frames to force into each batch; 0 to leave it alone.

    Shuffling already delivers ``batch_size * density`` labeled frames per batch
    on average, so balancing only earns its keep once that falls below the
    target. The count is capped at half the batch: past that the batch is mostly
    the small labeled set, which it then revisits fast enough to overfit.
    """
    if n_labeled <= 0 or n_labeled >= n_total:
        return 0
    natural = batch_size * (n_labeled / n_total)
    if natural >= target:
        return 0
    return max(1, min(target, batch_size // 2, n_labeled))


class LabelBalancedBatchSampler(Sampler):
    """Batches holding a fixed number of labeled frames, the rest unlabeled.

    With labels on a small fraction of frames, plain shuffling makes the count
    per batch Poisson around a mean below one, so most steps see no expensive
    data and the rest see one or two frames. Drawing the two groups separately
    fixes the count, which makes the high-fidelity term's effective batch size a
    setting rather than an accident.

    The two streams are consumed at different rates -- that is the point -- so
    the labeled one wraps around repeatedly within a single pass over the
    unlabeled one. An epoch is one pass over the UNLABELED frames, since those
    are the majority and the ones that define the data volume.

    ``sizes`` is optional and, when given, applies the same pooled size sort as
    ``BucketedBatchSampler``: both streams are sorted within a pool before being
    cut up, so a batch draws its labeled and its unlabeled frames from the same
    part of the size range and the TB backend keeps most of its padding savings.
    The label count is unaffected, since the sort only decides which frames land
    together, never how many of each kind.
    """

    def __init__(self, labeled: Sequence[bool], batch_size: int,
                 n_labeled_per_batch: int, sizes: Optional[Sequence[int]] = None,
                 pool_batches: int = 8, seed: int = 0):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 0 < n_labeled_per_batch < batch_size:
            raise ValueError(
                f"n_labeled_per_batch must be in (0, {batch_size}), "
                f"got {n_labeled_per_batch}")
        self.lab_idx = [i for i, f in enumerate(labeled) if f]
        self.unlab_idx = [i for i, f in enumerate(labeled) if not f]
        if not self.lab_idx or not self.unlab_idx:
            raise ValueError(
                "LabelBalancedBatchSampler needs both labeled and unlabeled "
                f"frames, got {len(self.lab_idx)} and {len(self.unlab_idx)}")
        self.batch_size = int(batch_size)
        self.n_lab = int(n_labeled_per_batch)
        self.n_unlab = self.batch_size - self.n_lab
        self.sizes = None if sizes is None else list(sizes)
        self.pool = max(1, int(pool_batches))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.unlab_idx) // self.n_unlab

    def _shuffled(self, idx: List[int], g) -> List[int]:
        perm = torch.randperm(len(idx), generator=g).tolist()
        return [idx[i] for i in perm]

    def _draw(self, pool: List[int], cursor: int, k: int, g) -> Tuple[List[int], List[int], int]:
        """Take ``k`` indices from ``pool`` at ``cursor``, reshuffling on wrap."""
        out = []
        while k > 0:
            if cursor >= len(pool):
                pool = self._shuffled(pool, g)
                cursor = 0
            take = min(k, len(pool) - cursor)
            out.extend(pool[cursor:cursor + take])
            cursor += take
            k -= take
        return out, pool, cursor

    def _batches(self) -> List[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        lab_pool = self._shuffled(self.lab_idx, g)
        unlab_pool = self._shuffled(self.unlab_idx, g)
        lab_cur = unlab_cur = 0

        batches: List[List[int]] = []
        remaining = len(self)
        while remaining > 0:
            k = min(self.pool, remaining)
            lab, lab_pool, lab_cur = self._draw(
                lab_pool, lab_cur, k * self.n_lab, g)
            unlab, unlab_pool, unlab_cur = self._draw(
                unlab_pool, unlab_cur, k * self.n_unlab, g)
            if self.sizes is not None:
                lab.sort(key=lambda i: self.sizes[i])
                unlab.sort(key=lambda i: self.sizes[i])
            for b in range(k):
                batches.append(lab[b * self.n_lab:(b + 1) * self.n_lab]
                               + unlab[b * self.n_unlab:(b + 1) * self.n_unlab])
            remaining -= k

        if self.sizes is not None:
            # Pools come out in increasing size order, which would correlate
            # consecutive steps; shuffle the batches themselves the way
            # BucketedBatchSampler does. Membership stays size-grouped.
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1          # so a plain training loop still varies by epoch
        return iter(batches)
