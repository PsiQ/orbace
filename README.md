# OrbACE: Equivariant Message-Passing Potentials with Quantum-Chemistry Features

OrbACE builds E(3)-equivariant MACE-style potentials whose node/edge features come
from the **electronic structure** rather than from element identity alone. It ships
two model families that trade accuracy, cost, and force conservativeness:

| Family | CLI | Features | Forces | Cost |
|--------|-----|----------|--------|------|
| **Orbital** | `--method orbital` | Localized-MO features (Fock/Coulomb/exchange integrals, orbital shape invariants, coefficient norms) | `-dE/dR` through orbital centroids | HF SCF **+ localization** per frame |
| **Atom** | `--method atom` | Atom-centered gauge-invariant **tight-binding** (extended-Hückel) features recomputed **in-forward** from geometry | **Conservative** (exact `-dE/dR`) | **No SCF** |

The atom family needs no SCF at all unless you ask for an HF delta baseline,
which makes it dramatically cheaper to prepare at scale.

The orbital family operates directly in localized-MO space and feeds the network:

- **Fock matrix elements** `|F_ij|` between orbital pairs (coupling strength)
- **Coulomb/exchange integrals** `(ii|jj)` and `(ij|ji)` between orbitals
- **Core Hamiltonian** `|h_ij|` in the MO basis (separates 1e from 2e effects)
- **Energy gap** `|ε_i − ε_j|` per edge (perturbation-theory coupling)
- **Orbital shape invariants** from the quadrupole tensor (spread + 3 eigenvalues)
- **MO coefficient norms** per angular-momentum channel
- **Per-orbital energy baselines** (E0 regression)
- **Occupation-weighted gating** and **orbital attention** readout
- **Per-group LayerNorm** on input features, **gradient-detached** scale/shift

The off-diagonal `F_ij` and `h_ij` change sign with the arbitrary phase of either
localized orbital, so the network receives their magnitudes.

## Installation

```bash
pip install -e .
```

This installs MACE as the separate package `mace-torch` (pinned to 0.3.15),
whose interaction, product and readout blocks form the equivariant backbone.
Any other `mace` on the path, such as an editable checkout of a MACE fork, takes
its place, so use an environment where `python -c "import mace; print(mace.__version__)"`
prints `0.3.15`.

## Quick Start

The workflow is always **prepare → train → (finetune) → eval → MD**. `prepare.py`
turns extended-XYZ frames into per-frame `.npz` feature files; everything
downstream reads those.

Input: an extended XYZ file with `REF_energy` and `REF_forces`. If it contains
`config_type=IsolatedAtom` frames, their `REF_energy` values are used as E0 —
matching whatever level of theory produced your labels. Otherwise E0 is computed
via HF on each isolated atom.

### 1. Prepare data

```bash
# Orbital family
python scripts/prepare.py --method orbital --input train.xyz \
    --output_dir data_train --basis sto-3g --localizer pm --num_workers 16

# Atom family -- no SCF is run at all
python scripts/prepare.py --method atom --input train.xyz \
    --output_dir adata_train --basis sto-3g --num_workers 16
```

Prepare each split you need (train / validation / test / finetune). Validation,
test, and finetune splits must **share the training split's E0s**:

```bash
python scripts/prepare.py --method orbital --input valid.xyz \
    --output_dir data_valid --basis sto-3g --localizer pm \
    --e0s_json data_train/e0s.json --skip_meta
```

The network **features** come from **Hartree–Fock at `--basis`** (default STO-3G),
localized with `--localizer` for the orbital family. Keep the basis identical
across preparation, training, and evaluation. This is independent of the optional
delta-learning **baseline** (see below), which may use a different level of theory.

### 2. Train

```bash
python scripts/train.py --method orbital \
    --train_dir data_train --val_dir data_valid \
    --checkpoint_dir ck --baseline_model none --fidelity_method regular \
    --r_max 6.0 --hidden_irreps '64x0e+64x1o+64x2e' \
    --n_epochs 1000 --lr 1e-3 --device auto
```

A YAML config also works:
`python scripts/train.py --method orbital --config configs/default.yaml`.

Because the TB featurizer is bound by kernel-launch latency rather than compute,
the atom model trains at `batch_size` 64 with the learning rate scaled by
`sqrt(64/16)`, which is 2.1x faster per epoch than the reference batch 16.
Passing `--batch_size` explicitly opts out of both the batch and the paired
rate. At a fixed epoch count the larger batch costs a few percent of test MAE, so
prefer the opt-out when you are chasing a final number rather than a wall-clock
budget; see `TB_FAST_BATCH_SIZE` in `scripts/train.py`.

**Resuming.** Every epoch writes `last.pt`, and each new best writes `best.pt`,
into `--checkpoint_dir`. Rerunning the same command with `--restart best.pt` (or
`last.pt`; the name is resolved inside the checkpoint directory) continues from
that epoch with the optimizer, EMA shadow and best-so-far validation scores
restored, so a run longer than one walltime can be chained across jobs. A
resume that lands past `--stage_two_start` re-applies the stage-two loss weights
and learning rate without treating it as a fresh transition, so model selection
is not reset and `best_stage_one_model.pt` is not overwritten.

### 3. Finetune

```bash
python scripts/train.py --method orbital \
    --foundation_model ck/best_model.pt --freeze_backbone true \
    --train_dir data_ft --val_dir data_valid --checkpoint_dir ck_ft \
    --lr 1e-4 --n_epochs 250
```

Finetune splits are usually prepared with `--skip_meta` (no `e0s.json`), so the
isolated-atom E0s are inherited from the foundation checkpoint automatically.

### 4. Evaluate

```bash
python scripts/eval.py --method orbital --model ck/best_model.pt --npz_data data_test
python scripts/eval.py --method atom   --model ck/best_model.pt --npz_data adata_test --r_cut 5.0
```

Large test sets are streamed lazily from disk, so hundreds of thousands of frames
evaluate without exhausting memory.

### 5. Molecular dynamics

```bash
python scripts/md.py --method orbital --model ck/best_model.pt \
    --xyz mol.xyz --ensemble nve --nsteps 1000 --dt 0.5 --temperature 300 \
    --basis sto-3g --localizer pm --r_cut 6.0 --device auto --out_traj md.xyz
```

Or through ASE:

```python
from orbace.calculators import OrbACECalculator

calc = OrbACECalculator(
    model_path="ck/best_model.pt", basis="sto-3g",
    localizer="pm", r_cut=6.0, device="cuda",
)
atoms.calc = calc
```

`device="cuda"` loads the neural network on the GPU; Hartree–Fock in the
calculator still runs with CPU PySCF unless GPU SCF is enabled (see below).

## Delta learning (baseline correction)

OrbACE can train on **residuals** relative to a cheap electronic-structure
baseline: the network learns a smooth correction on top of xTB / GFN / minimal HF
instead of the full absolute energy surface.

**What gets stored.** After E0 subtraction from isolated-atom references,
`prepare.py` optionally subtracts a per-frame baseline energy and forces:

- **Energy label:** `E_stored = E_REF − E_atomic_E0 − E_baseline`
- **Force label:** `F_stored = F_REF − F_baseline`

Each `.npz` also records `baseline_energy` / `baseline_forces`, and
`metadata.json` records which baseline was used so training and deployment stay
consistent. Setting `--baseline_model` to anything other than `none` **is** the
delta-learning switch — there is no separate flag. Two options refine SCF
baselines: `--baseline_basis` (basis for HF/DFT baselines) and `--baseline_xc`
(XC functional for the `dft` baseline).

| Value | Baseline | Basis / functional | Forces vs high-level DFT | Cost |
|-------|----------|--------------------|--------------------------|------|
| `none` | No baseline (absolute-energy training) | — | — | — |
| `xtb`, `gfn2` | GFN2-xTB (`pip install xtb`) | — | good | ~free |
| `gfn1`, `gfn0` | Earlier GFN parameterizations | — | fair | ~free |
| `hf_sto3g` | Minimal-basis HF | STO-3G (fixed) | poor | free (reuses feature SCF) |
| `hf` / `hf_self` | HF at the feature `--basis` | `--basis` (or `--baseline_basis`) | tracks basis | free (reuses feature SCF) |
| `hf_321g`, `hf_631g`, `hf_svp` | HF at a fixed larger basis | 3-21G / 6-31G / def2-SVP | better | extra HF SCF |
| `dft` | KS-DFT (configurable) | `--baseline_basis` (def2-SVP) + `--baseline_xc` (wB97X) | best per cost | one KS SCF |
| `dft_svp`, `wb97x_svp` | wB97X/def2-SVP preset | def2-SVP + wB97X | best per cost | one KS SCF |
| `wb97x_631gs`, `pbe_svp`, `b3lyp_svp` | Other curated DFT presets | as named | good–best | one KS SCF |

> **Cost note.** A delta baseline is recomputed on **every** energy/force
> evaluation (each MD step). `xtb`, `hf_sto3g`, and `hf`/`hf_self` are effectively
> free; `hf_321g`/`hf_svp`/`dft` add a separate SCF per step.

### Fidelity methods

`train.py --fidelity_method` selects how the baseline is used:

| Mode | Prepared data | What the network predicts | Baseline at inference |
|------|---------------|---------------------------|-----------------------|
| `regular` with `--baseline_model none` ("direct") | no baseline | absolute (E0-subtracted) energy | none |
| `regular` with a baseline | residual targets | the residual only | recomputed and added back |
| `mfdh` (multi-fidelity delta head) | residual targets **+** stored baseline | the **total**, as learned base head **+** delta head | **none** — the base head replaces it |
| `mfah` (multi-fidelity absolute head) | residual targets **+** stored baseline | two **independent** absolute energies, one per fidelity | **none** — never used |

Both MF modes fit the **absolute** reference energy/forces rather than the
inflated residual forces, are trained by the same four-term loss, and require
data prepared with a baseline. They differ only in whether the cheap-fidelity
head is part of the high-fidelity sum:

- **`mfdh`** adds it: `E_high = readout + delta + base`. The decomposition is
  physics-informed and the base head is load-bearing, so it runs at inference.
- **`mfah`** does not: the low head is a parallel absolute prediction that never
  enters `E_high`, which therefore stays *identical* to a direct model's. The
  cheap level acts purely as an auxiliary task shaping the shared backbone (the
  delta is implicit), and the low head can be discarded after training. This is
  the multi-head form used in the multi-fidelity literature.

Because `mfah`'s high-fidelity path is bit-identical to `direct` at equal
weights, an `mfah` vs `direct` comparison isolates exactly what the cheap
fidelity contributes as an auxiliary task, with no architectural confound.

> **Evaluating MF models:** point `eval.py` at the **same baseline dir** the
> model was trained on (not the no-baseline `direct` dir). The model never consumes
> the baseline, but the stored `baseline_energy` is needed to rebuild the reference
> on the model's (delta-)E0 convention. Using the no-baseline dir silently shifts
> the reference by the baseline's isolated-atom E0 sum.

### Uneven fidelities: more cheap frames than expensive ones

Neither MF arm requires the two levels to cover the same frames. The usual way
to build such a set is to run the cheap method over everything and the
expensive one over whatever the budget allowed, so most frames carry only
`baseline_energy` / `baseline_forces` and a minority also carry `energy` /
`forces`. `prepare.py` already writes it that way: the reference keys are
stored only for frames whose source had a reference value, so feeding it
geometries with no energy in the extxyz produces cheap-only frames.

Point training at both preps and nothing else changes:

```bash
python scripts/train.py --method atom --fidelity_method mfah \
    --data_dir cheap_prepped,dft_prepped --baseline_model xtb
```

`--data_dir` and `--train_dir` take a comma-separated list, which avoids
merging directories that may hold a million files. The directories must have
been prepared with the same baseline and the same isolated-atom references;
pass `--e0s_json` to the later prep to share them, and training refuses to
start if they disagree rather than training on two energy scales at once.

Every frame supervises the cheap head. The high-fidelity terms are reduced
over the labeled frames alone, so the loss a mixed batch reports is the one
those frames would have produced by themselves.

When labels are sparse, shuffling stops delivering them: at 1% and a batch of
64, over half of all batches contain no expensive frame at all and the
high-fidelity gradient arrives in bursts. Training therefore counts the labels
at startup and fixes the per-batch composition when it has to:

```
Multi-fidelity labels: 10,000 / 1,000,000 training frames (1.00%) carry a high-fidelity energy.
  Batch balancing ON (auto): 8 labeled + 56 cheap per batch of 64.
  The labeled set is revisited ~12.5x per pass over the cheap frames; watch the
  validation curve for overfitting to it.
```

`--mf_hf_per_batch` overrides the count: `auto` (default) decides from the
density and stays out of the way when shuffling already suffices, an integer
fixes it, and `0` disables balancing. Balancing subsumes `--bucket_by_size` on
the atom path, keeping the size grouping while holding the label count exact.

The revisit factor in that report is the one number worth watching. Balancing
draws the small labeled set far more often than the large cheap one, so the
expensive frames are the ones at risk of being overfit; the train/val split is
stratified so the validation metric always has labeled frames to measure.

Note that the cheap frames are cheap to *label*, not to *train on*: OrbACE-A
recomputes its tight-binding descriptors in-forward, so per-epoch cost scales
with the total frame count. OrbACE-O caches descriptors instead, but producing
them needs an SCF per frame, which is what makes very large cheap sets an
OrbACE-A proposition in practice. `examples/ani1x/dataset/extract_cheap_pool.py`
draws such a cheap-only pool from ANI-1x while excluding every frame already in
the labelled splits.

### Inference with a baseline

Training records `baseline_model` / `baseline_basis` / `baseline_xc` on the
checkpoint, and `regular` delta models add the baseline back at inference so
reported energies match your reference convention:

- `eval.py --npz_data ...` — the baseline is already stored per frame, nothing to specify.
- `eval.py --xyz file.xyz` — the baseline is recomputed on the fly; the `auto`
  defaults for `--baseline_model` / `--baseline_basis` / `--baseline_xc` read it from
  the checkpoint. Only override them for old checkpoints predating this metadata.
- `OrbACECalculator(..., baseline_model="auto")` does the same for ASE/MD.

The benchmark guides under `examples/` give the full prepare → train → eval
commands for the direct model and for the `regular`/`mfdh`/`mfah` variants on
both xTB and HF/STO-3G baselines.

## Efficient preparation at scale

Features are **baseline-independent** — only the labels change with the baseline.
Preparing the same split once per baseline therefore recomputes the (expensive)
SCF needlessly. Instead, prepare the features **once** with `--baseline_model none`
and then attach baselines with cheap re-baseline passes:

```bash
# 1) Expensive features ONCE (also stores the HF-self baseline for free)
python scripts/prepare.py --method orbital --input train.xyz \
    --output_dir data_train_direct --basis sto-3g --localizer pm \
    --baseline_model none --num_workers 16

# 2) Cheap re-baseline passes -- features are reused, NO SCF is re-run
python scripts/prepare.py --method orbital --input train.xyz \
    --output_dir data_train_xtb --basis sto-3g --localizer pm \
    --baseline_model xtb --features_from data_train_direct

python scripts/prepare.py --method orbital --input train.xyz \
    --output_dir data_train_hf --basis sto-3g --localizer pm \
    --baseline_model hf_sto3g --features_from data_train_direct
```

- **HF-family baselines** are free *for the orbital family*: its no-baseline pass
  stores `hf_self_energy` / `hf_self_forces` from the *same* SCF that produced the
  features. The atom family runs no SCF, so it stores nothing to re-baseline from
  and an HF delta set has to be prepared directly.
- **xTB/DFT baselines** run their own (cheap) calculation in the re-baseline pass,
  and work with `--features_from` for either family.
- Output is **bit-identical** to preparing that baseline from scratch.

The atom family stores only geometry and targets (the Hückel features are
rebuilt in-forward), so its `none` and `xtb` baselines run **no SCF at all**;
only an HF baseline triggers one.

## Architecture

```
Input: extxyz (geometry, E_ref, F_ref)
  |
  v
Features:
  orbital -> PySCF HF -> localization (Boys / Pipek-Mezey) -> localized MOs
  atom    -> extended Hückel from geometry, differentiable (no SCF)
  |
  v
Graph:
  orbital: nodes = localized MOs (centroids as positions), edges = MO pairs < r_cut
  atom:    nodes = atoms,                                  edges = atom pairs < r_cut
  |
  v
E(3)-Equivariant Message Passing:
  Spherical harmonics + radial basis + gated chemical edges
  -> Tensor product interactions
  -> Equivariant product basis (correlation order 3)
  -> Per-interaction readout
  |
  v
Energy = E0(species) + ScaleShift(sum of readouts)   [+ learned base head if mfdh]
Forces = -dE/dR
```

## Key differences from MACE

| Aspect | MACE | OrbACE |
|--------|------|--------|
| Graph nodes | Atoms | Localized MOs (orbital family) or atoms (atom families) |
| Node features | Z one-hot | Z + orbital/QC features (energies, occupancies, integrals, shape invariants) |
| Edge features | Bessel radial | Bessel + Fock `\|F_ij\|` + `J_ij` + `K_ij` + hcore `\|h_ij\|` + energy gap |
| E0 baseline | Per-atom E0(Z) | Per-orbital-species E0 / per-atom E0 (least squares) |
| Scale/shift | Standard | Gradient-detached |
| Readout | Scatter sum | Orbital attention + occupation-weighted gating + scatter sum |
| Delta learning | — | `regular` residual, or multi-fidelity as `mfdh` (additive base head) / `mfah` (parallel absolute heads) |
| Force derivation | `-dE/dR_atom` | via orbital centroids (orbital), detached QC (atom), or fully differentiable TB |

## Project layout

```
OrbACE/
├── orbace/
│   ├── modules/
│   │   ├── models.py            # OrbACE (orbital-space model)
│   │   ├── atom_model.py        # AtomOrbACE (atom-centered, QC + TB backends)
│   │   ├── invariant_model.py   # invariant feature model
│   │   ├── blocks.py            # readout, scale-shift, ZBL pair repulsion
│   │   └── embeddings.py
│   ├── data/
│   │   ├── orbital_features.py  # PySCF -> localized-MO features
│   │   ├── orbital_graph.py     # orbital graph construction
│   │   ├── invariant_graph.py   # atom graph construction (+ lazy datasets)
│   │   ├── tb_features.py       # differentiable extended-Hückel features
│   │   ├── diff_integrals.py    # differentiable integrals for TB
│   │   ├── dataset.py           # datasets + collation (lazy streaming)
│   │   ├── scf_backend.py       # CPU PySCF / gpu4pyscf SCF
│   │   └── preprocessing.py     # geometry -> graph, baselines, E0s
│   ├── tools/
│   │   ├── train.py             # Trainer (EMA, SWA, stage-two weighting)
│   │   ├── isolated_atoms.py    # E0 computation
│   │   └── timing.py            # step timing instrumentation
│   └── calculators/             # ASE calculators (orbital / atom / invariant)
├── scripts/
│   ├── prepare.py  # extxyz -> .npz   (--method orbital|atom, --features_from)
│   ├── train.py    # training         (--method orbital|atom)
│   ├── eval.py     # evaluation       (--npz_data or --xyz)
│   ├── md.py       # molecular dynamics
│   ├── fd_force_check.py       # finite-difference force check
│   └── check_gpu_scf_stack.py
├── examples/
│   ├── ani1x/                  # ANI-1x: build the splits, then train and evaluate
│   │   └── dataset/            # download, curate and verify the extxyz splits
│   └── benchmarks/             # rMD17 and 3BPA: fetch, convert, train, evaluate
├── configs/default.yaml
└── tests/
```

## Devices: quantum chemistry vs training

By default **all quantum chemistry** (HF, localization, integrals, E0 fallbacks)
uses **CPU PySCF**, while **PyTorch** training/inference uses **CUDA** when you pass
`--device cuda` / `auto`.

### GPU-accelerated SCF (optional)

Opt into **gpu4pyscf** for HF by passing **`--device cuda`** to `prepare.py`.
Otherwise OrbACE keeps SCF on CPU even when a GPU is present.

```bash
python scripts/check_gpu_scf_stack.py   # verify CuPy / gpu4pyscf / driver
```

1. Install **CuPy** and **gpu4pyscf** for your CUDA major version:
   - CUDA 12.x: `pip install cupy-cuda12x gpu4pyscf-cuda12x`
   - CUDA 11.x: `pip install cupy-cuda11x gpu4pyscf-cuda11x`
2. `--device auto` / `--device cpu` both mean **CPU PySCF**; use **`--device cuda`** for gpu4pyscf.
3. A **local gpu4pyscf checkout** on `PYTHONPATH` shadows the pip wheel and often
   breaks (e.g. `libcusolver.so.11`). OrbACE temporarily drops such non-`site-packages`
   paths while probing imports. Force a local/editable tree with
   `ORBACE_KEEP_LOCAL_GPU4PYSCF_PATH=1`.

If GPU SCF is disabled, check the warning from `orbace.data.scf_backend`, or:
`from orbace.data.scf_backend import gpu_unavailable_reason; print(gpu_unavailable_reason())`.

**`ImportError: libcusolver.so.11`** with `cupy-cuda12x` installed usually means the
CUDA toolkit shared libraries are not on the loader path (common on HPC nodes).
Fix with `module load cuda/...` and/or
`export LD_LIBRARY_PATH=<CUDA_ROOT>/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}`,
then verify `python -c "from cupy_backends.cuda.libs import cusolver"`.

If you see **`undefined symbol`** … `cusolver` / `cublas`, libraries from different
CUDA installs are being mixed. Set **`CUDA_HOME`** (or `CUDA_PATH`) to your toolkit
root — OrbACE preloads toolkit `libcublas`/`libcublasLt` with `RTLD_GLOBAL` before
importing CuPy. Disable with `ORBACE_SKIP_CUDA_PRELOAD=1`.

> **Note on density fitting.** At minimal basis (STO-3G) `density_fit()` is *not*
> a speedup — the RI auxiliary overhead exceeds the savings on small 4-center
> integrals and it introduces an approximation error. Density fitting only pays off
> with substantially larger basis sets.

## Citing OrbACE

If you use OrbACE, please cite:

> E. Osaro and D. Rocca, "OrbACE: Orbital and Atomic Cluster Expansion.
> Equivariant Message Passing on Electronic-Structure Graphs."

## Datasets

The example scripts download each benchmark from its original source and do not
redistribute it. Please cite the dataset papers when you use them:

- **ANI-1x**: J. S. Smith, B. Nebgen, N. Lubbers, O. Isayev, A. E. Roitberg,
  "Less is more: Sampling chemical space with active learning,"
  *J. Chem. Phys.* **148**, 241733 (2018).
- **rMD17**: A. S. Christensen, O. A. von Lilienfeld, "On the role of gradients
  for machine learning of molecular energies and forces,"
  *Mach. Learn.: Sci. Technol.* **1**, 045018 (2020).
- **3BPA**: D. P. Kovács *et al.*, "Linear Atomic Cluster Expansion Force Fields
  for Organic Molecules: Beyond RMSE," *J. Chem. Theory Comput.* **17**,
  7696–7711 (2021).

## References

- I. Batatia, D. P. Kovács, G. N. C. Simm, C. Ortner, G. Csányi, "MACE: Higher
  Order Equivariant Message Passing Neural Networks for Fast and Accurate Force
  Fields," NeurIPS (2022).
- Z. Qiao, M. Welborn, A. Anandkumar, F. R. Manby, T. F. Miller III, "OrbNet:
  Deep learning for quantum chemistry using symmetry-adapted atomic-orbital
  features," *J. Chem. Phys.* **153**, 124111 (2020).
- Z. Qiao, A. S. Christensen, M. Welborn, F. R. Manby, A. Anandkumar,
  T. F. Miller III, "Informing geometric deep learning with electronic
  interactions to accelerate quantum chemistry," *Proc. Natl. Acad. Sci. U.S.A.*
  **119**, e2205221119 (2022).
- D. S. King *et al.*, "Cartesian equivariant representations for learning and
  understanding molecular orbitals," *Proc. Natl. Acad. Sci. U.S.A.* **122**,
  e2510235122 (2025).
- C. Qian, V. Vitartas, J. Kermode, R. J. Maurer, "Equivariant Electronic
  Hamiltonian Prediction with Many-Body Message Passing," arXiv:2508.15108
  (2025).
- L. Thiede *et al.*, "Coupled Cluster con MōLe: Molecular Orbital Learning for
  Neural Wavefunctions," arXiv:2602.20232 (2026).

## License

Copyright (c) 2026 PsiQuantum Corp.

OrbACE is distributed under the Academic Software License v1.0 (ASL; see
`LICENSE` and `NOTICE`), which permits academic non-commercial use only;
commercial use requires a separate license from PsiQuantum Corp. OrbACE depends
on MACE, which is MIT-licensed and installed separately as `mace-torch`; the
parts of OrbACE adapted from MACE remain under MIT and are listed in `NOTICE`.

The optional GFN2-xTB baseline imports the separately installed `xtb` Python
package (LGPL-3.0), and ASE is LGPL-2.1+; neither is distributed with OrbACE.
