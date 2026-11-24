# ANI-1x

The ANI-1x benchmark end to end: turning the public ANI-1x release into labelled
extended-XYZ splits, then preparing, training and evaluating both model
families with and without a cheap baseline.

Nothing here is imported by the `orbace` package; the core code stays
dataset-agnostic. Generated data and checkpoints are not stored in the
repository.

## 1. Build the dataset

Download the upstream release (5.6 GB, md5 `98090dd6679106da861f52bed825ffb7`):

```bash
curl -L -o ani1x-release.h5 https://ndownloader.figshare.com/files/18112775
```

Convert it to extended XYZ at wB97X/6-31G* (`--level dz`). `--require
'ccsd(t)_cbs.energy'` keeps only the conformers that also carry a CCSD(T)*/CBS
energy; `--stride`, `--max_per_molecule` and `--max_frames` thin the pool
further:

```bash
python examples/ani1x/dataset/ani1x_to_extxyz.py \
    --h5 ani1x-release.h5 --out ani_all.extxyz --level dz \
    --require 'ccsd(t)_cbs.energy'
```

Isolated-atom energies are computed at the matching level and written as
`config_type=IsolatedAtom` frames at the top of the file, which is where
`scripts/prepare.py` looks for them.

ANI-1x forces are in Hartree/**Angstrom**, not Hartree/Bohr. Converting them with
the wrong factor inflates them by 1.8897 and no training metric will reveal it,
because the labels stay self-consistent with each other and disagree only with
physics. Always verify:

```bash
python examples/ani1x/dataset/check_force_consistency.py \
    --input ani_all.extxyz --check grad fd
```

At `--level dz` this recomputes the gradient at wB97X/6-31G*, the level of
theory of the labels, so a correct conversion returns a ratio of 1.00.

Carve the splits by farthest-point sampling in geometry-descriptor space. A
random training set drawn from a trajectory-derived pool is redundant and
leaves near-duplicates of training frames in the test set; FPS avoids both.
The paper's splits use 10,000 training and 1,000 validation frames, with the
full remainder of the pool as the test set:

```bash
python examples/ani1x/dataset/fps_select.py --mode geometry \
    --extxyz ani_all.extxyz \
    --n_train 10000 --n_valid 1000 --n_finetune 5000 --seed 42
```

This writes `train.xyz`, `valid.xyz`, `finetune.xyz` and `test.xyz`, each
carrying the isolated-atom frames.

### Other dataset tools

| Script | Purpose |
| --- | --- |
| `split_extxyz.py` | Stream a huge extxyz into fixed-size chunks for parallel preparation |
| `extract_test_xyz.py` | Write the complement of an FPS selection (the held-out frames in `fps_indices.npz`) to a flat extxyz |
| `extract_cheap_pool.py` | Draw unlabelled ANI-1x geometries for the cheap level of a multi-fidelity run, excluding every frame already in the pool by geometry hash |
| `generate_dft.py` | Generate DFT energies/forces for a multi-XYZ file at a chosen functional and basis |

## 2. Prepare features

Validation and test splits inherit the training split's isolated-atom
energies through `--e0s_json`, so all splits share one energy zero. Preparation
runs on CPU; set `--num_workers` to the cores available.

**OrbACE-A** (atom family) stores only geometry and targets, because its
tight-binding features are rebuilt in the forward pass. The direct and xTB sets
run no SCF; the HF/STO-3G set runs one per frame for the baseline:

```bash
for base in none xtb hf_sto3g; do
  tag=${base/hf_sto3g/hf}; tag=${tag/none/direct}
  python scripts/prepare.py --method atom --input train.xyz \
      --output_dir adata_train_$tag --basis sto-3g --device cpu \
      --num_workers 32 --baseline_model $base
  for split in valid test; do
    python scripts/prepare.py --method atom --input $split.xyz \
        --output_dir adata_${split}_$tag --basis sto-3g --device cpu \
        --num_workers 32 --baseline_model $base \
        --e0s_json adata_train_$tag/e0s.json --skip_meta
  done
done
```

**OrbACE-O** (orbital family) runs one HF/STO-3G SCF and Pipek–Mezey
localization per frame. Prepare the direct set once; the xTB and HF/STO-3G sets
then reuse its features through `--features_from`, and the HF baseline comes
from the same SCF:

```bash
python scripts/prepare.py --method orbital --input train.xyz \
    --output_dir data_train_direct --basis sto-3g --localizer pm --device cpu \
    --num_workers 32 --baseline_model none
for split in valid test; do
  python scripts/prepare.py --method orbital --input $split.xyz \
      --output_dir data_${split}_direct --basis sto-3g --localizer pm --device cpu \
      --num_workers 32 --baseline_model none \
      --e0s_json data_train_direct/e0s.json --skip_meta
done

for base in xtb hf_sto3g; do
  tag=${base/hf_sto3g/hf}
  python scripts/prepare.py --method orbital --input train.xyz \
      --output_dir data_train_$tag --basis sto-3g --localizer pm --device cpu \
      --num_workers 32 --baseline_model $base --features_from data_train_direct
  for split in valid test; do
    python scripts/prepare.py --method orbital --input $split.xyz \
        --output_dir data_${split}_$tag --basis sto-3g --localizer pm --device cpu \
        --num_workers 32 --baseline_model $base \
        --features_from data_${split}_direct \
        --e0s_json data_train_$tag/e0s.json --skip_meta
  done
done
```

## 3. Train

Each family runs seven arms that differ only in the prepared set and two flags:

| Arm | Prepared set | `--baseline_model` | `--fidelity_method` |
| --- | --- | --- | --- |
| direct | `_direct` | `none` | `regular` |
| regular xTB | `_xtb` | `xtb` | `regular` |
| mfdh xTB | `_xtb` | `xtb` | `mfdh --mf_hf_forces true` |
| mfah xTB | `_xtb` | `xtb` | `mfah --mf_hf_forces true` |
| regular HF | `_hf` | `hf_sto3g` | `regular` |
| mfdh HF | `_hf` | `hf_sto3g` | `mfdh --mf_hf_forces true` |
| mfah HF | `_hf` | `hf_sto3g` | `mfah --mf_hf_forces true` |

**OrbACE-A**, direct arm (3 interactions, 64 channels, 5 Å cutoff):

```bash
python scripts/train.py --method atom \
    --train_dir adata_train_direct --val_dir adata_valid_direct \
    --checkpoint_dir ck_atom_direct \
    --baseline_model none --fidelity_method regular \
    --basis sto-3g --r_max 5.0 --num_interactions 3 \
    --hidden_irreps '64x0e + 64x1o + 64x2e' --correlation 3 \
    --loss huber --huber_delta 0.01 \
    --energy_weight 200 --forces_weight 100 \
    --stage_two_start 400 --stage_two_energy_weight 500 --stage_two_forces_weight 100 \
    --n_epochs 1000 --batch_size 64 --lr 0.016 --warmup_epochs 15 \
    --clip_grad 10.0 --scheduler plateau --patience 200 \
    --device auto --seed 42 --save_cpu true
```

**OrbACE-O**, direct arm (2 interactions, 64 channels, 6 Å cutoff):

```bash
python scripts/train.py --method orbital \
    --train_dir data_train_direct --val_dir data_valid_direct \
    --checkpoint_dir ck_orb_direct \
    --baseline_model none --fidelity_method regular \
    --r_max 6.0 --num_bessel 8 --max_ell 2 --num_interactions 2 \
    --hidden_channels 64 --hidden_irreps '64x0e+64x1o+64x2e' \
    --mlp_channels 64 --radial_MLP '64,64,64' --correlation 3 \
    --n_scalar_features 10 --n_ao_shell_features 15 --n_coeff_norm_features 5 \
    --use_chem_edge_features true --n_chem_edge_features 6 \
    --use_equivariant_orbital_features true --n_equiv_channels 8 --dropout 0.05 \
    --optimizer adam --lr 0.001 --weight_decay 5e-7 --amsgrad true \
    --batch_size 32 --n_epochs 1000 \
    --energy_weight 1000 --forces_weight 10 \
    --stage_two_start 400 --stage_two_energy_weight 1000 --stage_two_forces_weight 300 \
    --scheduler plateau --scheduler_factor 0.8 --scheduler_patience 20 \
    --warmup_epochs 15 --ema_decay 0.999 --clip_grad 10.0 --patience 200 \
    --swa false --error_table PerAtomMAE --device auto --seed 42 --save_cpu true
```

For the other arms, swap the `_direct` directories for `_xtb` or `_hf` and set
the two flags from the table.

## 4. Evaluate

```bash
python scripts/eval.py --method atom --model ck_atom_direct/best_model.pt \
    --npz_data adata_test_direct --r_cut 5.0 --batch_size 64 --num_workers 4
python scripts/eval.py --method orbital --model ck_orb_direct/best_model.pt \
    --npz_data data_test_direct
```

Evaluate each arm on the test set prepared with the same baseline it was
trained on. For `regular` delta arms the stored baseline is added back, so the
reported errors are against the absolute reference.
