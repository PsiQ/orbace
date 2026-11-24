# rMD17

Revised MD17: ten small organic molecules, 100,000 conformers each, recomputed
at PBE/def2-SVP to remove the noise in the original MD17. The benchmark trains
on 950 frames per molecule, validates on 50 and tests on 1000, using the index
splits shipped with the dataset so the numbers line up with MACE, NequIP,
Allegro and NN-xTB.

Source: FigShare DOI 10.6084/m9.figshare.12672038. Please cite A. S. Christensen
and O. A. von Lilienfeld, *Mach. Learn.: Sci. Technol.* **1**, 045018 (2020).

## 1. Fetch and convert

```bash
bash examples/benchmarks/rmd17/00_fetch.sh     # NPZ files + index CSVs -> bench_data/rmd17/raw
bash examples/benchmarks/rmd17/01_convert.sh   # -> bench_data/rmd17/<molecule>/extxyz/{train,valid,test}.xyz
```

Both scripts write under `bench_data/` in the repository root; set `DATA_ROOT`
to put the data elsewhere. The conversion uses the shipped
`index_train_01`/`index_test_01` pair and splits the 1000 training indices
950/50. Nothing is drawn at random.

## 2. Prepare features

Run from the molecule's directory. Validation and test splits inherit the
training split's isolated-atom energies through `--e0s_json`.

**OrbACE-A**, three baseline sets (the HF/STO-3G set runs one SCF per frame,
the others none):

```bash
cd bench_data/rmd17/aspirin
for base in none xtb hf_sto3g; do
  tag=${base/hf_sto3g/hf}; tag=${tag/none/direct}
  python ../../../scripts/prepare.py --method atom --input extxyz/train.xyz \
      --output_dir atom/train_$tag --basis sto-3g --device cpu \
      --num_workers 16 --baseline_model $base
  for split in valid test; do
    python ../../../scripts/prepare.py --method atom --input extxyz/$split.xyz \
        --output_dir atom/${split}_$tag --basis sto-3g --device cpu \
        --num_workers 16 --baseline_model $base \
        --e0s_json atom/train_$tag/e0s.json --skip_meta
  done
done
```

**OrbACE-O**: prepare the direct set (HF/STO-3G and Pipek–Mezey localization),
then build the xTB and HF/STO-3G sets from its features with `--features_from`.
The HF baseline is taken from the same SCF, so it costs nothing extra:

```bash
python ../../../scripts/prepare.py --method orbital --input extxyz/train.xyz \
    --output_dir orbital/train_direct --basis sto-3g --localizer pm --device cpu \
    --num_workers 16 --baseline_model none
for split in valid test; do
  python ../../../scripts/prepare.py --method orbital --input extxyz/$split.xyz \
      --output_dir orbital/${split}_direct --basis sto-3g --localizer pm --device cpu \
      --num_workers 16 --baseline_model none \
      --e0s_json orbital/train_direct/e0s.json --skip_meta
done
for base in xtb hf_sto3g; do
  tag=${base/hf_sto3g/hf}
  python ../../../scripts/prepare.py --method orbital --input extxyz/train.xyz \
      --output_dir orbital/train_$tag --basis sto-3g --localizer pm --device cpu \
      --num_workers 16 --baseline_model $base --features_from orbital/train_direct
  for split in valid test; do
    python ../../../scripts/prepare.py --method orbital --input extxyz/$split.xyz \
        --output_dir orbital/${split}_$tag --basis sto-3g --localizer pm --device cpu \
        --num_workers 16 --baseline_model $base \
        --features_from orbital/${split}_direct \
        --e0s_json orbital/train_$tag/e0s.json --skip_meta
  done
done
```

The prepare steps run on CPU on purpose: these molecules are far too small to
fill a GPU, and `prepare.py` drops to a single worker when the SCF device is
CUDA.

## 3. Train and evaluate

Both families use the same encoder shape, 4 interactions and 150 channels, the
same objective and optimizer, and a 12,000-epoch cap with early stopping on the
validation set. Only the cutoff differs: 5.0 Å over nuclei for OrbACE-A and
6.0 Å over orbital centroids for OrbACE-O, since the two are not the same
quantity.

**OrbACE-A**, direct arm:

```bash
python ../../../scripts/train.py --method atom \
    --train_dir atom/train_direct --val_dir atom/valid_direct \
    --checkpoint_dir ck_atom_direct \
    --baseline_model none --fidelity_method regular \
    --basis sto-3g --r_max 5.0 --num_interactions 4 \
    --hidden_irreps "150x0e+150x1o+150x2e" --correlation 3 \
    --loss mse --energy_weight 200 --forces_weight 1000 --stage_two_start 0 \
    --n_epochs 12000 --batch_size 32 --lr 0.001 --warmup_epochs 15 \
    --optimizer adamw --weight_decay 1e-4 \
    --scheduler plateau --scheduler_factor 0.5 --scheduler_patience 10 \
    --clip_grad 10.0 --patience 100 --seed 42 --device auto --save_cpu true

python ../../../scripts/eval.py --method atom --batch_size 64 --num_workers 4 \
    --model ck_atom_direct/best_model.pt --npz_data atom/test_direct
```

**OrbACE-O**, direct arm:

```bash
python ../../../scripts/train.py --method orbital \
    --train_dir orbital/train_direct --val_dir orbital/valid_direct \
    --checkpoint_dir ck_orb_direct \
    --baseline_model none --fidelity_method regular \
    --r_max 6.0 --max_ell 2 --num_bessel 8 --num_interactions 4 \
    --hidden_channels 150 --hidden_irreps "150x0e+150x1o+150x2e" \
    --mlp_channels 150 --radial_MLP "150,150,150" --correlation 3 \
    --loss mse --energy_weight 200 --forces_weight 1000 --stage_two_start 0 \
    --n_epochs 12000 --batch_size 32 --lr 0.001 --warmup_epochs 15 \
    --optimizer adamw --weight_decay 1e-4 \
    --scheduler plateau --scheduler_factor 0.5 --scheduler_patience 10 \
    --clip_grad 10.0 --patience 100 --seed 42 --device auto --save_cpu true

python ../../../scripts/eval.py --method orbital --batch_size 64 \
    --model ck_orb_direct/best_model.pt --npz_data orbital/test_direct
```

## Arms

Each family runs the same seven arms, differing only in the prepared set and
two flags:

| Arm | Prepared set | `--baseline_model` | `--fidelity_method` | Baseline at inference |
| --- | --- | --- | --- | --- |
| direct | `_direct` | `none` | `regular` | none |
| regular xTB | `_xtb` | `xtb` | `regular` | GFN2-xTB |
| mfdh xTB | `_xtb` | `xtb` | `mfdh` | none |
| mfah xTB | `_xtb` | `xtb` | `mfah` | none |
| regular HF | `_hf` | `hf_sto3g` | `regular` | HF/STO-3G |
| mfdh HF | `_hf` | `hf_sto3g` | `mfdh` | none |
| mfah HF | `_hf` | `hf_sto3g` | `mfah` | none |

Regular delta learns the residual and adds the baseline back at inference. The
multi-fidelity arms fit the absolute reference and call nothing external at
inference: `mfdh` adds a learned base head to the high-fidelity sum, `mfah`
keeps a parallel cheap-level head as an auxiliary task.

Force MAE is reported per Cartesian component, as in the references above.
