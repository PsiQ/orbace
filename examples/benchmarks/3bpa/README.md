# 3BPA

3-(benzyloxy)pyridin-2-amine, a 27-atom flexible drug-like molecule. Training
uses configurations sampled at 300 K; testing uses separate sets at 300 K,
600 K and 1200 K. The elevated-temperature sets lie well outside the training
ensemble, so they measure extrapolation rather than fit quality.

Source: https://github.com/davkovacs/BOTNet-datasets. Please cite D. P. Kovács
*et al.*, *J. Chem. Theory Comput.* **17**, 7696–7711 (2021).

## 1. Fetch and convert

```bash
bash examples/benchmarks/3bpa/00_fetch.sh     # -> bench_data/3bpa/raw
bash examples/benchmarks/3bpa/01_convert.sh   # -> bench_data/3bpa/extxyz/{train,valid,test_300K,test_600K,test_1200K}.xyz
```

Both scripts write under `bench_data/` in the repository root; set `DATA_ROOT`
to put the data elsewhere. The converter tags the free atoms
`config_type=IsolatedAtom`, prepends them to every split so all splits share
one energy zero, and holds 50 of the 500 training configurations out for
validation.

## 2. Prepare features

OrbACE-A, three baseline sets, run from `bench_data/3bpa`. The HF/STO-3G set
runs one SCF per frame and is the long pole; the others run none:

```bash
cd bench_data/3bpa
for base in none xtb hf_sto3g; do
  tag=${base/hf_sto3g/hf}; tag=${tag/none/direct}
  python ../../scripts/prepare.py --method atom --input extxyz/train.xyz \
      --output_dir atom/train_$tag --basis sto-3g --device cpu \
      --num_workers 16 --baseline_model $base
  for split in valid test_300K test_600K test_1200K; do
    python ../../scripts/prepare.py --method atom --input extxyz/$split.xyz \
        --output_dir atom/${split}_$tag --basis sto-3g --device cpu \
        --num_workers 16 --baseline_model $base \
        --e0s_json atom/train_$tag/e0s.json --skip_meta
  done
done
```

## 3. Train and evaluate

Direct arm, 4 interactions and 150 channels, 12,000-epoch cap with early
stopping on the validation set:

```bash
python ../../scripts/train.py --method atom \
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

for split in test_300K test_600K test_1200K; do
  python ../../scripts/eval.py --method atom --batch_size 64 --num_workers 4 \
      --model ck_atom_direct/best_model.pt --npz_data atom/${split}_direct
done
```

The other arms swap the `_direct` sets for `_xtb` or `_hf` and set
`--baseline_model` (`xtb` or `hf_sto3g`) and `--fidelity_method` (`regular`,
`mfdh` or `mfah`), as tabulated in the rMD17 guide.
