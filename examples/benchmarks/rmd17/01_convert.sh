#!/bin/bash
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
# Convert the downloaded rMD17 NPZ files to OrbACE extended XYZ.
#
#   bash 01_convert.sh          (run 00_fetch.sh first)
#
# Writes $DATA_ROOT/rmd17/<molecule>/extxyz/{train,valid,test}.xyz, one
# directory per molecule (default DATA_ROOT: bench_data under the repository
# root).
#
# Uses the shipped index_train_01 / index_test_01 pair and splits the 1000
# training indices 950/50, which is the protocol MACE, NequIP, Allegro and
# NN-xTB all report against. Nothing is drawn at random.
set -euo pipefail

ORBACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DATA=${DATA_ROOT:-$ORBACE/bench_data}/rmd17
export PYTHONPATH=$ORBACE:${PYTHONPATH:-}

python "$ORBACE/examples/benchmarks/rmd17/rmd17_to_extxyz.py" \
    --npz_dir "$DATA/raw" \
    --out_dir "$DATA" \
    --split 01 \
    --n_valid 50
