#!/bin/bash
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
# Normalise the 3BPA release into OrbACE extended XYZ.
#
#   bash 01_convert.sh          (run 00_fetch.sh first)
#
# Tags the free atoms as config_type=IsolatedAtom so the E0 machinery picks them
# up, prepends them to every split so all splits share one energy zero, and
# holds 50 of the 500 training configurations out for validation.
#
# Data is read from and written to $DATA_ROOT/3bpa (default: bench_data/3bpa
# under the repository root).
set -euo pipefail

ORBACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DATA=${DATA_ROOT:-$ORBACE/bench_data}/3bpa
export PYTHONPATH=$ORBACE:${PYTHONPATH:-}

python "$ORBACE/examples/benchmarks/3bpa/3bpa_to_extxyz.py" \
    --src_dir "$DATA/raw" \
    --out_dir "$DATA/extxyz" \
    --n_valid 50
