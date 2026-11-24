#!/bin/bash
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
# Download the 3BPA release from the BOTNet dataset repository.
#
#   bash 00_fetch.sh
#
# Five files: the free-atom energies, the 300 K training set, and the three
# test sets. They are already extended XYZ in eV, so 01_convert.sh only has to
# tag the isolated atoms and carve out a validation split.
set -euo pipefail

ORBACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DEST=${1:-${DATA_ROOT:-$ORBACE/bench_data}/3bpa/raw}
BASE=https://raw.githubusercontent.com/davkovacs/BOTNet-datasets/main/dataset_3BPA

mkdir -p "$DEST"
cd "$DEST"

for f in iso_atoms train_300K test_300K test_600K test_1200K; do
    if [[ -s $f.xyz ]]; then
        echo "have    $f.xyz"
        continue
    fi
    echo "fetch   $f.xyz"
    curl -sSL --retry 5 --retry-delay 5 --retry-all-errors -O "$BASE/$f.xyz"
done

echo
for f in iso_atoms train_300K test_300K test_600K test_1200K; do
    printf "  %-16s %8d lines\n" "$f.xyz" "$(wc -l < $f.xyz)"
done
