#!/bin/bash
# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
# Fetch the rMD17 per-molecule NPZ files and the split index CSVs from FigShare
# (DOI 10.6084/m9.figshare.12672038).
#
# The record also ships a 1 GB rmd17.tar.bz2 holding the same arrays; the
# per-molecule files are pulled instead so a single failed transfer only costs
# one molecule. File IDs are pinned because FigShare mints a new one on every
# revision and an unpinned name lookup would silently drift.
set -euo pipefail

ORBACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DEST=${1:-${DATA_ROOT:-$ORBACE/bench_data}/rmd17/raw}
mkdir -p "$DEST"

declare -A FILES=(
  [rmd17_aspirin.npz]=62265757
  [rmd17_azobenzene.npz]=62265754
  [rmd17_benzene.npz]=62265739
  [rmd17_ethanol.npz]=62265733
  [rmd17_malonaldehyde.npz]=62265736
  [rmd17_naphthalene.npz]=62265751
  [rmd17_paracetamol.npz]=62265760
  [rmd17_salicylic.npz]=62265748
  [rmd17_toluene.npz]=62265742
  [rmd17_uracil.npz]=62265745
  [index_train_01.csv]=62265793
  [index_test_01.csv]=62265781
)

fail=0
for name in "${!FILES[@]}"; do
  out="$DEST/$name"
  if [[ -s "$out" ]]; then
    echo "have    $name"
    continue
  fi
  echo "fetch   $name"
  if curl -sSL --retry 5 --retry-delay 5 --retry-all-errors \
       -o "$out.part" "https://ndownloader.figshare.com/files/${FILES[$name]}"; then
    mv "$out.part" "$out"
  else
    echo "FAILED  $name" >&2
    rm -f "$out.part"
    fail=1
  fi
done

echo
echo "--- verifying NPZ archives are readable ---"
python - "$DEST" <<'PY'
import glob, os, sys
import numpy as np

bad = 0
for path in sorted(glob.glob(os.path.join(sys.argv[1], "rmd17_*.npz"))):
    try:
        with np.load(path) as d:
            n = len(d["energies"])
            natm = len(d["nuclear_charges"])
        print(f"  ok   {os.path.basename(path):28s} {n:6d} frames  {natm:3d} atoms")
    except Exception as exc:
        print(f"  BAD  {os.path.basename(path):28s} {exc}")
        bad += 1
sys.exit(1 if bad else 0)
PY

exit $fail
