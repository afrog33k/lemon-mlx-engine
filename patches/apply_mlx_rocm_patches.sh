#!/usr/bin/env sh
set -eu

root_dir="$1"

for patch_file in \
  "$root_dir/patches/mlx-rocm-gfx1151-disable-qmv-tiled.patch" \
  "$root_dir/patches/mlx-rocm-gfx1151-qmv-cols64.patch" \
  "$root_dir/patches/mlx-rocm-hipblaslt-debug-gate.patch"
do
  if git apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    continue
  fi
  git apply "$patch_file"
done
