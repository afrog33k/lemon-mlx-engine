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

# Apply group_size=256 support directly (more reliable than patch file)
# This adds gs256 type alias, case 256 to switch, and tiled path support
python3 "$root_dir/patches/apply_qmv_group256.py" "$root_dir"

# Apply Rdna35 TILE_N=8 optimization (validated on Strix Halo)
python3 "$root_dir/patches/apply_rdna35_tile_n.py" "$root_dir"
