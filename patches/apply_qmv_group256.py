#!/usr/bin/env python3
import sys

root_dir = sys.argv[1]
# During CMake patching, we're in the MLX source directory
import os
if os.path.exists('mlx/backend/rocm/quantized/qmm.hip'):
    file_path = 'mlx/backend/rocm/quantized/qmm.hip'
else:
    file_path = f"{root_dir}/build-gfx1151/_deps/mlx-src/mlx/backend/rocm/quantized/qmm.hip"

with open(file_path, 'r') as f:
    content = f.read()

# Track if any changes were made
changes = False

# 1. Add gs256 type alias after gs128
if 'using gs256' not in content:
    content = content.replace(
        '    using gs128 = std::integral_constant<int, 128>;\n',
        '    using gs128 = std::integral_constant<int, 128>;\n    using gs256 = std::integral_constant<int, 256>;\n'
    )
    changes = True
    print("Added gs256 type alias")

# 2. Add case 256 to DISPATCH_GROUP_SIZE macro
old_case128 = '      case 128:                                              \\\n        launch_qmv(type_tag, scale_tag, bits_tag, gs128{});  \\\n        break;                                               \\\n      default:'
new_case256 = '      case 128:                                              \\\n        launch_qmv(type_tag, scale_tag, bits_tag, gs128{});  \\\n        break;                                               \\\n      case 256:                                              \\\n        launch_qmv(type_tag, scale_tag, bits_tag, gs256{});  \\\n        break;                                               \\\n      default:'

if new_case256.split('\n')[0] not in content:
    content = content.replace(old_case128, new_case256)
    changes = True
    print("Added case 256 to DISPATCH_GROUP_SIZE macro")

# 3. Add group_size_ == 256 to tiled QMV paths (all dtype/bits combinations)
tiled_patterns = [
    ('LAUNCH_TILED(hip_bfloat16, hip_bfloat16, 4, 32)', 'LAUNCH_TILED(hip_bfloat16, hip_bfloat16, 4, 256)'),
    ('LAUNCH_TILED(hip_bfloat16, hip_bfloat16, 8, 32)', 'LAUNCH_TILED(hip_bfloat16, hip_bfloat16, 8, 256)'),
    ('LAUNCH_TILED(__half, __half, 4, 32)', 'LAUNCH_TILED(__half, __half, 4, 256)'),
    ('LAUNCH_TILED(__half, __half, 8, 32)', 'LAUNCH_TILED(__half, __half, 8, 256)'),
]

for old_pattern, new_launch in tiled_patterns:
    old_line = f'          if (group_size_ == 32)  {{ {old_pattern}; }}\n          else if (group_size_ == 64)'
    new_line = f'          if (group_size_ == 32)  {{ {old_pattern}; }}\n          else if (group_size_ == 256) {{ {new_launch}; }}\n          else if (group_size_ == 64)'
    if new_line.split('\n')[1] not in content:
        content = content.replace(old_line, new_line)
        changes = True
        print(f"Added group256 tiled support for {new_launch}")

if changes:
    with open(file_path, 'w') as f:
        f.write(content)
    print("Applied all group_size=256 changes")
else:
    print("group_size=256 changes already applied")
