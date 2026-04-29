# Qwen3.5 0.8B 4-bit vs hipfire

- prompt_arg_md5: `e4624962cbee3304616ca432239258a8`
- prompt_file_md5: `6a9046339c4d28146b2e7cc66eb1f691`
- prompt_arg_file: `/home/reckon/projects/lemon-mlx-engine/.codeinsight+research/qwen35_0p8b_vs_hipfire_20260429T193927Z/prompt_arg.txt`
- lemon_bench_md5: `1ea3b388e333ff017a7687d73a0c3e26`
- hipfire_ref_md5: `d757a8ae3471175983ef088d99b7f258`

| engine | model | ok/runs | median decode tok/s | avg decode tokens | output hashes | token hashes |
|---|---|---:|---:|---:|---:|---:|
| lemon | mlx-community/Qwen3.5-0.8B-MLX-4bit | 3/3 | 125.86 | 56.00 | 1 | 1 |
| hipfire | qwen3.5:0.8b | 3/3 | 229.5 | 56.00 | 1 | 1 |

Lemon median decode speed is 0.548x hipfire (125.86 / 229.50 tok/s).

## Artifact Audit

| label | kind | format | primary md5 | size bytes | quant |
|---|---|---|---|---:|---|
| source | hf_or_mlx | safetensors | `7d61ccb37ef5d4ad330fec4ad0e06934` | 1746942600 | `none` |
| lemon | hf_or_mlx | safetensors | `439d1796c1a3ce24ef83681eabd656db` | 625229487 | `{"bits": 4, "group_size": 64, "mode": "affine"}` |
| hipfire | hipfire | mq4 | `0769fdeaa08e82bd6ed555e3f151d04b` | 549221504 | `{"bits": 4, "group_size": 256, "mode": "magnumquant_fwht_rotated", "runtime_dtype": "MQ4G256"}` |

exact_artifact_match: `0`

These runs are same-source-family, not same exact artifact: hipfire uses MQ4G256/FWHT-rotated weights while MLX uses its native quantized safetensors format.
