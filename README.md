# Lemon MLX Engine

C++ inference engine for large language models, built on [MLX](https://github.com/ml-explore/mlx).

Run LLMs locally on **Apple M-series**, **AMD GPUs** (Linux/Windows), and CPU — no Python required.

## Features

- **46 LLM architectures** — Llama, Qwen, Gemma, Phi, DeepSeek, Mistral, and more
- **12 VLM architectures** — Qwen-VL, PaliGemma, Pixtral, Gemma3, and more
- **Embedders** — BERT, Nomic-BERT, Qwen3-Embed
- **Quantized inference** — 4-bit/8-bit via `quantized_matmul`
- **HuggingFace integration** — auto-downloads models, tokenizers, and chat templates
- **Streaming generation** — async token pipeline with KV caching
- **Chat templates** — Jinja2-compatible (minja), auto-loaded from model config

## Requirements

- CMake 3.20+
- C++17 compiler
- libcurl
- Rust toolchain (for tokenizers-cpp)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Usage

```bash
# Interactive chat (downloads model on first run)
./chat --model mlx-community/Qwen3-1.7B-4bit
```

## Libraries

| Library | Description |
|---------|-------------|
| `mlx-lm-core` | MLX module wrappers |
| `mlx-lm-common` | Tokenizer, generation, KV cache, hub API |
| `mlx-lm-llm` | LLM model implementations |
| `mlx-lm-vlm` | Vision-language model implementations |
| `mlx-lm-embedders` | Embedding model implementations |

## License

MIT
