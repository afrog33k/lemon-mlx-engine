// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/embedders/bert.h>
#include <mlx-lm/embedders/nomic_bert.h>
#include <mlx-lm/embedders/qwen3_embed.h>
#include <mlx-lm/embedders/pooling.h>
#include <mlx/mlx.h>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace mlx_lm {

// Type-erased embedder model using std::variant (no virtual functions).
using EmbedderModelVariant = std::variant<BertModel, NomicBertModel, Qwen3EmbedModel>;

// EmbedderContext holds a loaded embedder model and pooling layer.
// All operations are dispatched via std::visit on the variant.
struct EmbedderContext {
    EmbedderModelVariant model;
    Pooling pooler;
    std::string model_id;

    // Run the embedding model on input tokens.
    EmbeddingModelOutput operator()(
        const mlx::core::array& inputs,
        const std::optional<mlx::core::array>& position_ids = std::nullopt,
        const std::optional<mlx::core::array>& token_type_ids = std::nullopt,
        const std::optional<mlx::core::array>& attention_mask = std::nullopt)
    {
        return std::visit([&](auto& m) {
            return m(inputs, position_ids, token_type_ids, attention_mask);
        }, model);
    }

    // Run pooling on model output.
    mlx::core::array pool(
        const EmbeddingModelOutput& output,
        const std::optional<mlx::core::array>& mask = std::nullopt,
        bool normalize = false,
        bool apply_layer_norm = false) const
    {
        return pooler(output, mask, normalize, apply_layer_norm);
    }

    // Convenience: embed + pool in one call.
    mlx::core::array embed(
        const mlx::core::array& inputs,
        const std::optional<mlx::core::array>& attention_mask = std::nullopt,
        bool normalize = true)
    {
        auto output = (*this)(inputs, std::nullopt, std::nullopt, attention_mask);
        return pool(output, attention_mask, normalize);
    }

    // Get vocabulary size.
    int vocab_size() const {
        return std::visit([](const auto& m) { return m.vocab_size(); }, model);
    }
};

// Embedder type registry — maps model_type strings to factory functions.
class EmbedderTypeRegistry {
public:
    // Factory function: takes JSON config string, returns variant-wrapped model.
    using CreatorFn = std::function<EmbedderModelVariant(const std::string& config_json)>;

    EmbedderTypeRegistry();

    void register_type(const std::string& model_type, CreatorFn creator);
    bool has_type(const std::string& model_type) const;
    EmbedderModelVariant create(const std::string& model_type, const std::string& config_json) const;

private:
    std::unordered_map<std::string, CreatorFn> creators_;
};

// Get the global embedder type registry (singleton).
EmbedderTypeRegistry& embedder_type_registry();

// Load an embedder model from a local directory.
// The directory must contain config.json and *.safetensors files.
// Optionally loads pooling config from 1_Pooling/config.json.
EmbedderContext load_embedder_from_directory(const std::string& model_directory);

// Load an embedder model from a Hugging Face model ID.
// Downloads if not cached locally.
EmbedderContext load_embedder(
    const std::string& model_id,
    const std::string& cache_dir = "");

} // namespace mlx_lm
