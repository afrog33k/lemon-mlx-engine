// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/embedders/embedder_factory.h>
#include <mlx-lm/common/base_config.h>
#include <mlx-lm/common/safetensors.h>
#include <mlx-lm/common/hub_api.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mlx_lm {

// --- Helper: create typed embedder models from JSON ---

template <typename Config, typename Model, typename... Args>
static EmbedderModelVariant create_embedder(const std::string& config_json, Args&&... args) {
    auto j = nlohmann::json::parse(config_json);
    Config config = j.get<Config>();
    return Model(config, std::forward<Args>(args)...);
}

// --- EmbedderTypeRegistry ---

EmbedderTypeRegistry::EmbedderTypeRegistry() {
    // Register all supported embedder model types
    creators_["bert"] = [](const std::string& json) {
        return create_embedder<BertConfiguration, BertModel>(json);
    };
    creators_["roberta"] = [](const std::string& json) {
        return create_embedder<BertConfiguration, BertModel>(json);
    };
    creators_["xlm-roberta"] = [](const std::string& json) {
        return create_embedder<BertConfiguration, BertModel>(json);
    };
    creators_["distilbert"] = [](const std::string& json) {
        return create_embedder<BertConfiguration, BertModel>(json);
    };
    creators_["nomic_bert"] = [](const std::string& json) {
        auto j = nlohmann::json::parse(json);
        NomicBertConfiguration config = j.get<NomicBertConfiguration>();
        return EmbedderModelVariant(NomicBertModel(config, false));
    };
    creators_["qwen3"] = [](const std::string& json) {
        return create_embedder<Qwen3EmbedConfiguration, Qwen3EmbedModel>(json);
    };
}

void EmbedderTypeRegistry::register_type(const std::string& model_type, CreatorFn creator) {
    creators_[model_type] = std::move(creator);
}

bool EmbedderTypeRegistry::has_type(const std::string& model_type) const {
    return creators_.count(model_type) > 0;
}

EmbedderModelVariant EmbedderTypeRegistry::create(
    const std::string& model_type, const std::string& config_json) const
{
    auto it = creators_.find(model_type);
    if (it == creators_.end()) {
        throw std::runtime_error("Unsupported embedder model type: " + model_type);
    }
    return it->second(config_json);
}

// --- Singleton ---

EmbedderTypeRegistry& embedder_type_registry() {
    static EmbedderTypeRegistry registry;
    return registry;
}

// --- Load from directory ---

EmbedderContext load_embedder_from_directory(const std::string& model_directory) {
    // Read config.json
    auto config_path = fs::path(model_directory) / "config.json";
    if (!fs::exists(config_path)) {
        throw std::runtime_error("config.json not found in " + model_directory);
    }

    std::ifstream config_file(config_path);
    nlohmann::json config_json;
    config_file >> config_json;

    auto base_config = parse_base_configuration(config_json);

    // Create model from type registry
    auto& registry = embedder_type_registry();
    if (!registry.has_type(base_config.model_type)) {
        throw std::runtime_error("Unsupported embedder model type: " + base_config.model_type);
    }

    auto model = registry.create(base_config.model_type, config_json.dump());

    // Load weights from safetensors
    auto weights = load_safetensors_from_directory(model_directory);

    // Sanitize and load weights into the model
    std::visit([&weights](auto& m) {
        weights = m.sanitize(std::move(weights));
        m.load_weights(weights);
    }, model);

    // Load pooling configuration
    auto pooler = load_pooling(model_directory);

    return EmbedderContext{std::move(model), std::move(pooler), model_directory};
}

// --- Load from HF Hub ---

EmbedderContext load_embedder(
    const std::string& model_id,
    const std::string& cache_dir)
{
    auto& hub = HubApi::shared();
    if (!cache_dir.empty()) {
        hub.set_cache_dir(cache_dir);
    }

    auto model_dir = hub.snapshot_download(model_id);

    auto ctx = load_embedder_from_directory(model_dir);
    ctx.model_id = model_id;

    return ctx;
}

} // namespace mlx_lm
