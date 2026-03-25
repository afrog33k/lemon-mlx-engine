// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/generate_params.h>
#include <mlx-lm/common/types.h>
#include <nlohmann/json.hpp>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mlx_lm {

// Forward declarations
class KVCache;

// ModelContext holds the model, tokenizer, and processor together.
// Template-free — uses type-erased function objects for model operations.
struct ModelContext {
    // Model operations (type-erased via std::function to avoid virtuals).
    std::function<PrepareResult(const LMInput&, std::vector<KVCache>&, int)> prepare_fn;
    std::function<LMOutput(const LMInput::Text&, std::vector<KVCache>*, const LMOutput::State*)> call_fn;
    std::function<mlx::core::array(const mlx::core::array&, std::vector<KVCache>*)> forward_fn;
    std::function<std::vector<KVCache>(const GenerateParameters&)> new_cache_fn;
    std::function<std::unordered_map<std::string, mlx::core::array>(
        std::unordered_map<std::string, mlx::core::array>)> sanitize_fn;

    // Tokenizer operations (type-erased).
    std::function<std::vector<int>(const std::string&)> encode_fn;
    std::function<std::string(const std::vector<int>&)> decode_fn;
    std::function<std::vector<int>(const std::vector<std::unordered_map<std::string, std::string>>&)> apply_chat_template_fn;

    // Configuration
    std::string model_id;
    std::optional<std::vector<int>> eos_token_ids;

    // Extra context for chat template rendering (e.g., enable_thinking=false).
    // Shared with the apply_chat_template_fn lambda so mutations propagate.
    std::shared_ptr<nlohmann::json> template_extra_context;

    // Bind a concrete model into this context (non-owning reference).
    template <typename Model>
    static ModelContext from_model(Model& model) {
        ModelContext ctx;
        ctx.prepare_fn = [&model](const LMInput& input, std::vector<KVCache>& cache, int ws) {
            return model.prepare(input, cache, ws);
        };
        ctx.call_fn = [&model](const LMInput::Text& input, std::vector<KVCache>* cache,
                               const LMOutput::State* state) {
            return model(input, cache, state);
        };
        ctx.forward_fn = [&model](const mlx::core::array& inputs, std::vector<KVCache>* cache) {
            return model.forward(inputs, cache);
        };
        ctx.new_cache_fn = [&model](const GenerateParameters& params) {
            return model.new_cache(params);
        };
        ctx.sanitize_fn = [&model](std::unordered_map<std::string, mlx::core::array> w) {
            return model.sanitize(std::move(w));
        };
        return ctx;
    }

    // Bind an owned model via shared_ptr (model lifetime tied to context).
    template <typename Model>
    static ModelContext from_model_owned(std::shared_ptr<Model> model) {
        ModelContext ctx;
        ctx.prepare_fn = [model](const LMInput& input, std::vector<KVCache>& cache, int ws) {
            return model->prepare(input, cache, ws);
        };
        ctx.call_fn = [model](const LMInput::Text& input, std::vector<KVCache>* cache,
                               const LMOutput::State* state) {
            return (*model)(input, cache, state);
        };
        ctx.forward_fn = [model](const mlx::core::array& inputs, std::vector<KVCache>* cache) {
            return model->forward(inputs, cache);
        };
        ctx.new_cache_fn = [model](const GenerateParameters& params) {
            return model->new_cache(params);
        };
        ctx.sanitize_fn = [model](std::unordered_map<std::string, mlx::core::array> w) {
            return model->sanitize(std::move(w));
        };
        return ctx;
    }
};

// Thread-safe container for a ModelContext.
// Replaces Swift's actor-based ModelContainer.
class ModelContainer {
public:
    explicit ModelContainer(ModelContext context)
        : context_(std::make_shared<ModelContext>(std::move(context))) {}

    // Perform an action with exclusive access to the model context.
    template <typename Func>
    auto perform(Func&& action) -> decltype(action(std::declval<ModelContext&>())) {
        std::lock_guard<std::mutex> lock(mutex_);
        return action(*context_);
    }

    // Read-only access.
    template <typename Func>
    auto perform_read(Func&& action) const -> decltype(action(std::declval<const ModelContext&>())) {
        std::lock_guard<std::mutex> lock(mutex_);
        return action(*context_);
    }

    const std::string& model_id() const { return context_->model_id; }

private:
    std::shared_ptr<ModelContext> context_;
    mutable std::mutex mutex_;
};

} // namespace mlx_lm
