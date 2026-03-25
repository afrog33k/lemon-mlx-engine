// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/types.h>
#include <mlx/mlx.h>
#include <vector>

namespace mlx_lm {

// Default prepare step for LLM models.
// Evaluates the prompt in chunks until there is a small number
// of tokens left to feed into the TokenIterator.
//
// This is a free function that any LLM model can call from its
// prepare_impl.
template <typename Model>
PrepareResult llm_default_prepare(
    Model& model,
    const LMInput& input,
    std::vector<KVCache>& cache,
    int window_size)
{
    int prefill_step_size = (window_size > 0) ? window_size : 512;
    auto text = input.text;

    // Prepare the prompt in chunks if larger than the prefill size.
    // Tokens are 1D [seq_len]; add batch dim [1, seq_len] for model calls.
    while (text.tokens.shape(0) > prefill_step_size) {
        auto chunk_tokens = mlx::core::slice(
            text.tokens,
            {0},
            {prefill_step_size});

        // Add batch dimension for model call (matches Swift's newAxis)
        LMInput::Text chunk_text(mlx::core::expand_dims(chunk_tokens, 0));
        model(chunk_text, &cache, nullptr);

        // Eval the actual cache state arrays so the GPU materializes
        // the forward pass. Matches Python's mx.eval([c.state for c in cache]).
        // Without this, the computation graph keeps growing across chunks.
        {
            std::vector<mlx::core::array> to_eval;
            for (auto& c : cache) {
                auto s = c.state();
                to_eval.insert(to_eval.end(), s.begin(), s.end());
            }
            mlx::core::eval(to_eval);
        }
        mlx::core::clear_cache();

        text.tokens = mlx::core::slice(
            text.tokens,
            {prefill_step_size},
            {text.tokens.shape(0)});
    }

    return PrepareResult::tokens(std::move(text));
}

} // namespace mlx_lm
