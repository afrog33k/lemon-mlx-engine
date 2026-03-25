// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of VLMModel.swift
#pragma once

#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/types.h>

namespace mlx_lm {

// VLMModel is a CRTP base that extends LanguageModel with vision capabilities.
// In the Swift code, VLMModel just extends LanguageModel + LoRAModel.
// In C++, this is simply a tag/marker — the real distinction is that VLM models
// implement prepare() to handle multimodal inputs (images/video).
template <typename Derived>
class VLMModel : public LanguageModel<Derived> {
    // VLM models inherit all LanguageModel functionality.
    // The key difference is in their prepare_impl() which handles
    // LMInput with image/video data, not just text tokens.
};

} // namespace mlx_lm
