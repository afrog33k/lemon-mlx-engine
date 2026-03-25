// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx/mlx.h>
#include <optional>
#include <string>
#include <vector>

namespace mlx_lm {

// ---------------------------------------------------------------------------
// AttentionMask — type-safe wrapper matching Python's {None, "causal", array}.
//
// Python's create_attention_mask returns:
//   - None          when sequence length == 1 (single-token generation)
//   - "causal"      when sequence length > 1 (prefill, no window)
//   - array         when using windowed/rotating attention
//
// The "causal" string enables MLX's optimized causal SDPA Metal kernel,
// which is significantly faster than passing a full boolean mask array.
// ---------------------------------------------------------------------------
class AttentionMask {
public:
    AttentionMask() = default;

    // Named constructors.
    static AttentionMask none() { return {}; }

    static AttentionMask causal() {
        AttentionMask m;
        m.is_causal_ = true;
        return m;
    }

    static AttentionMask from_array(mlx::core::array arr) {
        AttentionMask m;
        m.array_ = std::move(arr);
        return m;
    }

    bool is_none() const { return !is_causal_ && !array_.has_value(); }
    bool is_causal() const { return is_causal_; }
    bool has_array() const { return array_.has_value(); }
    const mlx::core::array& as_array() const { return array_.value(); }

private:
    bool is_causal_ = false;
    std::optional<mlx::core::array> array_;
};

// ---------------------------------------------------------------------------
// sdpa — Scaled dot-product attention dispatcher.
//
// Calls mx::fast::scaled_dot_product_attention with the correct mask_mode:
//   - "causal"  → optimized causal kernel (no mask array needed)
//   - ""        → no mask (single-token generation)
//   - "" + arr  → explicit array mask (windowed/rotating attention)
// ---------------------------------------------------------------------------
inline mlx::core::array sdpa(
    const mlx::core::array& queries,
    const mlx::core::array& keys,
    const mlx::core::array& values,
    float scale,
    const AttentionMask& mask)
{
    if (mask.is_causal()) {
        return mlx::core::fast::scaled_dot_product_attention(
            queries, keys, values, scale, "causal");
    }
    if (mask.has_array()) {
        return mlx::core::fast::scaled_dot_product_attention(
            queries, keys, values, scale, "", mask.as_array());
    }
    return mlx::core::fast::scaled_dot_product_attention(
        queries, keys, values, scale, "");
}

// ---------------------------------------------------------------------------
// create_attention_mask — returns an AttentionMask.
//
// For t==1: returns none (no mask needed).
// For t>1 without window: returns causal (optimized kernel).
// For t>1 with window:    returns array mask.
// ---------------------------------------------------------------------------
AttentionMask create_attention_mask(
    const mlx::core::array& h,
    const KVCache* cache = nullptr,
    std::optional<int> window_size = std::nullopt);

// Scaled dot-product attention with KV cache update.
mlx::core::array attention_with_cache_update(
    const mlx::core::array& queries,
    const mlx::core::array& keys,
    const mlx::core::array& values,
    KVCache* cache,
    float scale,
    const AttentionMask& mask = AttentionMask::none());

} // namespace mlx_lm
