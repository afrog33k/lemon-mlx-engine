// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx/mlx.h>

namespace mlx_lm {

namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// create_attention_mask
// ---------------------------------------------------------------------------
// Matches Python's create_attention_mask(h, cache, window_size).
// Returns:
//   - AttentionMask::none()    when t == 1 (single-token decode)
//   - AttentionMask::causal()  when t > 1 and no window (optimized SDPA kernel)
//   - AttentionMask::from_array() when t > 1 with window or rotating cache

AttentionMask create_attention_mask(
    const mx::array& h,
    const KVCache* cache,
    std::optional<int> window_size)
{
    int t = h.shape(1);  // sequence length

    if (t <= 1) {
        return AttentionMask::none();
    }

    int offset = 0;
    std::optional<int> max_size = std::nullopt;
    if (cache) {
        offset = cache->offset();
        max_size = cache->max_size();
    }

    // If we have a window size from the cache (RotatingKVCache), use it
    // but prefer the explicit window_size parameter if provided
    if (!window_size.has_value() && max_size.has_value()) {
        // For rotating caches, clamp the offset to max_size - 1
        offset = std::min(max_size.value() - 1, offset);
    }

    // Use optimized "causal" string mode when no windowing is needed.
    // This matches Python's behavior: return "causal" for simple prefill.
    bool needs_window = window_size.has_value() || max_size.has_value();
    if (!needs_window && offset == 0) {
        return AttentionMask::causal();
    }

    // For windowed attention or with offset, we need the full mask array.
    return AttentionMask::from_array(create_causal_mask(t, offset, window_size));
}

// ---------------------------------------------------------------------------
// attention_with_cache_update
// ---------------------------------------------------------------------------

mx::array attention_with_cache_update(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& values,
    KVCache* cache,
    float scale,
    const AttentionMask& mask)
{
    mx::array k = keys;
    mx::array v = values;

    if (cache) {
        auto [updated_k, updated_v] = cache->update(keys, values);
        k = updated_k;
        v = updated_v;
    }

    return sdpa(queries, k, v, scale, mask);
}

} // namespace mlx_lm
