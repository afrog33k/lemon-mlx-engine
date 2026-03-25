// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx/mlx.h>

namespace mlx_lm {

// ---------------------------------------------------------------------------
// WiredLimitGuard — RAII guard that upgrades GPU wired memory to
// max_recommended_working_set_size for the duration of generation.
//
// Matches Python mlx-lm's `with wired_limit(model)` context manager in
// generate.py. Without this, GPU buffers are not kept resident and must be
// paged in on each access, causing significant performance degradation.
//
// On destruction, synchronizes all GPU work and restores the previous limit.
// ---------------------------------------------------------------------------
class WiredLimitGuard {
public:
    WiredLimitGuard() : old_limit_(0), active_(false) {
        if (!mlx::core::gpu::is_available()) return;
        auto& info = mlx::core::gpu::device_info(0);
        auto it = info.find("max_recommended_working_set_size");
        if (it == info.end()) return;
        size_t max_rec_size = std::get<size_t>(it->second);
        if (max_rec_size == 0) return;
        old_limit_ = mlx::core::set_wired_limit(max_rec_size);
        active_ = true;
    }

    ~WiredLimitGuard() {
        if (active_) {
            mlx::core::synchronize();
            mlx::core::set_wired_limit(old_limit_);
        }
    }

    // Non-copyable, non-movable.
    WiredLimitGuard(const WiredLimitGuard&) = delete;
    WiredLimitGuard& operator=(const WiredLimitGuard&) = delete;

private:
    size_t old_limit_;
    bool active_;
};

} // namespace mlx_lm
