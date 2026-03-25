// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of SSM.swift — Mamba2 SSM step functions (pure MLX ops, no Metal kernel)

#include <mlx-lm/common/ssm_utils.h>
#include <cmath>

namespace mx = mlx::core;

namespace mlx_lm {

mx::array compute_dt(const mx::array& dt, const mx::array& dt_bias,
                      float time_step_min, float time_step_max) {
    auto result = mx::log(mx::add(mx::exp(mx::add(dt, dt_bias)), mx::array(1.0f))); // softplus
    return mx::clip(result, mx::array(time_step_min), mx::array(time_step_max));
}

mx::array segsum(const mx::array& x, const std::optional<mx::array>& mask) {
    int l = x.shape(-1);
    mx::array x_val = x;

    if (mask.has_value()) {
        auto expanded_mask = mx::expand_dims(mask.value(), 1);
        x_val = mx::multiply(x_val, expanded_mask);
    }

    // Repeat along new axis and take lower triangular
    x_val = mx::repeat(mx::expand_dims(x_val, -1), l, -1);
    x_val = mx::tril(x_val, -1);
    auto x_segsum = mx::cumsum(x_val, -2);

    if (mask.has_value()) {
        auto m1 = mx::expand_dims(mask.value(), -1);
        auto m2 = mx::expand_dims(mask.value(), -2);
        auto combined = mx::multiply(m1, m2);
        x_segsum = mx::where(combined, x_segsum, mx::array(-std::numeric_limits<float>::infinity()));
    }

    return x_segsum;
}

std::pair<mx::array, mx::array> ssm_attn(
    const mx::array& x,
    const mx::array& A_log,
    const mx::array& B_ssm,
    const mx::array& C_ssm,
    const mx::array& D,
    const mx::array& dt,
    const mx::array& dt_bias,
    const std::optional<mx::array>& state,
    float time_step_min,
    float time_step_max,
    const std::optional<mx::array>& mask)
{
    int b = x.shape(0), l = x.shape(1), h = x.shape(2), dh = x.shape(3);
    int g = B_ssm.shape(2), d = B_ssm.shape(3);

    auto dt_val = compute_dt(dt, dt_bias, time_step_min, time_step_max);
    int repeats = h / g;
    auto A = mx::negative(mx::exp(A_log));

    // B: [b, l, g, d] → [b, g, d, l]
    auto B_t = mx::transpose(B_ssm, {0, 2, 3, 1});

    // CB = C^T @ B → [b, g, l, l], then repeat for heads
    auto C_t = mx::swapaxes(C_ssm, 1, 2); // [b, g, l, d]
    auto CB = mx::matmul(C_t, B_t);       // [b, g, l, l]
    CB = mx::repeat(CB, repeats, 1);       // [b, h, l, l]

    // dtA = dt * A → [b, l, h]
    auto dtA = mx::multiply(dt_val, mx::reshape(A, {1, 1, -1}));
    // decay = exp(segsum(dtA swapped)) → [b, h, l, l]
    auto decay = mx::exp(segsum(mx::swapaxes(dtA, 1, 2), mask));

    // Surrogate attention: tril(CB * decay)
    auto surr_attn = mx::tril(mx::multiply(CB, decay), 0);

    // dtx = dt * x → [b, l, h, dh]
    auto dtx = mx::multiply(mx::reshape(dt_val, {b, l, h, 1}), x);

    // y = surr_attn @ dtx^T → need dtx as [b, h, l, dh]
    auto dtx_t = mx::swapaxes(dtx, 1, 2); // [b, h, l, dh]
    auto y = mx::matmul(surr_attn, dtx_t); // [b, h, l, dh]
    y = mx::swapaxes(y, 1, 2);             // [b, l, h, dh]

    // Compute next state
    auto decay_last = mx::transpose(
        mx::slice(decay, {0, 0, l - 1, 0}, {b, h, l, decay.shape(3)}),
        {0, 3, 1, 2}); // [b, l, h, 1] → rearranged

    auto B_rep = mx::swapaxes(mx::repeat(B_t, repeats, 1), 2, 3); // [b, h, l, d]
    auto dtx_decay = mx::multiply(dtx, decay_last);
    dtx_decay = mx::swapaxes(mx::swapaxes(dtx_decay, 1, 2), 2, 3); // [b, h, dh, l]
    auto next_state = mx::matmul(dtx_decay, B_rep); // [b, h, dh, d]

    if (state.has_value()) {
        auto exp_dtA_cumsum = mx::exp(mx::cumsum(dtA, 1));
        auto last_exp = mx::slice(exp_dtA_cumsum,
            {0, l - 1, 0}, {b, l, h}); // [b, 1, h]
        next_state = mx::add(next_state,
            mx::multiply(mx::reshape(last_exp, {b, h, 1, 1}), state.value()));

        // Add contribution from previous state
        auto state_r = mx::reshape(state.value(), {b, 1, g, repeats, dh, d});
        auto C_r = mx::reshape(C_ssm, {b, l, g, 1, d, 1});
        auto y_prev = mx::squeeze(mx::matmul(state_r, C_r), -1);
        y_prev = mx::reshape(y_prev, {b, l, g * repeats, dh}); // flatten g*repeats → h
        y = mx::add(y, mx::multiply(mx::expand_dims(exp_dtA_cumsum, -1), y_prev));
    }

    // Add D * x
    y = mx::add(y, mx::multiply(mx::reshape(D, {1, 1, h, 1}), x));

    return {y, next_state};
}

std::pair<mx::array, mx::array> ssm_update(
    const mx::array& hidden_states,
    const mx::array& A_log,
    const mx::array& B_ssm,
    const mx::array& C_ssm,
    const mx::array& D,
    const mx::array& dt,
    const mx::array& dt_bias,
    const std::optional<mx::array>& state,
    float time_step_min,
    float time_step_max,
    const std::optional<mx::array>& mask)
{
    // Always use the attention-based path (no Metal kernel in C++ port)
    return ssm_attn(hidden_states, A_log, B_ssm, C_ssm, D, dt, dt_bias,
                     state, time_step_min, time_step_max, mask);
}

} // namespace mlx_lm
