// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of NemotronH.swift

#include <mlx-lm/llm/models/nemotron_h.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/ssm_utils.h>
#include <mlx-lm/common/activations.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Helpers ---

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// Squared ReLU — now uses compiled relu_squared() from activations.h

// --- Block type parsing ---

NemotronHBlockType parse_block_type(char c) {
    switch (c) {
        case 'M': return NemotronHBlockType::Mamba;
        case '*': return NemotronHBlockType::Attention;
        case '-': return NemotronHBlockType::MLP;
        case 'E': return NemotronHBlockType::MoE;
        default: throw std::runtime_error(std::string("Unknown NemotronH block type: ") + c);
    }
}

// --- Configuration ---

void from_json(const nlohmann::json& j, NemotronHConfiguration& c) {
    auto get = [&](const char* key, auto& dst, auto def) {
        if (j.contains(key) && !j[key].is_null()) dst = j[key].get<std::decay_t<decltype(dst)>>();
        else dst = def;
    };
    auto get_bool = [&](const char* key, bool& dst, bool def) {
        if (j.contains(key)) dst = j[key].get<bool>(); else dst = def;
    };

    get("vocab_size", c.vocab_size, 0);
    get("hidden_size", c.hidden_size, 0);
    get("num_hidden_layers", c.num_hidden_layers, 0);
    get("layer_norm_epsilon", c.layer_norm_epsilon, 1e-5f);
    get("rope_theta", c.rope_theta, 10000.0f);
    get("num_attention_heads", c.num_attention_heads, 0);
    get("num_key_value_heads", c.num_key_value_heads, 0);
    get_bool("attention_bias", c.attention_bias, false);
    get_bool("tie_word_embeddings", c.tie_word_embeddings, false);
    get("mamba_num_heads", c.mamba_num_heads, 0);
    get("mamba_head_dim", c.mamba_head_dim, 0);
    get("ssm_state_size", c.ssm_state_size, 0);
    get("conv_kernel", c.conv_kernel, 0);
    get("n_groups", c.n_groups, 1);
    get_bool("mamba_proj_bias", c.mamba_proj_bias, false);
    get_bool("use_conv_bias", c.use_conv_bias, true);
    get("intermediate_size", c.intermediate_size, 0);
    get_bool("mlp_bias", c.mlp_bias, false);
    get("n_routed_experts", c.n_routed_experts, 0);
    get("num_experts_per_tok", c.num_experts_per_tok, 0);
    get("moe_intermediate_size", c.moe_intermediate_size, 0);
    get("moe_shared_expert_intermediate_size", c.moe_shared_expert_intermediate_size, 0);
    get("n_group", c.n_group, 1);
    get("topk_group", c.topk_group, 1);
    get_bool("norm_topk_prob", c.norm_topk_prob, true);
    get("routed_scaling_factor", c.routed_scaling_factor, 1.0f);

    // head_dim (optional)
    if (j.contains("head_dim") && !j["head_dim"].is_null())
        c.head_dim = j["head_dim"].get<int>();

    // n_shared_experts (optional)
    if (j.contains("n_shared_experts") && !j["n_shared_experts"].is_null())
        c.n_shared_experts = j["n_shared_experts"].get<int>();

    // hybrid_override_pattern: can be string or array of strings
    if (j.contains("hybrid_override_pattern")) {
        auto& hp = j["hybrid_override_pattern"];
        if (hp.is_string()) {
            c.hybrid_override_pattern = hp.get<std::string>();
        } else if (hp.is_array()) {
            std::string joined;
            for (auto& elem : hp) joined += elem.get<std::string>();
            c.hybrid_override_pattern = joined;
        }
    }

    // time_step_limit: can be [min, max] array or separate fields
    if (j.contains("time_step_limit") && j["time_step_limit"].is_array()) {
        auto limits = j["time_step_limit"].get<std::vector<float>>();
        c.time_step_limit_min = limits[0];
        c.time_step_limit_max = limits.size() > 1 ? limits[1] : limits[0];
    } else {
        get("time_step_limit_min", c.time_step_limit_min, 0.0f);
        if (j.contains("time_step_limit_max") && !j["time_step_limit_max"].is_null()) {
            c.time_step_limit_max = j["time_step_limit_max"].get<float>();
        } else {
            c.time_step_limit_max = std::numeric_limits<float>::infinity();
        }
    }
}

// --- NemotronHRMSNormGated ---

NemotronHRMSNormGated::NemotronHRMSNormGated(int dims, float eps, int group_size)
    : weight_(mx::ones({dims})), eps_(eps), group_size_(group_size)
{}

mx::array NemotronHRMSNormGated::operator()(const mx::array& x,
                                              const std::optional<mx::array>& gate) {
    auto states = x;
    if (gate.has_value()) {
        auto g = gate.value();
        // silu(gate) * states
        states = swiglu(g, states);
    }

    // Unflatten: [..., hidden] -> [..., nGroups, groupSize]
    const auto& shape = states.shape();
    mx::Shape new_shape(shape.begin(), shape.end() - 1);
    new_shape.push_back(-1);
    new_shape.push_back(group_size_);
    auto unflattened = mx::reshape(states, new_shape);

    // Per-group RMS norm without learned weight (use identity)
    auto identity_weight = mx::ones({group_size_});
    auto normed = mx::fast::rms_norm(unflattened, identity_weight, eps_);

    // Flatten back to [..., hidden] and scale by learned weight
    auto flattened = mx::reshape(normed, shape);
    return mx::multiply(weight_, flattened);
}

std::unordered_map<std::string, mx::array*> NemotronHRMSNormGated::weight_map() {
    return {{"weight", &weight_}};
}

// --- NemotronHMamba2Mixer ---

static int compute_nemotron_conv_dim(const NemotronHConfiguration& config) {
    int intermediate = config.mamba_num_heads * config.mamba_head_dim;
    return intermediate + 2 * config.n_groups * config.ssm_state_size;
}

static int compute_nemotron_intermediate(const NemotronHConfiguration& config) {
    return config.mamba_num_heads * config.mamba_head_dim;
}

static int compute_nemotron_group_size(const NemotronHConfiguration& config) {
    return compute_nemotron_intermediate(config) / config.n_groups;
}

NemotronHMamba2Mixer::NemotronHMamba2Mixer(const NemotronHConfiguration& config)
    : num_heads_(config.mamba_num_heads),
      hidden_size_(config.hidden_size),
      ssm_state_size_(config.ssm_state_size),
      conv_kernel_size_(config.conv_kernel),
      intermediate_size_(compute_nemotron_intermediate(config)),
      num_groups_(config.n_groups),
      head_dim_(config.mamba_head_dim),
      conv_dim_(compute_nemotron_conv_dim(config)),
      heads_per_group_(config.mamba_num_heads / config.n_groups),
      time_step_limit_min_(config.time_step_limit_min),
      time_step_limit_max_(config.time_step_limit_max),
      in_proj_weight_(mx::zeros({compute_nemotron_intermediate(config) + compute_nemotron_conv_dim(config) + config.mamba_num_heads, config.hidden_size})),
      conv1d_weight_(mx::zeros({compute_nemotron_conv_dim(config), config.conv_kernel, 1})),
      out_proj_weight_(mx::zeros({config.hidden_size, compute_nemotron_intermediate(config)})),
      dt_bias_(mx::ones({config.mamba_num_heads})),
      A_log_(mx::zeros({config.mamba_num_heads})),
      D_(mx::ones({config.mamba_num_heads})),
      norm_(compute_nemotron_intermediate(config), config.layer_norm_epsilon, compute_nemotron_group_size(config))
{
    if (config.mamba_proj_bias) {
        in_proj_bias_ = mx::zeros({intermediate_size_ + conv_dim_ + num_heads_});
        out_proj_bias_ = mx::zeros({hidden_size_});
    }
    if (config.use_conv_bias) {
        conv1d_bias_ = mx::zeros({conv_dim_});
    }
}

mx::array NemotronHMamba2Mixer::apply_conv(const mx::array& conv_input,
                                             const std::optional<mx::array>& mask,
                                             MambaCache* mc) {
    auto input = conv_input;

    // Apply mask if present
    if (mask.has_value()) {
        auto expanded_mask = mx::expand_dims(mask.value(), -1);
        input = mx::where(expanded_mask, input, mx::zeros_like(input));
    }

    int batch = input.shape(0);
    auto dtype = input.dtype();

    // Get or create conv state
    mx::array conv_state = (mc && (*mc)[0].has_value())
        ? (*mc)[0].value()
        : (conv_kernel_size_ > 1
            ? mx::zeros({batch, conv_kernel_size_ - 1, conv_dim_}, dtype)
            : mx::zeros({batch, 0, conv_dim_}, dtype));

    auto padded = mx::concatenate({conv_state, input}, 1);

    // Update cache
    if (mc) {
        int end = padded.shape(1);
        int start = std::max(0, end - (conv_kernel_size_ - 1));
        (*mc)[0] = mx::slice(padded, {0, start, 0}, {padded.shape(0), end, padded.shape(2)});
    }

    // Depthwise conv1d (groups = conv_dim)
    auto conv_out = mx::conv1d(padded, conv1d_weight_, 1, 0, 1, conv_dim_);
    if (conv1d_bias_.has_value()) conv_out = mx::add(conv_out, conv1d_bias_.value());

    // silu activation
    return silu(conv_out);
}

mx::array NemotronHMamba2Mixer::operator()(const mx::array& x,
                                             const AttentionMask& /*attn_mask*/,
                                             const std::optional<mx::array>& ssm_mask,
                                             KVCache* cache) {
    auto* mc = cache ? cache->as_mamba() : nullptr;

    auto projected = linear_fwd(x, in_proj_weight_, in_proj_bias_.has_value() ? &in_proj_bias_.value() : nullptr);

    // Split: gate | conv_input | dt
    auto gate = mx::slice(projected, {0, 0, 0},
                          {projected.shape(0), projected.shape(1), intermediate_size_});
    auto conv_input = mx::slice(projected, {0, 0, intermediate_size_},
                                {projected.shape(0), projected.shape(1), intermediate_size_ + conv_dim_});
    auto dt = mx::slice(projected, {0, 0, intermediate_size_ + conv_dim_},
                        {projected.shape(0), projected.shape(1), projected.shape(2)});

    auto conv_output = apply_conv(conv_input, ssm_mask, mc);

    // Split conv output: hidden | B | C
    auto hidden = mx::slice(conv_output, {0, 0, 0},
                            {conv_output.shape(0), conv_output.shape(1), intermediate_size_});
    auto B_ssm = mx::slice(conv_output, {0, 0, intermediate_size_},
                           {conv_output.shape(0), conv_output.shape(1), intermediate_size_ + num_groups_ * ssm_state_size_});
    auto C_ssm = mx::slice(conv_output, {0, 0, intermediate_size_ + num_groups_ * ssm_state_size_},
                           {conv_output.shape(0), conv_output.shape(1), conv_output.shape(2)});

    // Reshape for SSM
    int b = hidden.shape(0), l = hidden.shape(1);
    hidden = mx::reshape(hidden, {b, l, num_heads_, head_dim_});
    B_ssm = mx::reshape(B_ssm, {b, l, num_groups_, ssm_state_size_});
    C_ssm = mx::reshape(C_ssm, {b, l, num_groups_, ssm_state_size_});
    auto dt_reshaped = mx::reshape(dt, {b, l, num_heads_});

    // Get previous SSM state
    std::optional<mx::array> prev_state;
    if (mc && (*mc)[1].has_value()) prev_state = (*mc)[1].value();

    auto [y, new_state] = ssm_update(
        hidden, A_log_, B_ssm, C_ssm, D_, dt_reshaped, dt_bias_,
        prev_state, time_step_limit_min_, time_step_limit_max_, ssm_mask);

    // Update SSM state in cache
    if (mc) (*mc)[1] = new_state;

    // Flatten y back: [B, L, H, Dh] -> [B, L, H*Dh]
    auto y_flat = mx::reshape(y, {b, l, intermediate_size_});

    // Apply gated norm and output projection
    auto normed = norm_(y_flat, gate);
    return linear_fwd(normed, out_proj_weight_, out_proj_bias_.has_value() ? &out_proj_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> NemotronHMamba2Mixer::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"in_proj.weight", &in_proj_weight_},
        {"conv1d.weight", &conv1d_weight_},
        {"out_proj.weight", &out_proj_weight_},
        {"dt_bias", &dt_bias_},
        {"A_log", &A_log_},
        {"D", &D_},
    };
    if (in_proj_bias_.has_value()) map["in_proj.bias"] = &in_proj_bias_.value();
    if (conv1d_bias_.has_value()) map["conv1d.bias"] = &conv1d_bias_.value();
    if (out_proj_bias_.has_value()) map["out_proj.bias"] = &out_proj_bias_.value();
    for (auto& [k, v] : norm_.weight_map()) map["norm." + k] = v;
    return map;
}

// --- NemotronHAttention ---

static int compute_nemotron_head_dim(const NemotronHConfiguration& config) {
    return config.head_dim.value_or(config.hidden_size / config.num_attention_heads);
}

NemotronHAttention::NemotronHAttention(const NemotronHConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(compute_nemotron_head_dim(config)),
      scale_(std::pow(static_cast<float>(compute_nemotron_head_dim(config)), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * compute_nemotron_head_dim(config), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * compute_nemotron_head_dim(config), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * compute_nemotron_head_dim(config), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * compute_nemotron_head_dim(config)}))
{
    if (config.attention_bias) {
        wq_bias_ = mx::zeros({config.num_attention_heads * head_dim_});
        wk_bias_ = mx::zeros({config.num_key_value_heads * head_dim_});
        wv_bias_ = mx::zeros({config.num_key_value_heads * head_dim_});
        wo_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array NemotronHAttention::operator()(const mx::array& x,
                                           const AttentionMask& attn_mask,
                                           const std::optional<mx::array>& /*ssm_mask*/,
                                           KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, wq_bias_.has_value() ? &wq_bias_.value() : nullptr);
    auto keys = linear_fwd(x, wk_weight_, wk_bias_.has_value() ? &wk_bias_.value() : nullptr);
    auto values = linear_fwd(x, wv_weight_, wv_bias_.has_value() ? &wv_bias_.value() : nullptr);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    // NOTE: NemotronH attention does NOT use RoPE

    // Update cache and get full keys/values
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, attn_mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, wo_bias_.has_value() ? &wo_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> NemotronHAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_}, {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_}, {"o_proj.weight", &wo_weight_},
    };
    if (wq_bias_.has_value()) {
        map["q_proj.bias"] = &wq_bias_.value();
        map["k_proj.bias"] = &wk_bias_.value();
    }
    if (wv_bias_.has_value()) {
        map["v_proj.bias"] = &wv_bias_.value();
        map["o_proj.bias"] = &wo_bias_.value();
    }
    return map;
}

// --- NemotronHMLP ---

NemotronHMLP::NemotronHMLP(const NemotronHConfiguration& config, int intermediate_size_override)
    : up_proj_weight_(mx::zeros({intermediate_size_override > 0 ? intermediate_size_override : config.intermediate_size, config.hidden_size})),
      down_proj_weight_(mx::zeros({config.hidden_size, intermediate_size_override > 0 ? intermediate_size_override : config.intermediate_size}))
{
    int intermediate = intermediate_size_override > 0 ? intermediate_size_override : config.intermediate_size;
    if (config.mlp_bias) {
        up_proj_bias_ = mx::zeros({intermediate});
        down_proj_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array NemotronHMLP::operator()(const mx::array& x) {
    auto up = linear_fwd(x, up_proj_weight_, up_proj_bias_.has_value() ? &up_proj_bias_.value() : nullptr);
    auto activated = relu_squared(up);
    return linear_fwd(activated, down_proj_weight_, down_proj_bias_.has_value() ? &down_proj_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> NemotronHMLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"up_proj.weight", &up_proj_weight_},
        {"down_proj.weight", &down_proj_weight_},
    };
    if (up_proj_bias_.has_value()) map["up_proj.bias"] = &up_proj_bias_.value();
    if (down_proj_bias_.has_value()) map["down_proj.bias"] = &down_proj_bias_.value();
    return map;
}

// --- NemotronHMoEGate ---

NemotronHMoEGate::NemotronHMoEGate(const NemotronHConfiguration& config)
    : top_k_(config.num_experts_per_tok),
      n_group_(config.n_group),
      topk_group_(config.topk_group),
      routed_scaling_factor_(config.routed_scaling_factor),
      norm_topk_prob_(config.norm_topk_prob),
      n_routed_experts_(config.n_routed_experts),
      weight_(mx::zeros({config.n_routed_experts, config.hidden_size})),
      e_score_correction_bias_(mx::zeros({config.n_routed_experts}))
{}

std::pair<mx::array, mx::array> NemotronHMoEGate::operator()(const mx::array& x) {
    int B = x.shape(0), S = x.shape(1);

    auto gates = mx::matmul(x, mx::transpose(weight_));
    auto orig_scores = mx::sigmoid(mx::astype(gates, mx::float32));
    auto scores = mx::add(orig_scores, e_score_correction_bias_);

    // Group-based selection if n_group > 1
    if (n_group_ > 1) {
        int experts_per_group = n_routed_experts_ / n_group_;
        auto group_scores = mx::reshape(scores, {B, S, n_group_, -1});

        // Get top-2 per group, sum for group scoring
        auto sorted_gs = mx::sort(group_scores, -1);
        auto top2_vals = mx::slice(sorted_gs,
            {0, 0, 0, sorted_gs.shape(-1) - 2},
            {B, S, n_group_, sorted_gs.shape(-1)});
        auto group_top = mx::sum(top2_vals, -1, true);

        // Keep only top topk_group groups
        int k_drop = n_group_ - topk_group_;
        auto neg_group_top = mx::negative(group_top);
        auto group_idx = mx::argpartition(neg_group_top, k_drop - 1, -2);
        group_idx = mx::slice(group_idx, {0, 0, 0, 0}, {B, S, k_drop, 1});
        group_idx = mx::broadcast_to(group_idx, {B, S, k_drop, experts_per_group});

        // Zero out scores from non-selected groups
        // Scatter zeros at dropped group indices
        auto zeroed = mx::zeros_like(group_idx);
        // put_along_axis equivalent: create a full group_scores, scatter zeros
        // We manually zero out by using put_along_axis pattern
        // For simplicity, use the argpartition approach with score masking
        auto flat_scores = mx::reshape(scores, {B, S, n_group_, experts_per_group});

        // Create mask: start with ones, then zero out dropped groups
        auto mask = mx::ones({B, S, n_group_, 1});
        auto drop_idx = mx::slice(mx::argpartition(neg_group_top, k_drop - 1, -2),
                                   {0, 0, 0, 0}, {B, S, k_drop, 1});
        // Use scatter to put zeros at dropped positions
        auto zeros_to_scatter = mx::zeros({B, S, k_drop, 1});
        mask = mx::put_along_axis(mask, drop_idx, zeros_to_scatter, -2);
        mask = mx::broadcast_to(mask, {B, S, n_group_, experts_per_group});
        flat_scores = mx::multiply(flat_scores, mask);

        scores = mx::reshape(flat_scores, {B, S, n_routed_experts_});
    }

    // Get top-k experts
    auto neg_scores = mx::negative(scores);
    auto inds = mx::argpartition(neg_scores, top_k_ - 1, -1);
    inds = mx::slice(inds, {0, 0, 0}, {B, S, top_k_});
    auto final_scores = mx::take_along_axis(orig_scores, inds, -1);

    // Normalize if needed
    if (top_k_ > 1 && norm_topk_prob_) {
        auto denom = mx::add(mx::sum(final_scores, -1, true), mx::array(1e-20f));
        final_scores = mx::divide(final_scores, denom);
    }

    // Apply scaling factor
    final_scores = mx::multiply(final_scores, mx::array(routed_scaling_factor_));

    return {inds, final_scores};
}

std::unordered_map<std::string, mx::array*> NemotronHMoEGate::weight_map() {
    return {
        {"weight", &weight_},
        {"e_score_correction_bias", &e_score_correction_bias_},
    };
}

// --- NemotronHSwitchMLP ---

NemotronHSwitchMLP::NemotronHSwitchMLP(int input_dims, int hidden_dims, int num_experts)
    : fc1_(input_dims, hidden_dims, num_experts, false),
      fc2_(hidden_dims, input_dims, num_experts, false)
{}

mx::array NemotronHSwitchMLP::operator()(const mx::array& x, const mx::array& indices) {
    // Expand dims for gather_mm: add [-2, -3]
    auto x_expanded = mx::expand_dims(mx::expand_dims(x, -2), -3);

    bool do_sort = (indices.size() > 64);

    mx::array work_x = x_expanded;
    mx::array idx = indices;
    mx::array inverse_order(0.0f);

    if (do_sort) {
        auto [sx, si, io] = gather_sort(x_expanded, indices);
        work_x = sx;
        idx = si;
        inverse_order = io;
    }

    auto y = fc1_(work_x, idx, do_sort);
    y = relu_squared(y);  // squared relu, not silu!
    y = fc2_(y, idx, do_sort);

    if (do_sort) {
        auto shape = indices.shape();
        y = scatter_unsort(y, inverse_order, &shape);
    }

    return mx::squeeze(y, -2);
}

std::unordered_map<std::string, mx::array*> NemotronHSwitchMLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : fc1_.weight_map()) map["fc1." + k] = v;
    for (auto& [k, v] : fc2_.weight_map()) map["fc2." + k] = v;
    return map;
}

// --- NemotronHMoE ---

NemotronHMoE::NemotronHMoE(const NemotronHConfiguration& config)
    : num_experts_per_tok_(config.num_experts_per_tok),
      gate_(config),
      switch_mlp_(config.hidden_size, config.moe_intermediate_size, config.n_routed_experts)
{
    if (config.n_shared_experts.has_value() && config.n_shared_experts.value() > 0) {
        shared_experts_.emplace(config, config.moe_shared_expert_intermediate_size);
    }
}

mx::array NemotronHMoE::operator()(const mx::array& x) {
    auto [inds, scores] = gate_(x);
    auto y = switch_mlp_(x, inds);
    // Weighted sum: (y * scores[..., newaxis]).sum(axis=-2)
    y = mx::sum(mx::multiply(y, mx::expand_dims(scores, -1)), -2);
    y = mx::astype(y, x.dtype());

    if (shared_experts_.has_value()) {
        y = mx::add(y, (*shared_experts_)(x));
    }

    return y;
}

std::unordered_map<std::string, mx::array*> NemotronHMoE::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : gate_.weight_map()) map["gate." + k] = v;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    if (shared_experts_.has_value()) {
        for (auto& [k, v] : shared_experts_->weight_map()) map["shared_experts." + k] = v;
    }
    return map;
}

// --- NemotronHBlock ---

NemotronHBlock::NemotronHBlock(const NemotronHConfiguration& config, char block_char)
    : block_type_(parse_block_type(block_char)),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.layer_norm_epsilon)
{
    switch (block_type_) {
        case NemotronHBlockType::Mamba:
            mamba_mixer_.emplace(config);
            break;
        case NemotronHBlockType::Attention:
            attention_.emplace(config);
            break;
        case NemotronHBlockType::MLP:
            mlp_.emplace(config);
            break;
        case NemotronHBlockType::MoE:
            moe_.emplace(config);
            break;
    }
}

mx::array NemotronHBlock::operator()(const mx::array& x,
                                       const AttentionMask& attn_mask,
                                       const std::optional<mx::array>& ssm_mask,
                                       KVCache* cache) {
    auto hidden = mx::fast::rms_norm(x, norm_weight_, norm_eps_);

    mx::array output(0.0f);
    switch (block_type_) {
        case NemotronHBlockType::Mamba:
            output = (*mamba_mixer_)(hidden, attn_mask, ssm_mask, cache);
            break;
        case NemotronHBlockType::Attention:
            output = (*attention_)(hidden, attn_mask, ssm_mask, cache);
            break;
        case NemotronHBlockType::MLP:
            output = (*mlp_)(hidden);
            break;
        case NemotronHBlockType::MoE:
            output = (*moe_)(hidden);
            break;
    }

    return mx::add(x, output);
}

std::unordered_map<std::string, mx::array*> NemotronHBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["norm.weight"] = &norm_weight_;
    switch (block_type_) {
        case NemotronHBlockType::Mamba:
            for (auto& [k, v] : mamba_mixer_->weight_map()) map["mixer." + k] = v;
            break;
        case NemotronHBlockType::Attention:
            for (auto& [k, v] : attention_->weight_map()) map["mixer." + k] = v;
            break;
        case NemotronHBlockType::MLP:
            for (auto& [k, v] : mlp_->weight_map()) map["mixer." + k] = v;
            break;
        case NemotronHBlockType::MoE:
            for (auto& [k, v] : moe_->weight_map()) map["mixer." + k] = v;
            break;
    }
    return map;
}

// --- NemotronHBackbone ---

NemotronHBackbone::NemotronHBackbone(const NemotronHConfiguration& config)
    : embeddings_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_f_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.layer_norm_epsilon),
      pattern_(config.hybrid_override_pattern)
{
    layers_.reserve(static_cast<int>(pattern_.size()));
    for (char c : pattern_) {
        layers_.emplace_back(config, c);
    }

    // Calculate first_attention_cache_index:
    // Count Mamba layers ('M') before the first Attention layer ('*')
    {
        int mamba_count = 0;
        for (char c : pattern_) {
            if (c == '*') {
                first_attention_cache_index_ = mamba_count;
                break;
            } else if (c == 'M') {
                mamba_count++;
            }
        }
    }

    // Calculate first_mamba_cache_index:
    // Count Attention layers ('*') before the first Mamba layer ('M')
    {
        int attn_count = 0;
        for (char c : pattern_) {
            if (c == 'M') {
                first_mamba_cache_index_ = attn_count;
                break;
            } else if (c == '*') {
                attn_count++;
            }
        }
    }
}

mx::array NemotronHBackbone::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embeddings_weight_, inputs, 0);

    // Create attention mask using the first attention layer's cache
    AttentionMask attn_mask;
    if (first_attention_cache_index_.has_value() && cache &&
        first_attention_cache_index_.value() < static_cast<int>(cache->size())) {
        attn_mask = create_attention_mask(h, &(*cache)[first_attention_cache_index_.value()]);
    } else if (!cache) {
        // No cache, create mask from sequence length only
        attn_mask = create_attention_mask(h, nullptr);
    }

    // Create SSM mask using the first Mamba layer's cache
    // In C++, the MambaCache does not have a makeMask method,
    // so this is typically nullopt (used for left-padding scenarios only)
    std::optional<mx::array> ssm_mask;

    // Iterate layers, maintaining a separate cache counter
    int cache_counter = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = nullptr;
        auto bt = layers_[i].block_type();
        if (bt == NemotronHBlockType::Mamba || bt == NemotronHBlockType::Attention) {
            if (cache && cache_counter < static_cast<int>(cache->size())) {
                lc = &(*cache)[cache_counter];
            }
            cache_counter++;
        }
        // MLP and MoE blocks get no cache

        h = layers_[i](h, attn_mask, ssm_mask, lc);
    }

    return mx::fast::rms_norm(h, norm_f_weight_, norm_eps_);
}

std::unordered_map<std::string, mx::array*> NemotronHBackbone::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embeddings.weight"] = &embeddings_weight_;
    map["norm_f.weight"] = &norm_f_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- NemotronHModel ---

NemotronHModel::NemotronHModel(const NemotronHConfiguration& config)
    : config_(config),
      backbone_(config)
{
    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult NemotronHModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput NemotronHModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array NemotronHModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = backbone_(inputs, cache);
    if (lm_head_weight_.has_value()) {
        return linear_fwd(out, lm_head_weight_.value());
    } else {
        // tie_word_embeddings: use embeddings weight as lm_head
        return linear_fwd(out, backbone_.embeddings_weight());
    }
}

std::vector<KVCache> NemotronHModel::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    for (char c : config_.hybrid_override_pattern) {
        auto bt = parse_block_type(c);
        switch (bt) {
            case NemotronHBlockType::Mamba:
                caches.emplace_back(MambaCache());
                break;
            case NemotronHBlockType::Attention:
                if (params.max_kv_size.has_value()) {
                    caches.emplace_back(RotatingKVCache(params.max_kv_size.value(), 4));
                } else {
                    caches.emplace_back(KVCacheSimple{});
                }
                break;
            case NemotronHBlockType::MLP:
            case NemotronHBlockType::MoE:
                // No cache for MLP/MoE layers
                break;
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array>
NemotronHModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    std::unordered_map<std::string, mx::array> sanitized;

    for (auto& [key, value] : weights) {
        auto final_value = std::move(value);

        // Handle conv1d weight axis swap
        if (key.find("conv1d.weight") != std::string::npos && final_value.shape(-1) != 1) {
            final_value = mx::swapaxes(final_value, 1, 2);
        }

        sanitized.insert_or_assign(key, std::move(final_value));
    }

    // Stack experts: backbone.layers.{l}.mixer.experts.{e}.{proj}.weight
    //            -> backbone.layers.{l}.mixer.switch_mlp.{fc1|fc2}.weight
    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        std::string prefix = "backbone.layers." + std::to_string(l) + ".mixer";

        // Map: (source_proj_name, dest_fc_name)
        std::pair<std::string, std::string> proj_map[] = {
            {"up_proj", "fc1"},
            {"down_proj", "fc2"},
        };

        for (auto& [src_proj, dst_fc] : proj_map) {
            std::string first_key = prefix + ".experts.0." + src_proj + ".weight";
            if (sanitized.find(first_key) != sanitized.end()) {
                std::vector<mx::array> to_join;
                for (int e = 0; e < config_.n_routed_experts; ++e) {
                    std::string expert_key = prefix + ".experts." + std::to_string(e) + "." + src_proj + ".weight";
                    auto it = sanitized.find(expert_key);
                    if (it != sanitized.end()) {
                        to_join.push_back(std::move(it->second));
                        sanitized.erase(it);
                    }
                }
                if (!to_join.empty()) {
                    sanitized.insert_or_assign(
                        prefix + ".switch_mlp." + dst_fc + ".weight",
                        mx::stack(to_join));
                }
            }
        }
    }

    return sanitized;
}

void NemotronHModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> NemotronHModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : backbone_.weight_map()) map["backbone." + k] = v;
    if (lm_head_weight_.has_value()) {
        map["lm_head.weight"] = &lm_head_weight_.value();
    }
    return map;
}

} // namespace mlx_lm
