// Tests for NemotronH model — ported from NemotronHTests.swift

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <mlx-lm/llm/models/nemotron_h.h>

using Catch::Approx;
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>

namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// Helper: create a minimal NemotronH config for fast tests
// ---------------------------------------------------------------------------
static mlx_lm::NemotronHConfiguration make_test_config(const std::string& pattern = "M*M-E") {
    nlohmann::json j = {
        {"vocab_size", 100},
        {"hidden_size", 64},
        {"num_hidden_layers", static_cast<int>(pattern.size())},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 2},
        {"mamba_num_heads", 4},
        {"mamba_head_dim", 16},
        {"ssm_state_size", 16},
        {"conv_kernel", 4},
        {"n_groups", 2},
        {"intermediate_size", 128},
        {"moe_intermediate_size", 64},
        {"moe_shared_expert_intermediate_size", 64},
        {"n_routed_experts", 4},
        {"num_experts_per_tok", 2},
        {"hybrid_override_pattern", pattern},
        {"layer_norm_epsilon", 1e-5},
        {"n_group", 2},
        {"topk_group", 1}
    };
    mlx_lm::NemotronHConfiguration config;
    mlx_lm::from_json(j, config);
    return config;
}

// ============================================================================
// Block type parsing
// ============================================================================

TEST_CASE("NemotronH block type parsing", "[nemotron_h]") {
    SECTION("Mamba") {
        REQUIRE(mlx_lm::parse_block_type('M') == mlx_lm::NemotronHBlockType::Mamba);
    }
    SECTION("Attention") {
        REQUIRE(mlx_lm::parse_block_type('*') == mlx_lm::NemotronHBlockType::Attention);
    }
    SECTION("MLP") {
        REQUIRE(mlx_lm::parse_block_type('-') == mlx_lm::NemotronHBlockType::MLP);
    }
    SECTION("MoE") {
        REQUIRE(mlx_lm::parse_block_type('E') == mlx_lm::NemotronHBlockType::MoE);
    }
    SECTION("Invalid character throws") {
        REQUIRE_THROWS_AS(mlx_lm::parse_block_type('X'), std::runtime_error);
    }
}

// ============================================================================
// Configuration decoding from JSON
// ============================================================================

TEST_CASE("NemotronH config decoding from JSON", "[nemotron_h]") {
    nlohmann::json j = {
        {"model_type", "nemotron_h"},
        {"vocab_size", 131072},
        {"hidden_size", 4096},
        {"num_hidden_layers", 32},
        {"num_attention_heads", 32},
        {"num_key_value_heads", 8},
        {"mamba_num_heads", 64},
        {"mamba_head_dim", 64},
        {"ssm_state_size", 128},
        {"conv_kernel", 4},
        {"n_groups", 8},
        {"intermediate_size", 16384},
        {"moe_intermediate_size", 1024},
        {"moe_shared_expert_intermediate_size", 8192},
        {"n_routed_experts", 64},
        {"num_experts_per_tok", 4},
        {"hybrid_override_pattern", "M*M-E*"},
        {"layer_norm_epsilon", 1e-5},
        {"n_group", 4},
        {"topk_group", 2}
    };

    mlx_lm::NemotronHConfiguration config;
    mlx_lm::from_json(j, config);

    CHECK(config.vocab_size == 131072);
    CHECK(config.hidden_size == 4096);
    CHECK(config.num_hidden_layers == 32);
    CHECK(config.num_attention_heads == 32);
    CHECK(config.num_key_value_heads == 8);
    CHECK(config.mamba_num_heads == 64);
    CHECK(config.mamba_head_dim == 64);
    CHECK(config.ssm_state_size == 128);
    CHECK(config.conv_kernel == 4);
    CHECK(config.n_groups == 8);
    CHECK(config.intermediate_size == 16384);
    CHECK(config.moe_intermediate_size == 1024);
    CHECK(config.n_routed_experts == 64);
    CHECK(config.num_experts_per_tok == 4);
    CHECK(config.hybrid_override_pattern == "M*M-E*");
    CHECK(config.n_group == 4);
    CHECK(config.topk_group == 2);
}

TEST_CASE("NemotronH config decoding with array pattern", "[nemotron_h]") {
    // Some configs have hybrid_override_pattern as array of strings
    nlohmann::json j = {
        {"vocab_size", 100},
        {"hidden_size", 64},
        {"num_hidden_layers", 4},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 2},
        {"mamba_num_heads", 4},
        {"mamba_head_dim", 16},
        {"ssm_state_size", 16},
        {"conv_kernel", 4},
        {"n_groups", 2},
        {"intermediate_size", 128},
        {"moe_intermediate_size", 64},
        {"moe_shared_expert_intermediate_size", 64},
        {"n_routed_experts", 4},
        {"num_experts_per_tok", 2},
        {"hybrid_override_pattern", {"M", "*", "M", "-"}}
    };

    mlx_lm::NemotronHConfiguration config;
    mlx_lm::from_json(j, config);

    REQUIRE(config.hybrid_override_pattern == "M*M-");
}

TEST_CASE("NemotronH config decoding with time_step_limit array", "[nemotron_h]") {
    // time_step_limit_min can be [min, max] as separate fields
    nlohmann::json j = {
        {"vocab_size", 100},
        {"hidden_size", 64},
        {"num_hidden_layers", 2},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 2},
        {"mamba_num_heads", 4},
        {"mamba_head_dim", 16},
        {"ssm_state_size", 16},
        {"conv_kernel", 4},
        {"n_groups", 2},
        {"intermediate_size", 128},
        {"moe_intermediate_size", 64},
        {"moe_shared_expert_intermediate_size", 64},
        {"n_routed_experts", 4},
        {"num_experts_per_tok", 2},
        {"hybrid_override_pattern", "M*"},
        {"time_step_limit", {0.0, 1000.0}}
    };

    mlx_lm::NemotronHConfiguration config;
    mlx_lm::from_json(j, config);

    CHECK(config.time_step_limit_min == 0.0f);
    CHECK(config.time_step_limit_max == 1000.0f);
}

TEST_CASE("NemotronH config decoding with defaults", "[nemotron_h]") {
    // Minimal config - should use defaults for optional fields
    nlohmann::json j = {
        {"vocab_size", 100},
        {"hidden_size", 64},
        {"num_hidden_layers", 2},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 2},
        {"mamba_num_heads", 4},
        {"mamba_head_dim", 16},
        {"ssm_state_size", 16},
        {"conv_kernel", 4},
        {"n_groups", 2},
        {"intermediate_size", 128},
        {"moe_intermediate_size", 64},
        {"moe_shared_expert_intermediate_size", 64},
        {"n_routed_experts", 4},
        {"num_experts_per_tok", 2},
        {"hybrid_override_pattern", "M*"}
    };

    mlx_lm::NemotronHConfiguration config;
    mlx_lm::from_json(j, config);

    // Check defaults
    CHECK(config.attention_bias == false);
    CHECK(config.mamba_proj_bias == false);
    CHECK(config.mlp_bias == false);
    CHECK(config.use_conv_bias == true);
    CHECK(config.tie_word_embeddings == false);
    CHECK(config.layer_norm_epsilon == Approx(1e-5f));
    CHECK(config.rope_theta == Approx(10000.0f));
    CHECK(config.n_group == 1);
    CHECK(config.topk_group == 1);
    CHECK(config.norm_topk_prob == true);
    CHECK(config.routed_scaling_factor == Approx(1.0f));
}

// ============================================================================
// Pattern parsing — verify each character maps to the right block type
// ============================================================================

TEST_CASE("NemotronH pattern to block types", "[nemotron_h]") {
    SECTION("Mixed pattern M*M-E") {
        std::string pattern = "M*M-E";
        std::vector<mlx_lm::NemotronHBlockType> expected = {
            mlx_lm::NemotronHBlockType::Mamba,
            mlx_lm::NemotronHBlockType::Attention,
            mlx_lm::NemotronHBlockType::Mamba,
            mlx_lm::NemotronHBlockType::MLP,
            mlx_lm::NemotronHBlockType::MoE,
        };
        for (size_t i = 0; i < pattern.size(); ++i) {
            REQUIRE(mlx_lm::parse_block_type(pattern[i]) == expected[i]);
        }
    }

    SECTION("Mamba only") {
        std::string pattern = "MMM";
        for (char c : pattern) {
            REQUIRE(mlx_lm::parse_block_type(c) == mlx_lm::NemotronHBlockType::Mamba);
        }
    }

    SECTION("Attention only") {
        std::string pattern = "***";
        for (char c : pattern) {
            REQUIRE(mlx_lm::parse_block_type(c) == mlx_lm::NemotronHBlockType::Attention);
        }
    }
}

// ============================================================================
// Cache creation — verify correct count and types
// ============================================================================

TEST_CASE("NemotronH cache creation M*M-", "[nemotron_h]") {
    // Pattern: M*M- has 2 Mamba + 1 Attention = 3 cacheable layers
    auto config = make_test_config("M*M-");
    mlx_lm::NemotronHModel model(config);
    auto cache = model.new_cache();

    // M, *, M are cacheable; '-' (MLP) is not
    REQUIRE(cache.size() == 3);
}

TEST_CASE("NemotronH cache creation Mamba only", "[nemotron_h]") {
    auto config = make_test_config("MMM");
    mlx_lm::NemotronHModel model(config);
    auto cache = model.new_cache();

    // 3 Mamba layers = 3 caches
    REQUIRE(cache.size() == 3);
}

TEST_CASE("NemotronH cache creation Attention only", "[nemotron_h]") {
    auto config = make_test_config("***");
    mlx_lm::NemotronHModel model(config);
    auto cache = model.new_cache();

    // 3 Attention layers = 3 caches
    REQUIRE(cache.size() == 3);
}

TEST_CASE("NemotronH cache count mixed (M-E*-E)", "[nemotron_h]") {
    // Pattern with MLP (-) and MoE (E) which don't have caches
    auto config = make_test_config("M-E*-E");
    mlx_lm::NemotronHModel model(config);
    auto cache = model.new_cache();

    // Only M and * have caches: M, * = 2 caches
    REQUIRE(cache.size() == 2);
}

// ============================================================================
// Cache type verification
// ============================================================================

TEST_CASE("NemotronH cache types are correct", "[nemotron_h]") {
    auto config = make_test_config("M*");
    mlx_lm::NemotronHModel model(config);
    auto cache = model.new_cache();

    REQUIRE(cache.size() == 2);

    // First cache (M) should be a MambaCache
    REQUIRE(cache[0].as_mamba() != nullptr);

    // Second cache (*) should be a KVCacheSimple (not MambaCache)
    REQUIRE(cache[1].as_mamba() == nullptr);
}

// ============================================================================
// Model construction — verify vocab size
// ============================================================================

TEST_CASE("NemotronH vocabulary size", "[nemotron_h]") {
    auto config = make_test_config("M*");
    mlx_lm::NemotronHModel model(config);

    REQUIRE(model.vocab_size() == 100);
}

// ============================================================================
// Tied vs untied embeddings (construction only)
// ============================================================================

TEST_CASE("NemotronH tied embeddings construction", "[nemotron_h]") {
    nlohmann::json j = {
        {"vocab_size", 100},
        {"hidden_size", 64},
        {"num_hidden_layers", 2},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 2},
        {"mamba_num_heads", 4},
        {"mamba_head_dim", 16},
        {"ssm_state_size", 16},
        {"conv_kernel", 4},
        {"n_groups", 2},
        {"intermediate_size", 128},
        {"moe_intermediate_size", 64},
        {"moe_shared_expert_intermediate_size", 64},
        {"n_routed_experts", 4},
        {"num_experts_per_tok", 2},
        {"hybrid_override_pattern", "M*"},
        {"tie_word_embeddings", true}
    };

    mlx_lm::NemotronHConfiguration config;
    mlx_lm::from_json(j, config);
    REQUIRE(config.tie_word_embeddings == true);

    mlx_lm::NemotronHModel model(config);
    auto wmap = model.weight_map();
    // With tied embeddings, there should be no lm_head.weight
    REQUIRE(wmap.find("lm_head.weight") == wmap.end());
    // But embeddings should still exist
    REQUIRE(wmap.find("backbone.embeddings.weight") != wmap.end());
}

TEST_CASE("NemotronH untied embeddings construction", "[nemotron_h]") {
    nlohmann::json j = {
        {"vocab_size", 100},
        {"hidden_size", 64},
        {"num_hidden_layers", 2},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 2},
        {"mamba_num_heads", 4},
        {"mamba_head_dim", 16},
        {"ssm_state_size", 16},
        {"conv_kernel", 4},
        {"n_groups", 2},
        {"intermediate_size", 128},
        {"moe_intermediate_size", 64},
        {"moe_shared_expert_intermediate_size", 64},
        {"n_routed_experts", 4},
        {"num_experts_per_tok", 2},
        {"hybrid_override_pattern", "M*"},
        {"tie_word_embeddings", false}
    };

    mlx_lm::NemotronHConfiguration config;
    mlx_lm::from_json(j, config);
    REQUIRE(config.tie_word_embeddings == false);

    mlx_lm::NemotronHModel model(config);
    auto wmap = model.weight_map();
    // With untied embeddings, lm_head.weight should exist
    REQUIRE(wmap.find("lm_head.weight") != wmap.end());
    REQUIRE(wmap.find("backbone.embeddings.weight") != wmap.end());
}

// ============================================================================
// Weight map structure — verify expected keys exist for various patterns
// ============================================================================

TEST_CASE("NemotronH weight map keys", "[nemotron_h]") {
    SECTION("Mamba layer weights") {
        auto config = make_test_config("M");
        mlx_lm::NemotronHModel model(config);
        auto wmap = model.weight_map();

        CHECK(wmap.find("backbone.layers.0.norm.weight") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.in_proj.weight") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.conv1d.weight") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.out_proj.weight") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.A_log") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.D") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.dt_bias") != wmap.end());
        CHECK(wmap.find("backbone.norm_f.weight") != wmap.end());
        CHECK(wmap.find("backbone.embeddings.weight") != wmap.end());
    }

    SECTION("Attention layer weights") {
        auto config = make_test_config("*");
        mlx_lm::NemotronHModel model(config);
        auto wmap = model.weight_map();

        CHECK(wmap.find("backbone.layers.0.mixer.q_proj.weight") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.k_proj.weight") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.v_proj.weight") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.o_proj.weight") != wmap.end());
    }

    SECTION("MLP layer weights") {
        auto config = make_test_config("-");
        mlx_lm::NemotronHModel model(config);
        auto wmap = model.weight_map();

        CHECK(wmap.find("backbone.layers.0.mixer.up_proj.weight") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.down_proj.weight") != wmap.end());
    }

    SECTION("MoE layer weights") {
        auto config = make_test_config("E");
        mlx_lm::NemotronHModel model(config);
        auto wmap = model.weight_map();

        CHECK(wmap.find("backbone.layers.0.mixer.gate.weight") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.gate.e_score_correction_bias") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.switch_mlp.fc1.weight") != wmap.end());
        CHECK(wmap.find("backbone.layers.0.mixer.switch_mlp.fc2.weight") != wmap.end());
    }
}

// ============================================================================
// Sanitize — conv1d weight axis swap
// ============================================================================

TEST_CASE("NemotronH sanitize conv1d weights", "[nemotron_h]") {
    auto config = make_test_config("M*");
    mlx_lm::NemotronHModel model(config);

    // The sanitization swaps axes 1 and 2 when dim(-1) != 1
    // Python format comes in as [convDim, 1, kernelSize]
    // C++ expects: [convDim, kernelSize, 1]
    int conv_dim = config.mamba_num_heads * config.mamba_head_dim
                   + 2 * config.n_groups * config.ssm_state_size;

    auto mock_conv_weight = mx::ones({conv_dim, 1, config.conv_kernel});

    std::unordered_map<std::string, mx::array> weights;
    weights.insert_or_assign("backbone.layers.0.mixer.conv1d.weight", mock_conv_weight);

    auto sanitized = model.sanitize(std::move(weights));

    auto it = sanitized.find("backbone.layers.0.mixer.conv1d.weight");
    REQUIRE(it != sanitized.end());
    CHECK(it->second.shape(0) == conv_dim);
    CHECK(it->second.shape(1) == config.conv_kernel);
    CHECK(it->second.shape(2) == 1);
}

TEST_CASE("NemotronH sanitize conv1d weights no-op when already correct", "[nemotron_h]") {
    auto config = make_test_config("M*");
    mlx_lm::NemotronHModel model(config);

    // When dim(-1) == 1, no transpose needed
    int conv_dim = config.mamba_num_heads * config.mamba_head_dim
                   + 2 * config.n_groups * config.ssm_state_size;

    auto mock_conv_weight = mx::ones({conv_dim, config.conv_kernel, 1});

    std::unordered_map<std::string, mx::array> weights;
    weights.insert_or_assign("backbone.layers.0.mixer.conv1d.weight", mock_conv_weight);

    auto sanitized = model.sanitize(std::move(weights));

    auto it = sanitized.find("backbone.layers.0.mixer.conv1d.weight");
    REQUIRE(it != sanitized.end());
    CHECK(it->second.shape(0) == conv_dim);
    CHECK(it->second.shape(1) == config.conv_kernel);
    CHECK(it->second.shape(2) == 1);
}

// ============================================================================
// Sanitize — expert weight stacking
// ============================================================================

TEST_CASE("NemotronH sanitize expert weights", "[nemotron_h]") {
    auto config = make_test_config("E");
    mlx_lm::NemotronHModel model(config);

    // Create mock per-expert weights that need stacking
    std::unordered_map<std::string, mx::array> weights;
    for (int e = 0; e < config.n_routed_experts; ++e) {
        std::string up_key = "backbone.layers.0.mixer.experts." + std::to_string(e) + ".up_proj.weight";
        std::string down_key = "backbone.layers.0.mixer.experts." + std::to_string(e) + ".down_proj.weight";
        weights.insert_or_assign(up_key, mx::ones({config.moe_intermediate_size, config.hidden_size}));
        weights.insert_or_assign(down_key, mx::ones({config.hidden_size, config.moe_intermediate_size}));
    }

    auto sanitized = model.sanitize(std::move(weights));

    // Experts should be stacked into switch_mlp format
    auto fc1_it = sanitized.find("backbone.layers.0.mixer.switch_mlp.fc1.weight");
    auto fc2_it = sanitized.find("backbone.layers.0.mixer.switch_mlp.fc2.weight");

    REQUIRE(fc1_it != sanitized.end());
    REQUIRE(fc2_it != sanitized.end());

    CHECK(fc1_it->second.shape(0) == config.n_routed_experts);
    CHECK(fc1_it->second.shape(1) == config.moe_intermediate_size);
    CHECK(fc1_it->second.shape(2) == config.hidden_size);

    CHECK(fc2_it->second.shape(0) == config.n_routed_experts);
    CHECK(fc2_it->second.shape(1) == config.hidden_size);
    CHECK(fc2_it->second.shape(2) == config.moe_intermediate_size);

    // Original expert keys should be removed
    CHECK(sanitized.find("backbone.layers.0.mixer.experts.0.up_proj.weight") == sanitized.end());
    CHECK(sanitized.find("backbone.layers.0.mixer.experts.0.down_proj.weight") == sanitized.end());
}

// ============================================================================
// Sanitize — preserves other weights
// ============================================================================

TEST_CASE("NemotronH sanitize preserves other weights", "[nemotron_h]") {
    auto config = make_test_config("M*");
    mlx_lm::NemotronHModel model(config);

    std::unordered_map<std::string, mx::array> weights;
    weights.insert_or_assign("backbone.embeddings.weight", mx::ones({config.vocab_size, config.hidden_size}));
    weights.insert_or_assign("backbone.norm_f.weight", mx::ones({config.hidden_size}));

    auto sanitized = model.sanitize(std::move(weights));

    REQUIRE(sanitized.find("backbone.embeddings.weight") != sanitized.end());
    REQUIRE(sanitized.find("backbone.norm_f.weight") != sanitized.end());

    auto& emb = sanitized.at("backbone.embeddings.weight");
    CHECK(emb.shape(0) == config.vocab_size);
    CHECK(emb.shape(1) == config.hidden_size);
}

// ============================================================================
// Shared experts configuration
// ============================================================================

TEST_CASE("NemotronH with shared experts construction", "[nemotron_h]") {
    nlohmann::json j = {
        {"vocab_size", 100},
        {"hidden_size", 64},
        {"num_hidden_layers", 1},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 2},
        {"mamba_num_heads", 4},
        {"mamba_head_dim", 16},
        {"ssm_state_size", 16},
        {"conv_kernel", 4},
        {"n_groups", 2},
        {"intermediate_size", 128},
        {"moe_intermediate_size", 64},
        {"moe_shared_expert_intermediate_size", 64},
        {"n_routed_experts", 4},
        {"num_experts_per_tok", 2},
        {"hybrid_override_pattern", "E"},
        {"n_shared_experts", 1}
    };

    mlx_lm::NemotronHConfiguration config;
    mlx_lm::from_json(j, config);
    REQUIRE(config.n_shared_experts.has_value());
    REQUIRE(config.n_shared_experts.value() == 1);

    // Model should construct without error
    mlx_lm::NemotronHModel model(config);

    // Verify shared expert weights exist in weight map
    auto wmap = model.weight_map();
    CHECK(wmap.find("backbone.layers.0.mixer.shared_experts.up_proj.weight") != wmap.end());
    CHECK(wmap.find("backbone.layers.0.mixer.shared_experts.down_proj.weight") != wmap.end());
}

// ============================================================================
// Complex pattern tests — alternating and MoE-heavy
// ============================================================================

TEST_CASE("NemotronH alternating pattern cache count", "[nemotron_h]") {
    auto config = make_test_config("M*M*M*M*");
    mlx_lm::NemotronHModel model(config);
    auto cache = model.new_cache();

    // M*M*M*M* has 8 cacheable layers (all M and * are cacheable)
    REQUIRE(cache.size() == 8);
}

TEST_CASE("NemotronH MoE-heavy pattern cache count", "[nemotron_h]") {
    // Pattern with multiple MoE layers
    auto config = make_test_config("MEE*EE");
    mlx_lm::NemotronHModel model(config);
    auto cache = model.new_cache();

    // Only M and * contribute caches: M, * = 2 caches
    REQUIRE(cache.size() == 2);
}

// ============================================================================
// RotatingKVCache creation with max_kv_size
// ============================================================================

TEST_CASE("NemotronH cache with max_kv_size", "[nemotron_h]") {
    auto config = make_test_config("M*");
    mlx_lm::NemotronHModel model(config);

    mlx_lm::GenerateParameters params;
    params.max_kv_size = 128;
    auto cache = model.new_cache(params);

    REQUIRE(cache.size() == 2);

    // Mamba cache should still be MambaCache
    REQUIRE(cache[0].as_mamba() != nullptr);

    // Attention cache with max_kv_size should be a RotatingKVCache
    REQUIRE(cache[1].as_mamba() == nullptr);
    REQUIRE(cache[1].max_size().has_value());
    CHECK(cache[1].max_size().value() == 128);
}

// ============================================================================
// Edge case: single layer patterns
// ============================================================================

TEST_CASE("NemotronH single Mamba layer", "[nemotron_h]") {
    auto config = make_test_config("M");
    mlx_lm::NemotronHModel model(config);

    REQUIRE(model.vocab_size() == 100);

    auto cache = model.new_cache();
    REQUIRE(cache.size() == 1);
    REQUIRE(cache[0].as_mamba() != nullptr);
}

TEST_CASE("NemotronH single Attention layer", "[nemotron_h]") {
    auto config = make_test_config("*");
    mlx_lm::NemotronHModel model(config);

    REQUIRE(model.vocab_size() == 100);

    auto cache = model.new_cache();
    REQUIRE(cache.size() == 1);
    REQUIRE(cache[0].as_mamba() == nullptr);
}

TEST_CASE("NemotronH single MLP layer", "[nemotron_h]") {
    auto config = make_test_config("-");
    mlx_lm::NemotronHModel model(config);

    // MLP does not produce a cache entry
    auto cache = model.new_cache();
    REQUIRE(cache.size() == 0);
}

TEST_CASE("NemotronH single MoE layer", "[nemotron_h]") {
    auto config = make_test_config("E");
    mlx_lm::NemotronHModel model(config);

    // MoE does not produce a cache entry
    auto cache = model.new_cache();
    REQUIRE(cache.size() == 0);
}
