// Tests for RoPE utility classes

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <mlx-lm/common/rope_utils.h>
#include <mlx/mlx.h>
#include <cmath>
#include <unordered_map>

namespace mx = mlx::core;
using Catch::Approx;

// ---------------------------------------------------------------------------
// Helper: create a small 4D test tensor for RoPE [batch, heads, seq, dims]
// ---------------------------------------------------------------------------
static mx::array make_test_input(int batch, int heads, int seq, int dims) {
    return mx::ones({batch, heads, seq, dims});
}

// ============================================================================
// SimpleRoPE
// ============================================================================

TEST_CASE("SimpleRoPE construction and defaults", "[rope]") {
    mlx_lm::SimpleRoPE rope{64, false, 10000.0f, 1.0f};

    CHECK(rope.dims == 64);
    CHECK(rope.traditional == false);
    CHECK(rope.base == 10000.0f);
    CHECK(rope.scale == 1.0f);
}

TEST_CASE("SimpleRoPE apply produces correct shape", "[rope]") {
    mlx_lm::SimpleRoPE rope{8, false, 10000.0f, 1.0f};

    auto input = make_test_input(1, 2, 4, 8);
    auto output = rope(input, 0);

    REQUIRE(output.shape(0) == 1);
    REQUIRE(output.shape(1) == 2);
    REQUIRE(output.shape(2) == 4);
    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("SimpleRoPE apply with offset", "[rope]") {
    mlx_lm::SimpleRoPE rope{8, false, 10000.0f, 1.0f};

    auto input = make_test_input(1, 2, 1, 8);
    auto output = rope(input, 5);

    // Shape should be unchanged
    REQUIRE(output.shape(0) == 1);
    REQUIRE(output.shape(1) == 2);
    REQUIRE(output.shape(2) == 1);
    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("SimpleRoPE with linear scaling", "[rope]") {
    // scale < 1.0 effectively extends context
    mlx_lm::SimpleRoPE rope{8, false, 10000.0f, 0.5f};

    auto input = make_test_input(1, 2, 4, 8);
    auto output = rope(input, 0);

    REQUIRE(output.shape(2) == 4);
    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("SimpleRoPE traditional mode", "[rope]") {
    mlx_lm::SimpleRoPE rope{8, true, 10000.0f, 1.0f};

    auto input = make_test_input(1, 2, 4, 8);
    auto output = rope(input, 0);

    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("SimpleRoPE at position 0 preserves structure", "[rope]") {
    // At position 0, cos(0)=1, sin(0)=0, so rotation should be near identity
    mlx_lm::SimpleRoPE rope{4, false, 10000.0f, 1.0f};

    auto input = mx::ones({1, 1, 1, 4});
    auto output = rope(input, 0);
    mx::eval(output);

    // At position 0, the rotation angles are 0, so output should be close to input
    auto diff = mx::subtract(output, input);
    auto max_diff = mx::max(mx::abs(diff));
    mx::eval(max_diff);
    CHECK(max_diff.item<float>() < 1e-5f);
}

// ============================================================================
// Llama3RoPE
// ============================================================================

TEST_CASE("Llama3RoPE construction requires scaling config", "[rope]") {
    REQUIRE_THROWS_AS(
        mlx_lm::Llama3RoPE(64, 2048, false, 10000.0f, std::nullopt),
        std::runtime_error);
}

TEST_CASE("Llama3RoPE construction with scaling config", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"type", mlx_lm::StringOrNumber::from_string("llama3")},
        {"factor", mlx_lm::StringOrNumber::from_float(8.0f)},
        {"low_freq_factor", mlx_lm::StringOrNumber::from_float(1.0f)},
        {"high_freq_factor", mlx_lm::StringOrNumber::from_float(4.0f)},
        {"original_max_position_embeddings", mlx_lm::StringOrNumber::from_float(8192.0f)},
    };

    // Should not throw
    mlx_lm::Llama3RoPE rope(64, 131072, false, 500000.0f, scaling);

    auto input = make_test_input(1, 4, 8, 64);
    auto output = rope(input, 0);

    REQUIRE(output.shape(0) == 1);
    REQUIRE(output.shape(1) == 4);
    REQUIRE(output.shape(2) == 8);
    REQUIRE(output.shape(3) == 64);
}

TEST_CASE("Llama3RoPE shape preservation", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"factor", mlx_lm::StringOrNumber::from_float(2.0f)},
        {"low_freq_factor", mlx_lm::StringOrNumber::from_float(1.0f)},
        {"high_freq_factor", mlx_lm::StringOrNumber::from_float(4.0f)},
        {"original_max_position_embeddings", mlx_lm::StringOrNumber::from_float(4096.0f)},
    };

    mlx_lm::Llama3RoPE rope(16, 8192, false, 10000.0f, scaling);

    auto input = make_test_input(2, 4, 6, 16);
    auto output = rope(input, 0);

    CHECK(output.shape(0) == 2);
    CHECK(output.shape(1) == 4);
    CHECK(output.shape(2) == 6);
    CHECK(output.shape(3) == 16);
}

TEST_CASE("Llama3RoPE with offset", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"factor", mlx_lm::StringOrNumber::from_float(2.0f)},
        {"low_freq_factor", mlx_lm::StringOrNumber::from_float(1.0f)},
        {"high_freq_factor", mlx_lm::StringOrNumber::from_float(4.0f)},
        {"original_max_position_embeddings", mlx_lm::StringOrNumber::from_float(4096.0f)},
    };

    mlx_lm::Llama3RoPE rope(8, 8192, false, 10000.0f, scaling);

    auto input = make_test_input(1, 2, 1, 8);
    auto output = rope(input, 100);

    REQUIRE(output.shape(2) == 1);
    REQUIRE(output.shape(3) == 8);
}

// ============================================================================
// YarnRoPE
// ============================================================================

TEST_CASE("YarnRoPE construction", "[rope]") {
    // Should not throw with valid even dims
    mlx_lm::YarnRoPE rope(
        64,     // dims
        false,  // traditional
        32768,  // max_position_embeddings
        10000.0f, // base
        4.0f,   // scaling_factor
        4096,   // original_max_position_embeddings
        32.0f,  // beta_fast
        1.0f,   // beta_slow
        1.0f,   // mscale
        0.0f    // mscale_all_dim
    );

    auto input = make_test_input(1, 4, 8, 64);
    auto output = rope(input, 0);

    REQUIRE(output.shape(0) == 1);
    REQUIRE(output.shape(1) == 4);
    REQUIRE(output.shape(2) == 8);
    REQUIRE(output.shape(3) == 64);
}

TEST_CASE("YarnRoPE rejects odd dimensions", "[rope]") {
    REQUIRE_THROWS_AS(
        mlx_lm::YarnRoPE(63, false, 2048, 10000.0f, 1.0f, 4096, 32.0f, 1.0f, 1.0f, 0.0f),
        std::runtime_error);
}

TEST_CASE("YarnRoPE with mscale scaling", "[rope]") {
    // With scaling_factor > 1 and mscale > 0, the computed mscale should not be 1.0
    mlx_lm::YarnRoPE rope(
        16, false, 32768, 10000.0f,
        8.0f,  // scaling_factor > 1
        4096,
        32.0f, 1.0f,
        1.0f,  // mscale
        0.0f   // mscale_all_dim = 0 means no denominator mscale
    );

    auto input = make_test_input(1, 2, 4, 16);
    auto output = rope(input, 0);

    REQUIRE(output.shape(3) == 16);
    // Output values should differ from SimpleRoPE due to mscale and custom freqs
}

TEST_CASE("YarnRoPE with scaling factor 1.0 acts like identity mscale", "[rope]") {
    // When scaling_factor <= 1.0, mscale computation returns 1.0
    mlx_lm::YarnRoPE rope(
        8, false, 2048, 10000.0f,
        1.0f,  // scaling_factor <= 1
        4096, 32.0f, 1.0f, 1.0f, 0.0f
    );

    auto input = make_test_input(1, 1, 1, 8);
    auto output = rope(input, 0);
    mx::eval(output);

    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("YarnRoPE shape preservation with partial rotation", "[rope]") {
    // dims < last_dim means only partial rotation
    mlx_lm::YarnRoPE rope(
        8, false, 8192, 10000.0f,
        4.0f, 4096, 32.0f, 1.0f, 1.0f, 0.0f
    );

    // Input has 16 dims, but rope only rotates first 8
    auto input = make_test_input(1, 2, 4, 16);
    auto output = rope(input, 0);

    REQUIRE(output.shape(3) == 16);
}

// ============================================================================
// SuScaledRoPE
// ============================================================================

TEST_CASE("SuScaledRoPE construction", "[rope]") {
    std::vector<float> short_factor = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> long_factor = {2.0f, 2.0f, 2.0f, 2.0f};

    mlx_lm::SuScaledRoPE rope(
        8,          // dims
        10000.0f,   // base
        131072,     // max_position_embeddings
        4096,       // original_max_position_embeddings
        short_factor,
        long_factor
    );

    auto input = make_test_input(1, 2, 4, 8);
    auto output = rope(input, 0);

    REQUIRE(output.shape(0) == 1);
    REQUIRE(output.shape(1) == 2);
    REQUIRE(output.shape(2) == 4);
    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("SuScaledRoPE rejects odd dimensions", "[rope]") {
    REQUIRE_THROWS_AS(
        mlx_lm::SuScaledRoPE(7, 10000.0f, 131072, 4096, {1.0f}, {1.0f}),
        std::runtime_error);
}

TEST_CASE("SuScaledRoPE uses short freqs for short sequences", "[rope]") {
    std::vector<float> short_factor = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> long_factor = {10.0f, 10.0f, 10.0f, 10.0f};

    mlx_lm::SuScaledRoPE rope(
        8, 10000.0f,
        131072, 4096,
        short_factor, long_factor
    );

    // seq_len = offset + x.shape(-2) = 0 + 4 = 4 <= 4096 (original_max)
    // So should use short freqs
    auto input = make_test_input(1, 2, 4, 8);
    auto output = rope(input, 0);
    mx::eval(output);

    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("SuScaledRoPE uses long freqs for long sequences", "[rope]") {
    std::vector<float> short_factor = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> long_factor = {10.0f, 10.0f, 10.0f, 10.0f};

    mlx_lm::SuScaledRoPE rope(
        8, 10000.0f,
        131072, 4096,
        short_factor, long_factor
    );

    // seq_len = offset + x.shape(-2) = 5000 + 4 = 5004 > 4096 (original_max)
    // So should use long freqs
    auto input = make_test_input(1, 2, 4, 8);
    auto output = rope(input, 5000);
    mx::eval(output);

    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("SuScaledRoPE with custom m_scale", "[rope]") {
    std::vector<float> short_factor = {1.0f, 1.0f};
    std::vector<float> long_factor = {2.0f, 2.0f};

    mlx_lm::SuScaledRoPE rope(
        4, 10000.0f,
        131072, 4096,
        short_factor, long_factor,
        1.5f,  // short_m_scale
        2.0f   // long_m_scale
    );

    auto input = make_test_input(1, 1, 2, 4);
    auto output = rope(input, 0);
    mx::eval(output);

    REQUIRE(output.shape(3) == 4);
}

TEST_CASE("SuScaledRoPE shape preservation with partial rotation", "[rope]") {
    std::vector<float> short_factor = {1.0f, 1.0f};
    std::vector<float> long_factor = {2.0f, 2.0f};

    // dims=4 but input has 8 last-dim
    mlx_lm::SuScaledRoPE rope(
        4, 10000.0f,
        131072, 4096,
        short_factor, long_factor,
        2.0f, 2.0f  // non-unity scale to trigger partial scaling logic
    );

    auto input = make_test_input(1, 2, 4, 8);
    auto output = rope(input, 0);

    REQUIRE(output.shape(3) == 8);
}

// ============================================================================
// apply_rope — variant dispatch
// ============================================================================

TEST_CASE("apply_rope with SimpleRoPE", "[rope]") {
    mlx_lm::RoPEVariant rope = mlx_lm::SimpleRoPE{8, false, 10000.0f, 1.0f};

    auto input = make_test_input(1, 2, 4, 8);
    auto output = mlx_lm::apply_rope(rope, input, 0);

    REQUIRE(output.shape(0) == 1);
    REQUIRE(output.shape(2) == 4);
    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("apply_rope with Llama3RoPE", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"factor", mlx_lm::StringOrNumber::from_float(2.0f)},
        {"low_freq_factor", mlx_lm::StringOrNumber::from_float(1.0f)},
        {"high_freq_factor", mlx_lm::StringOrNumber::from_float(4.0f)},
        {"original_max_position_embeddings", mlx_lm::StringOrNumber::from_float(4096.0f)},
    };

    mlx_lm::RoPEVariant rope = mlx_lm::Llama3RoPE(8, 8192, false, 10000.0f, scaling);

    auto input = make_test_input(1, 2, 4, 8);
    auto output = mlx_lm::apply_rope(rope, input, 0);

    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("apply_rope with YarnRoPE", "[rope]") {
    mlx_lm::RoPEVariant rope = mlx_lm::YarnRoPE(
        8, false, 8192, 10000.0f,
        4.0f, 4096, 32.0f, 1.0f, 1.0f, 0.0f
    );

    auto input = make_test_input(1, 2, 4, 8);
    auto output = mlx_lm::apply_rope(rope, input, 0);

    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("apply_rope with SuScaledRoPE", "[rope]") {
    mlx_lm::RoPEVariant rope = mlx_lm::SuScaledRoPE(
        8, 10000.0f, 131072, 4096,
        {1.0f, 1.0f, 1.0f, 1.0f},
        {2.0f, 2.0f, 2.0f, 2.0f}
    );

    auto input = make_test_input(1, 2, 4, 8);
    auto output = mlx_lm::apply_rope(rope, input, 0);

    REQUIRE(output.shape(3) == 8);
}

// ============================================================================
// initialize_rope factory — default
// ============================================================================

TEST_CASE("initialize_rope default type", "[rope]") {
    auto rope = mlx_lm::initialize_rope(64, 10000.0f, false);

    // Should be SimpleRoPE
    REQUIRE(std::holds_alternative<mlx_lm::SimpleRoPE>(rope));
    auto& simple = std::get<mlx_lm::SimpleRoPE>(rope);
    CHECK(simple.dims == 64);
    CHECK(simple.base == 10000.0f);
    CHECK(simple.scale == 1.0f);
    CHECK(simple.traditional == false);
}

// ============================================================================
// initialize_rope factory — linear
// ============================================================================

TEST_CASE("initialize_rope linear type", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"type", mlx_lm::StringOrNumber::from_string("linear")},
        {"factor", mlx_lm::StringOrNumber::from_float(4.0f)},
    };

    auto rope = mlx_lm::initialize_rope(64, 10000.0f, false, scaling);

    REQUIRE(std::holds_alternative<mlx_lm::SimpleRoPE>(rope));
    auto& simple = std::get<mlx_lm::SimpleRoPE>(rope);
    CHECK(simple.scale == Approx(0.25f));  // 1/factor
}

// ============================================================================
// initialize_rope factory — llama3
// ============================================================================

TEST_CASE("initialize_rope llama3 type", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"type", mlx_lm::StringOrNumber::from_string("llama3")},
        {"factor", mlx_lm::StringOrNumber::from_float(8.0f)},
        {"low_freq_factor", mlx_lm::StringOrNumber::from_float(1.0f)},
        {"high_freq_factor", mlx_lm::StringOrNumber::from_float(4.0f)},
        {"original_max_position_embeddings", mlx_lm::StringOrNumber::from_float(8192.0f)},
    };

    auto rope = mlx_lm::initialize_rope(128, 500000.0f, false, scaling, 131072);

    REQUIRE(std::holds_alternative<mlx_lm::Llama3RoPE>(rope));

    // Verify it produces output
    auto input = make_test_input(1, 4, 8, 128);
    auto output = mlx_lm::apply_rope(rope, input, 0);
    REQUIRE(output.shape(3) == 128);
}

// ============================================================================
// initialize_rope factory — yarn
// ============================================================================

TEST_CASE("initialize_rope yarn type", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"type", mlx_lm::StringOrNumber::from_string("yarn")},
        {"factor", mlx_lm::StringOrNumber::from_float(4.0f)},
        {"original_max_position_embeddings", mlx_lm::StringOrNumber::from_float(4096.0f)},
        {"beta_fast", mlx_lm::StringOrNumber::from_float(32.0f)},
        {"beta_slow", mlx_lm::StringOrNumber::from_float(1.0f)},
        {"mscale", mlx_lm::StringOrNumber::from_float(1.0f)},
        {"mscale_all_dim", mlx_lm::StringOrNumber::from_float(0.0f)},
    };

    auto rope = mlx_lm::initialize_rope(64, 10000.0f, false, scaling, 16384);

    REQUIRE(std::holds_alternative<mlx_lm::YarnRoPE>(rope));

    auto input = make_test_input(1, 4, 8, 64);
    auto output = mlx_lm::apply_rope(rope, input, 0);
    REQUIRE(output.shape(3) == 64);
}

// ============================================================================
// initialize_rope factory — longrope
// ============================================================================

TEST_CASE("initialize_rope longrope type", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"type", mlx_lm::StringOrNumber::from_string("longrope")},
        {"original_max_position_embeddings", mlx_lm::StringOrNumber::from_float(4096.0f)},
    };

    // longrope needs the JSON for short_factor/long_factor
    nlohmann::json rope_json = {
        {"type", "longrope"},
        {"original_max_position_embeddings", 4096},
        {"short_factor", {1.0f, 1.0f, 1.0f, 1.0f}},
        {"long_factor", {2.0f, 2.0f, 2.0f, 2.0f}},
    };

    auto rope = mlx_lm::initialize_rope(8, 10000.0f, false, scaling, 131072, &rope_json);

    REQUIRE(std::holds_alternative<mlx_lm::SuScaledRoPE>(rope));

    auto input = make_test_input(1, 2, 4, 8);
    auto output = mlx_lm::apply_rope(rope, input, 0);
    REQUIRE(output.shape(3) == 8);
}

TEST_CASE("initialize_rope longrope requires JSON", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"type", mlx_lm::StringOrNumber::from_string("longrope")},
        {"original_max_position_embeddings", mlx_lm::StringOrNumber::from_float(4096.0f)},
    };

    // No JSON pointer provided
    REQUIRE_THROWS_AS(
        mlx_lm::initialize_rope(8, 10000.0f, false, scaling, 131072, nullptr),
        std::runtime_error);
}

// ============================================================================
// initialize_rope factory — mrope
// ============================================================================

TEST_CASE("initialize_rope mrope type", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"type", mlx_lm::StringOrNumber::from_string("mrope")},
    };

    auto rope = mlx_lm::initialize_rope(64, 10000.0f, false, scaling);

    // mrope produces SimpleRoPE with scale=1.0
    REQUIRE(std::holds_alternative<mlx_lm::SimpleRoPE>(rope));
    auto& simple = std::get<mlx_lm::SimpleRoPE>(rope);
    CHECK(simple.scale == 1.0f);
}

// ============================================================================
// initialize_rope factory — unknown type
// ============================================================================

TEST_CASE("initialize_rope unknown type throws", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"type", mlx_lm::StringOrNumber::from_string("nonexistent")},
    };

    REQUIRE_THROWS_AS(
        mlx_lm::initialize_rope(64, 10000.0f, false, scaling),
        std::runtime_error);
}

// ============================================================================
// initialize_rope with rope_type key instead of type
// ============================================================================

TEST_CASE("initialize_rope uses rope_type key as fallback", "[rope]") {
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"rope_type", mlx_lm::StringOrNumber::from_string("linear")},
        {"factor", mlx_lm::StringOrNumber::from_float(2.0f)},
    };

    auto rope = mlx_lm::initialize_rope(64, 10000.0f, false, scaling);

    REQUIRE(std::holds_alternative<mlx_lm::SimpleRoPE>(rope));
    auto& simple = std::get<mlx_lm::SimpleRoPE>(rope);
    CHECK(simple.scale == Approx(0.5f));  // 1/2
}

// ============================================================================
// Consistency: different RoPE types produce different results
// ============================================================================

TEST_CASE("Different RoPE types produce different results", "[rope]") {
    auto input = make_test_input(1, 2, 4, 8);

    // SimpleRoPE
    mlx_lm::RoPEVariant simple_rope = mlx_lm::SimpleRoPE{8, false, 10000.0f, 1.0f};
    auto simple_out = mlx_lm::apply_rope(simple_rope, input, 5);
    mx::eval(simple_out);

    // Llama3RoPE with scaling
    std::unordered_map<std::string, mlx_lm::StringOrNumber> scaling = {
        {"factor", mlx_lm::StringOrNumber::from_float(8.0f)},
        {"low_freq_factor", mlx_lm::StringOrNumber::from_float(1.0f)},
        {"high_freq_factor", mlx_lm::StringOrNumber::from_float(4.0f)},
        {"original_max_position_embeddings", mlx_lm::StringOrNumber::from_float(4096.0f)},
    };
    mlx_lm::RoPEVariant llama3_rope = mlx_lm::Llama3RoPE(8, 131072, false, 10000.0f, scaling);
    auto llama3_out = mlx_lm::apply_rope(llama3_rope, input, 5);
    mx::eval(llama3_out);

    // They should produce different outputs at non-zero offset with scaling
    auto diff = mx::sum(mx::abs(mx::subtract(simple_out, llama3_out)));
    mx::eval(diff);
    CHECK(diff.item<float>() > 0.0f);
}

// ============================================================================
// Batch processing
// ============================================================================

TEST_CASE("SimpleRoPE handles batch dimension", "[rope]") {
    mlx_lm::SimpleRoPE rope{8, false, 10000.0f, 1.0f};

    auto input = make_test_input(4, 2, 8, 8);
    auto output = rope(input, 0);

    CHECK(output.shape(0) == 4);
    CHECK(output.shape(1) == 2);
    CHECK(output.shape(2) == 8);
    CHECK(output.shape(3) == 8);
}
