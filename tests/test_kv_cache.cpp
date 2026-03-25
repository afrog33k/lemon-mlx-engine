// Tests for KV cache

#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/kv_cache.h>
#include <mlx/mlx.h>

TEST_CASE("KVCacheSimple basic update", "[kvcache]") {
    mlx_lm::KVCacheSimple cache;
    REQUIRE(cache.offset() == 0);
    REQUIRE(!cache.max_size().has_value());

    auto keys = mlx::core::ones({1, 4, 3, 64});   // [B, heads, seq, dim]
    auto values = mlx::core::ones({1, 4, 3, 64});

    auto [k, v] = cache.update(keys, values);
    REQUIRE(cache.offset() == 3);
    REQUIRE(k.shape(2) == 3);
}

TEST_CASE("KVCacheSimple accumulates", "[kvcache]") {
    mlx_lm::KVCacheSimple cache;

    auto k1 = mlx::core::ones({1, 4, 5, 64});
    auto v1 = mlx::core::ones({1, 4, 5, 64});
    cache.update(k1, v1);
    REQUIRE(cache.offset() == 5);

    auto k2 = mlx::core::ones({1, 4, 3, 64});
    auto v2 = mlx::core::ones({1, 4, 3, 64});
    auto [k, v] = cache.update(k2, v2);
    REQUIRE(cache.offset() == 8);
    REQUIRE(k.shape(2) == 8);
}

TEST_CASE("KVCacheSimple trim", "[kvcache]") {
    mlx_lm::KVCacheSimple cache;

    auto keys = mlx::core::ones({1, 4, 10, 64});
    auto values = mlx::core::ones({1, 4, 10, 64});
    cache.update(keys, values);

    int trimmed = cache.trim(3);
    REQUIRE(trimmed == 3);
    REQUIRE(cache.offset() == 7);
}

TEST_CASE("RotatingKVCache basic", "[kvcache]") {
    mlx_lm::RotatingKVCache cache(16, 4);
    REQUIRE(cache.offset() == 0);
    REQUIRE(cache.max_size().value() == 16);
}

TEST_CASE("Type-erased KVCache", "[kvcache]") {
    mlx_lm::KVCache cache; // defaults to KVCacheSimple
    REQUIRE(cache.offset() == 0);
    REQUIRE(!cache.max_size().has_value());
    REQUIRE(cache.is_trimmable());
}

TEST_CASE("create_causal_mask", "[kvcache]") {
    auto mask = mlx_lm::create_causal_mask(4, 0);
    REQUIRE(mask.shape(0) == 4);
    REQUIRE(mask.shape(1) == 4);
}
