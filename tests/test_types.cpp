// Tests for core types

#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/types.h>
#include <mlx-lm/common/chat.h>
#include <mlx/mlx.h>

TEST_CASE("THW basic operations", "[types]") {
    mlx_lm::THW thw(2, 224, 224);
    REQUIRE(thw.t == 2);
    REQUIRE(thw.h == 224);
    REQUIRE(thw.w == 224);
    REQUIRE(thw.product() == 2 * 224 * 224);
}

TEST_CASE("LMInput construction", "[types]") {
    auto tokens = mlx::core::array({1, 2, 3, 4}, mlx::core::int32);
    mlx_lm::LMInput input(tokens);
    REQUIRE(input.text.tokens.size() == 4);
    REQUIRE(!input.image.has_value());
    REQUIRE(!input.video.has_value());
}

TEST_CASE("LMOutput construction", "[types]") {
    auto logits = mlx::core::zeros({1, 32000});
    mlx_lm::LMOutput output(logits);
    REQUIRE(output.logits.shape(1) == 32000);
    REQUIRE(!output.state.has_value());
}

TEST_CASE("PrepareResult variants", "[types]") {
    auto tokens = mlx::core::array({1, 2, 3}, mlx::core::int32);

    SECTION("tokens variant") {
        auto result = mlx_lm::PrepareResult::tokens(mlx_lm::LMInput::Text(tokens));
        REQUIRE(result.is_tokens());
        REQUIRE(!result.is_logits());
    }

    SECTION("logits variant") {
        auto logits = mlx::core::zeros({1, 1000});
        auto result = mlx_lm::PrepareResult::logits(mlx_lm::LMOutput(logits));
        REQUIRE(result.is_logits());
        REQUIRE(!result.is_tokens());
    }
}

TEST_CASE("Chat message creation", "[chat]") {
    auto msg = mlx_lm::chat::ChatMessage::user("Hello");
    REQUIRE(msg.role == mlx_lm::chat::Role::User);
    REQUIRE(msg.content == "Hello");

    auto sys = mlx_lm::chat::ChatMessage::system("You are helpful.");
    REQUIRE(sys.role == mlx_lm::chat::Role::System);
}

TEST_CASE("DefaultMessageGenerator", "[chat]") {
    mlx_lm::DefaultMessageGenerator gen;
    auto msg = mlx_lm::chat::ChatMessage::user("test");
    auto result = gen.generate(msg);
    REQUIRE(result["role"] == "user");
    REQUIRE(result["content"] == "test");
}

TEST_CASE("NoSystemMessageGenerator", "[chat]") {
    mlx_lm::NoSystemMessageGenerator gen;
    std::vector<mlx_lm::chat::ChatMessage> messages = {
        mlx_lm::chat::ChatMessage::system("sys"),
        mlx_lm::chat::ChatMessage::user("hello"),
    };
    auto result = gen.generate(messages);
    REQUIRE(result.size() == 1);
    REQUIRE(result[0]["role"] == "user");
}
