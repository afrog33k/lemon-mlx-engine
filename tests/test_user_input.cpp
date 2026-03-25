// Tests for UserInput, Chat messages, and MessageGenerators.
//
// Ports of Swift UserInputTests: testStandardConversion, plus additional
// coverage for UserInput construction, ChatMessage factory methods, and
// the NoSystemMessageGenerator.
//
// Note: Qwen2VL-specific tests from the Swift suite are skipped because
// the C++ port does not yet have a Qwen2VLMessageGenerator.

#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/chat.h>
#include <mlx-lm/common/user_input.h>

// ===== ChatMessage tests =====

TEST_CASE("ChatMessage factory methods", "[user_input]") {
    SECTION("user message") {
        auto msg = mlx_lm::chat::ChatMessage::user("Hello");
        REQUIRE(msg.role == mlx_lm::chat::Role::User);
        REQUIRE(msg.content == "Hello");
    }

    SECTION("system message") {
        auto msg = mlx_lm::chat::ChatMessage::system("You are helpful.");
        REQUIRE(msg.role == mlx_lm::chat::Role::System);
        REQUIRE(msg.content == "You are helpful.");
    }

    SECTION("assistant message") {
        auto msg = mlx_lm::chat::ChatMessage::assistant("Sure, I can help.");
        REQUIRE(msg.role == mlx_lm::chat::Role::Assistant);
        REQUIRE(msg.content == "Sure, I can help.");
    }

    SECTION("tool message") {
        auto msg = mlx_lm::chat::ChatMessage::tool("{\"result\": 42}");
        REQUIRE(msg.role == mlx_lm::chat::Role::Tool);
        REQUIRE(msg.content == "{\"result\": 42}");
    }
}

// ===== Role string conversion tests =====

TEST_CASE("Role to/from string roundtrip", "[user_input]") {
    REQUIRE(mlx_lm::chat::role_to_string(mlx_lm::chat::Role::User) == "user");
    REQUIRE(mlx_lm::chat::role_to_string(mlx_lm::chat::Role::Assistant) == "assistant");
    REQUIRE(mlx_lm::chat::role_to_string(mlx_lm::chat::Role::System) == "system");
    REQUIRE(mlx_lm::chat::role_to_string(mlx_lm::chat::Role::Tool) == "tool");

    REQUIRE(mlx_lm::chat::role_from_string("user") == mlx_lm::chat::Role::User);
    REQUIRE(mlx_lm::chat::role_from_string("assistant") == mlx_lm::chat::Role::Assistant);
    REQUIRE(mlx_lm::chat::role_from_string("system") == mlx_lm::chat::Role::System);
    REQUIRE(mlx_lm::chat::role_from_string("tool") == mlx_lm::chat::Role::Tool);

    // Unknown role defaults to User.
    REQUIRE(mlx_lm::chat::role_from_string("unknown") == mlx_lm::chat::Role::User);
}

// ===== DefaultMessageGenerator tests =====

// Port of Swift UserInputTests.testStandardConversion — verifies that
// DefaultMessageGenerator converts ChatMessages to the expected
// {"role": ..., "content": ...} format.
TEST_CASE("DefaultMessageGenerator standard conversion", "[user_input]") {
    using namespace mlx_lm;
    using namespace mlx_lm::chat;

    std::vector<ChatMessage> chat_messages = {
        ChatMessage::system("You are a useful agent."),
        ChatMessage::user("Tell me a story."),
    };

    DefaultMessageGenerator gen;
    auto messages = gen.generate(chat_messages);

    REQUIRE(messages.size() == 2);

    // First message: system.
    CHECK(messages[0].at("role") == "system");
    CHECK(messages[0].at("content") == "You are a useful agent.");

    // Second message: user.
    CHECK(messages[1].at("role") == "user");
    CHECK(messages[1].at("content") == "Tell me a story.");
}

// Test single message generation.
TEST_CASE("DefaultMessageGenerator single message", "[user_input]") {
    mlx_lm::DefaultMessageGenerator gen;
    auto msg = mlx_lm::chat::ChatMessage::assistant("Here is your answer.");
    auto result = gen.generate(msg);

    REQUIRE(result.at("role") == "assistant");
    REQUIRE(result.at("content") == "Here is your answer.");
}

// ===== NoSystemMessageGenerator tests =====

TEST_CASE("NoSystemMessageGenerator filters system messages", "[user_input]") {
    using namespace mlx_lm;
    using namespace mlx_lm::chat;

    std::vector<ChatMessage> chat_messages = {
        ChatMessage::system("You are a useful agent."),
        ChatMessage::user("Tell me a story."),
        ChatMessage::assistant("Once upon a time..."),
    };

    NoSystemMessageGenerator gen;
    auto messages = gen.generate(chat_messages);

    // System message should be filtered out.
    REQUIRE(messages.size() == 2);
    CHECK(messages[0].at("role") == "user");
    CHECK(messages[0].at("content") == "Tell me a story.");
    CHECK(messages[1].at("role") == "assistant");
    CHECK(messages[1].at("content") == "Once upon a time...");
}

TEST_CASE("NoSystemMessageGenerator with no system messages", "[user_input]") {
    using namespace mlx_lm;
    using namespace mlx_lm::chat;

    std::vector<ChatMessage> chat_messages = {
        ChatMessage::user("Hello"),
        ChatMessage::assistant("Hi there!"),
    };

    NoSystemMessageGenerator gen;
    auto messages = gen.generate(chat_messages);

    // All messages should be preserved.
    REQUIRE(messages.size() == 2);
}

// ===== UserInput construction tests =====

TEST_CASE("UserInput default constructor", "[user_input]") {
    mlx_lm::UserInput input;
    REQUIRE(input.images.empty());
    REQUIRE(input.videos.empty());

    // Default prompt is a TextPrompt with empty string.
    REQUIRE(std::holds_alternative<mlx_lm::UserInput::TextPrompt>(input.prompt));
}

TEST_CASE("UserInput from text string", "[user_input]") {
    mlx_lm::UserInput input("What is MLX?");

    // Should create a ChatPrompt with a single user message.
    REQUIRE(std::holds_alternative<mlx_lm::UserInput::ChatPrompt>(input.prompt));
    auto& chat_prompt = std::get<mlx_lm::UserInput::ChatPrompt>(input.prompt);
    REQUIRE(chat_prompt.messages.size() == 1);
    CHECK(chat_prompt.messages[0].role == mlx_lm::chat::Role::User);
    CHECK(chat_prompt.messages[0].content == "What is MLX?");
    REQUIRE(input.images.empty());
}

TEST_CASE("UserInput from text with images", "[user_input]") {
    mlx_lm::UserInput::Image img;
    img.file_path = "/path/to/image.png";

    mlx_lm::UserInput input("What is this?", {img});

    REQUIRE(std::holds_alternative<mlx_lm::UserInput::ChatPrompt>(input.prompt));
    REQUIRE(input.images.size() == 1);
    CHECK(input.images[0].file_path == "/path/to/image.png");
}

TEST_CASE("UserInput from chat messages", "[user_input]") {
    using namespace mlx_lm::chat;

    std::vector<ChatMessage> messages = {
        ChatMessage::system("You are helpful."),
        ChatMessage::user("Hello!"),
    };

    mlx_lm::UserInput input(messages);

    REQUIRE(std::holds_alternative<mlx_lm::UserInput::ChatPrompt>(input.prompt));
    auto& chat_prompt = std::get<mlx_lm::UserInput::ChatPrompt>(input.prompt);
    REQUIRE(chat_prompt.messages.size() == 2);
    CHECK(chat_prompt.messages[0].role == Role::System);
    CHECK(chat_prompt.messages[1].role == Role::User);
}

TEST_CASE("UserInput from raw messages", "[user_input]") {
    std::vector<mlx_lm::Message> messages = {
        {{"role", "user"}, {"content", "Hi"}},
    };

    mlx_lm::UserInput input(messages);

    REQUIRE(std::holds_alternative<mlx_lm::UserInput::MessagesPrompt>(input.prompt));
    auto& msg_prompt = std::get<mlx_lm::UserInput::MessagesPrompt>(input.prompt);
    REQUIRE(msg_prompt.messages.size() == 1);
    CHECK(msg_prompt.messages[0].at("role") == "user");
    CHECK(msg_prompt.messages[0].at("content") == "Hi");
}
