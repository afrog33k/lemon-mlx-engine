// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/chat.h>
#include <mlx-lm/common/types.h>
#include <string>
#include <variant>
#include <vector>

namespace mlx_lm {

// Container for raw user input before processing.
struct UserInput {

    // Prompt types.
    struct TextPrompt {
        std::string text;
    };
    struct MessagesPrompt {
        std::vector<Message> messages;
    };
    struct ChatPrompt {
        std::vector<chat::ChatMessage> messages;
    };

    using Prompt = std::variant<TextPrompt, MessagesPrompt, ChatPrompt>;

    // Image source — file path or raw pixel data.
    struct Image {
        std::string file_path;        // load from file
        // OR:
        std::vector<uint8_t> data;    // raw RGBA pixels
        int width = 0;
        int height = 0;
    };

    // Video source — file path.
    struct Video {
        std::string file_path;
    };

    // Processing options.
    struct Processing {
        int resize_width = 0;
        int resize_height = 0;
        Processing() : resize_width(0), resize_height(0) {}
        Processing(int w, int h) : resize_width(w), resize_height(h) {}
    };

    Prompt prompt;
    std::vector<Image> images;
    std::vector<Video> videos;
    Processing processing;

    // Convenience constructors.
    UserInput() : prompt(TextPrompt{""}) {}

    explicit UserInput(const std::string& text)
        : prompt(ChatPrompt{{chat::ChatMessage::user(text)}}) {}

    UserInput(const std::string& text, std::vector<Image> images)
        : prompt(ChatPrompt{{chat::ChatMessage::user(text)}}),
          images(std::move(images)) {}

    explicit UserInput(std::vector<chat::ChatMessage> chat_messages,
                       Processing proc = {})
        : prompt(ChatPrompt{std::move(chat_messages)}),
          processing(proc) {}

    explicit UserInput(std::vector<Message> messages,
                       std::vector<Image> images = {},
                       std::vector<Video> videos = {})
        : prompt(MessagesPrompt{std::move(messages)}),
          images(std::move(images)),
          videos(std::move(videos)) {}
};

// UserInputProcessor — CRTP base, no virtual functions.
// Derived must implement: LMInput prepare_impl(const UserInput& input)
template <typename Derived>
class UserInputProcessor {
public:
    LMInput prepare(const UserInput& input) {
        return static_cast<Derived*>(this)->prepare_impl(input);
    }
};

} // namespace mlx_lm
