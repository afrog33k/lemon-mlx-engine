// Copyright © 2025 Apple Inc. — Ported to C++
#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mlx_lm {

// A single key-value map representing a message in model-specific format.
using Message = std::unordered_map<std::string, std::string>;

namespace chat {

enum class Role {
    User,
    Assistant,
    System,
    Tool,
};

inline std::string role_to_string(Role role) {
    switch (role) {
        case Role::User:      return "user";
        case Role::Assistant: return "assistant";
        case Role::System:    return "system";
        case Role::Tool:      return "tool";
    }
    return "user";
}

inline Role role_from_string(const std::string& s) {
    if (s == "assistant") return Role::Assistant;
    if (s == "system")    return Role::System;
    if (s == "tool")      return Role::Tool;
    return Role::User;
}

// A structured chat message.
struct ChatMessage {
    Role role;
    std::string content;
    // Image/video references are handled separately in UserInput.

    ChatMessage() = default;
    ChatMessage(Role role, std::string content)
        : role(role), content(std::move(content)) {}

    static ChatMessage system(const std::string& content) {
        return {Role::System, content};
    }
    static ChatMessage assistant(const std::string& content) {
        return {Role::Assistant, content};
    }
    static ChatMessage user(const std::string& content) {
        return {Role::User, content};
    }
    static ChatMessage tool(const std::string& content) {
        return {Role::Tool, content};
    }
};

} // namespace chat

// MessageGenerator: converts structured ChatMessages into model-specific Messages.
// CRTP base — no virtual functions.
template <typename Derived>
class MessageGenerator {
public:
    Message generate(const chat::ChatMessage& message) const {
        return static_cast<const Derived*>(this)->generate_impl(message);
    }

    std::vector<Message> generate(const std::vector<chat::ChatMessage>& messages) const {
        return static_cast<const Derived*>(this)->generate_messages_impl(messages);
    }
};

// Default message generator producing {"role": ..., "content": ...}.
class DefaultMessageGenerator : public MessageGenerator<DefaultMessageGenerator> {
public:
    Message generate_impl(const chat::ChatMessage& msg) const {
        return {
            {"role", chat::role_to_string(msg.role)},
            {"content", msg.content},
        };
    }

    std::vector<Message> generate_messages_impl(const std::vector<chat::ChatMessage>& messages) const {
        std::vector<Message> result;
        result.reserve(messages.size());
        for (const auto& msg : messages) {
            result.push_back(generate_impl(msg));
        }
        return result;
    }
};

// Message generator that omits system role messages.
class NoSystemMessageGenerator : public MessageGenerator<NoSystemMessageGenerator> {
public:
    Message generate_impl(const chat::ChatMessage& msg) const {
        return {
            {"role", chat::role_to_string(msg.role)},
            {"content", msg.content},
        };
    }

    std::vector<Message> generate_messages_impl(const std::vector<chat::ChatMessage>& messages) const {
        std::vector<Message> result;
        for (const auto& msg : messages) {
            if (msg.role != chat::Role::System) {
                result.push_back(generate_impl(msg));
            }
        }
        return result;
    }
};

} // namespace mlx_lm
