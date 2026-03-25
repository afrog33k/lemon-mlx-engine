// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace mlx_lm {

// ---------------------------------------------------------------------------
// JSONValue -- type-safe representation of JSON values
// ---------------------------------------------------------------------------

/// Thin wrapper around nlohmann::json that provides explicit type tagging.
/// In C++ we simply use nlohmann::json directly, which already handles
/// null, bool, int, double, string, array, and object types.
using JSONValue = nlohmann::json;

// Convenience: ToolSpec is a JSON object describing a tool schema.
using ToolSpec = nlohmann::json;

// ---------------------------------------------------------------------------
// ToolCall -- represents a single tool/function call extracted from model output
// ---------------------------------------------------------------------------

struct ToolCall {
    /// The function details for a tool call.
    struct Function {
        std::string name;
        nlohmann::json arguments;  // always an object ({...})

        Function() = default;
        Function(std::string name, nlohmann::json arguments)
            : name(std::move(name)), arguments(std::move(arguments)) {}
    };

    Function function;

    ToolCall() = default;
    explicit ToolCall(Function function) : function(std::move(function)) {}
};

// ---------------------------------------------------------------------------
// ToolParameterType -- describes a parameter's JSON-Schema type
// ---------------------------------------------------------------------------

struct ToolParameter;  // forward declaration

/// Recursive parameter type representation for JSON Schema generation.
struct ToolParameterType {
    enum Tag {
        kString,
        kBool,
        kInt,
        kDouble,
        kData,      // base64-encoded string
        kArray,     // has element_type
        kObject,    // has properties
    };

    Tag tag;

    /// For kArray -- the element type.
    std::shared_ptr<ToolParameterType> element_type;

    /// For kObject -- nested properties.
    std::shared_ptr<std::vector<ToolParameter>> properties;

    // Factory helpers ---
    static ToolParameterType string_type()  { return {kString, nullptr, nullptr}; }
    static ToolParameterType bool_type()    { return {kBool,   nullptr, nullptr}; }
    static ToolParameterType int_type()     { return {kInt,    nullptr, nullptr}; }
    static ToolParameterType double_type()  { return {kDouble, nullptr, nullptr}; }
    static ToolParameterType data_type()    { return {kData,   nullptr, nullptr}; }

    static ToolParameterType array_type(ToolParameterType elem) {
        return {kArray, std::make_shared<ToolParameterType>(std::move(elem)), nullptr};
    }

    static ToolParameterType object_type(std::vector<ToolParameter> props) {
        return {kObject, nullptr,
                std::make_shared<std::vector<ToolParameter>>(std::move(props))};
    }

    /// Convert to JSON Schema representation.
    nlohmann::json to_schema() const;
};

// ---------------------------------------------------------------------------
// ToolParameter -- a single parameter in a tool's schema
// ---------------------------------------------------------------------------

struct ToolParameter {
    std::string name;
    ToolParameterType type;
    std::string description;
    bool is_required;
    nlohmann::json extra_properties;  // additional schema fields

    /// Build the JSON Schema for this parameter.
    nlohmann::json to_schema() const;

    // Convenience constructors ---
    static ToolParameter required(
        const std::string& name,
        ToolParameterType type,
        const std::string& description,
        nlohmann::json extra = nlohmann::json::object());

    static ToolParameter optional(
        const std::string& name,
        ToolParameterType type,
        const std::string& description,
        nlohmann::json extra = nlohmann::json::object());
};

// ---------------------------------------------------------------------------
// Tool -- a callable tool with a JSON Schema spec and a handler
// ---------------------------------------------------------------------------

struct Tool {
    ToolSpec schema;
    std::function<nlohmann::json(const nlohmann::json&)> handler;

    /// The tool name extracted from the schema.
    std::string name() const;

    /// Construct from explicit parts.
    Tool(const std::string& name,
         const std::string& description,
         const std::vector<ToolParameter>& parameters,
         std::function<nlohmann::json(const nlohmann::json&)> handler);

    /// Construct from a raw schema + handler.
    Tool(ToolSpec schema,
         std::function<nlohmann::json(const nlohmann::json&)> handler);

    /// Execute the tool with a ToolCall (checks name match).
    nlohmann::json execute(const ToolCall& call) const;
};

// ---------------------------------------------------------------------------
// ToolCallParser -- abstract interface for parsing tool calls from text
// ---------------------------------------------------------------------------

class ToolCallParser {
public:
    virtual ~ToolCallParser() = default;

    /// The start tag that indicates a tool call is beginning.
    /// Returns std::nullopt for inline formats that don't use wrapper tags.
    virtual std::optional<std::string> start_tag() const = 0;

    /// The end tag that indicates a tool call has ended.
    /// Returns std::nullopt for inline formats that don't use wrapper tags.
    virtual std::optional<std::string> end_tag() const = 0;

    /// Parse the content into a ToolCall.
    /// @param content  The text content to parse (may include tags).
    /// @param tools    Optional tool schemas for type-aware parsing.
    /// @return A ToolCall if parsing succeeds, std::nullopt otherwise.
    virtual std::optional<ToolCall> parse(
        const std::string& content,
        const std::optional<nlohmann::json>& tools) const = 0;
};

// ---------------------------------------------------------------------------
// Concrete parsers
// ---------------------------------------------------------------------------

/// JSON format: <tag>{"name": "...", "arguments": {...}}</tag>
class JSONToolCallParser : public ToolCallParser {
public:
    JSONToolCallParser(std::string start, std::string end);

    std::optional<std::string> start_tag() const override;
    std::optional<std::string> end_tag() const override;
    std::optional<ToolCall> parse(
        const std::string& content,
        const std::optional<nlohmann::json>& tools) const override;

private:
    std::string start_tag_;
    std::string end_tag_;
};

/// GLM4 format: func<arg_key>k</arg_key><arg_value>v</arg_value>
class GLM4ToolCallParser : public ToolCallParser {
public:
    GLM4ToolCallParser() = default;

    std::optional<std::string> start_tag() const override;
    std::optional<std::string> end_tag() const override;
    std::optional<ToolCall> parse(
        const std::string& content,
        const std::optional<nlohmann::json>& tools) const override;
};

/// Gemma format: call:name{key:value,k:<escape>str<escape>}
class GemmaFunctionParser : public ToolCallParser {
public:
    GemmaFunctionParser() = default;

    std::optional<std::string> start_tag() const override;
    std::optional<std::string> end_tag() const override;
    std::optional<ToolCall> parse(
        const std::string& content,
        const std::optional<nlohmann::json>& tools) const override;

private:
    static constexpr const char* escape_marker_ = "<escape>";
};

/// XML function format: <function=name><parameter=key>value</parameter></function>
class XMLFunctionParser : public ToolCallParser {
public:
    XMLFunctionParser() = default;

    std::optional<std::string> start_tag() const override;
    std::optional<std::string> end_tag() const override;
    std::optional<ToolCall> parse(
        const std::string& content,
        const std::optional<nlohmann::json>& tools) const override;
};

/// Kimi K2 format: functions.name:0<|tool_call_argument_begin|>{JSON}
class KimiK2ToolCallParser : public ToolCallParser {
public:
    KimiK2ToolCallParser() = default;

    std::optional<std::string> start_tag() const override;
    std::optional<std::string> end_tag() const override;
    std::optional<ToolCall> parse(
        const std::string& content,
        const std::optional<nlohmann::json>& tools) const override;
};

/// MiniMax M2 format: <invoke name="f"><parameter name="k">v</parameter></invoke>
class MiniMaxM2ToolCallParser : public ToolCallParser {
public:
    MiniMaxM2ToolCallParser() = default;

    std::optional<std::string> start_tag() const override;
    std::optional<std::string> end_tag() const override;
    std::optional<ToolCall> parse(
        const std::string& content,
        const std::optional<nlohmann::json>& tools) const override;
};

// ---------------------------------------------------------------------------
// ToolCallFormat -- enum of supported tool call formats
// ---------------------------------------------------------------------------

enum class ToolCallFormat {
    json,
    lfm2,
    xml_function,
    glm4,
    gemma,
    kimi_k2,
    minimax_m2,
};

/// Convert a ToolCallFormat to its string representation.
std::string to_string(ToolCallFormat fmt);

/// Parse a ToolCallFormat from a string. Returns std::nullopt on failure.
std::optional<ToolCallFormat> tool_call_format_from_string(const std::string& s);

/// Create the appropriate parser for a given format.
std::unique_ptr<ToolCallParser> create_parser(ToolCallFormat fmt);

/// Infer the tool call format from a model_type string (from config.json).
/// Returns std::nullopt to use the default format.
std::optional<ToolCallFormat> infer_tool_call_format(const std::string& model_type);

// ---------------------------------------------------------------------------
// ToolCallProcessor -- streaming tool-call detection from generated text
// ---------------------------------------------------------------------------

class ToolCallProcessor {
public:
    /// Construct a processor for a specific format.
    /// @param format  The tool call format to detect (defaults to json).
    /// @param tools   Optional tool schemas for type-aware parsing.
    explicit ToolCallProcessor(
        ToolCallFormat format = ToolCallFormat::json,
        std::optional<nlohmann::json> tools = std::nullopt);

    /// Process a generated text chunk and extract any tool call content.
    /// @param chunk  The text chunk to process.
    /// @return Regular text that should be displayed (non-tool-call content),
    ///         or std::nullopt if the chunk was buffered internally.
    std::optional<std::string> process_chunk(const std::string& chunk);

    /// The tool calls extracted during processing.
    const std::vector<ToolCall>& tool_calls() const { return tool_calls_; }

    /// Mutable access to tool calls (e.g. to move out).
    std::vector<ToolCall>& tool_calls() { return tool_calls_; }

private:
    enum class State {
        normal,
        potential_tool_call,
        collecting_tool_call,
    };

    std::unique_ptr<ToolCallParser> parser_;
    std::optional<nlohmann::json> tools_;
    State state_ = State::normal;
    std::string buffer_;
    std::vector<ToolCall> tool_calls_;

    bool is_inline_format() const;
    std::optional<char> start_tag_first_char() const;

    std::optional<std::string> process_inline_chunk(const std::string& chunk);
    std::optional<std::string> process_tagged_chunk(const std::string& chunk);

    /// Separate a token from the buffer around a separator.
    /// If return_leading is true, returns text before separator and keeps
    /// separator+rest in buffer. Otherwise, returns text after separator
    /// and keeps buffer up to (including) separator.
    std::optional<std::string> separate_token(
        std::string& buffer, const std::string& separator, bool return_leading);

    /// Check if the buffer is a partial (prefix) match of the tag.
    bool partial_match(const std::string& buffer, const std::string& tag) const;
};

// ---------------------------------------------------------------------------
// Parser utility functions (used by multiple parsers)
// ---------------------------------------------------------------------------

namespace detail {

/// Attempt JSON deserialization of a string; returns the parsed JSON value
/// on success, or the original string as a JSON string on failure.
nlohmann::json deserialize(const std::string& value);

/// Check if a parameter is a "string" type in the tool schema.
bool is_string_type(
    const std::string& func_name,
    const std::string& arg_name,
    const std::optional<nlohmann::json>& tools);

/// Get the parameter type string from the tool schema.
std::optional<std::string> get_parameter_type(
    const std::string& func_name,
    const std::string& param_name,
    const std::optional<nlohmann::json>& tools);

/// Get the full parameter config (properties) for a function.
nlohmann::json get_parameter_config(
    const std::string& func_name,
    const std::optional<nlohmann::json>& tools);

/// Extract types from JSON schema (handles anyOf, oneOf, allOf, enums).
std::vector<std::string> extract_types_from_schema(const nlohmann::json& schema);

/// Convert a parameter value based on multiple possible types.
nlohmann::json convert_value_with_types(
    const std::string& value, const std::vector<std::string>& types);

/// Convert a parameter value based on schema type lookup.
nlohmann::json convert_parameter_value(
    const std::string& value,
    const std::string& param_name,
    const std::string& func_name,
    const std::optional<nlohmann::json>& tools);

/// Extract a name from a potentially quoted string.
std::string extract_name(const std::string& name_str);

}  // namespace detail

}  // namespace mlx_lm
