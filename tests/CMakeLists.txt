include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# Core tests
add_executable(test_types test_types.cpp)
target_link_libraries(test_types PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_types COMMAND test_types)

# KV Cache tests
add_executable(test_kv_cache test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_kv_cache COMMAND test_kv_cache)

# Config parsing tests
add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_config COMMAND test_config)

# Sampler tests
add_executable(test_generate test_generate.cpp)
target_link_libraries(test_generate PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_generate COMMAND test_generate)

# Tool calling tests
add_executable(test_tool_calling test_tool_calling.cpp)
target_link_libraries(test_tool_calling PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_tool_calling COMMAND test_tool_calling)

# Chat session tests
add_executable(test_chat_session test_chat_session.cpp)
target_link_libraries(test_chat_session PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_chat_session COMMAND test_chat_session)

# User input and message generator tests
add_executable(test_user_input test_user_input.cpp)
target_link_libraries(test_user_input PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_user_input COMMAND test_user_input)

# RoPE utility tests
add_executable(test_rope_utils test_rope_utils.cpp)
target_link_libraries(test_rope_utils PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_rope_utils COMMAND test_rope_utils)

# Chat template tests
add_executable(test_chat_template test_chat_template.cpp)
target_link_libraries(test_chat_template PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_chat_template COMMAND test_chat_template)

# NemotronH model tests
add_executable(test_nemotron_h test_nemotron_h.cpp)
target_link_libraries(test_nemotron_h PRIVATE mlx-lm-llm Catch2::Catch2WithMain)
add_test(NAME test_nemotron_h COMMAND test_nemotron_h)
