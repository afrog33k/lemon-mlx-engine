// OpenAI-compatible HTTP server for mlx-lm inference.
// Uses cpp-httplib for HTTP and SSE streaming.
// Supports single-model and multi-model (auto-load) modes.
#pragma once

#include <mlx-lm/common/generate_params.h>
#include <mlx-lm/common/model_container.h>
#include <mlx-lm/common/model_manager.h>
#include <memory>
#include <string>

// Forward declare httplib types to avoid header in public API.
namespace httplib { class Server; }

namespace mlx_lm {

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;

    // Default generation parameters (overridden per-request).
    GenerateParameters default_params;
};

class Server {
public:
    // Single-model mode (backward compatible).
    explicit Server(std::shared_ptr<ModelContainer> model,
                    ServerConfig config = {});

    // Multi-model mode with auto-load.
    explicit Server(std::shared_ptr<ModelManager> manager,
                    ServerConfig config = {});

    ~Server();

    // Non-copyable, non-movable (owns httplib::Server via pImpl).
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Start listening (blocks until stop() is called or signal).
    void start();

    // Stop the server from another thread.
    void stop();

    int port() const { return config_.port; }
    const std::string& host() const { return config_.host; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ServerConfig config_;

    void setup_routes();
};

} // namespace mlx_lm
