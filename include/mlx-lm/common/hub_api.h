// Copyright © 2025 — Ported to C++
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace mlx_lm {

// Progress callback for downloads.
using ProgressCallback = std::function<void(size_t bytes_downloaded, size_t total_bytes)>;

// Hugging Face Hub API client using libcurl.
class HubApi {
public:
    // Set the API token for authentication.
    void set_token(const std::string& token);

    // Get cache directory (defaults to ~/.cache/huggingface/hub).
    std::string cache_dir() const;

    // Set custom cache directory.
    void set_cache_dir(const std::string& dir);

    // Download a single file from a HF repository.
    // Returns the local path to the downloaded file.
    std::string download_file(
        const std::string& repo_id,
        const std::string& filename,
        const std::string& revision = "main",
        ProgressCallback progress = nullptr);

    // Download a model snapshot (all required files).
    // Returns the local directory path.
    std::string snapshot_download(
        const std::string& repo_id,
        const std::string& revision = "main",
        const std::vector<std::string>& allow_patterns = {},
        ProgressCallback progress = nullptr);

    // Check if a model exists locally.
    bool is_cached(const std::string& repo_id,
                   const std::string& revision = "main") const;

    // Get the local path for a cached model.
    std::string model_directory(const std::string& repo_id,
                                const std::string& revision = "main") const;

    // Shared default instance.
    static HubApi& shared();

private:
    std::string token_;
    std::string cache_dir_;

    // HTTP helpers
    std::string http_get(const std::string& url,
                         const std::string& output_path = "",
                         ProgressCallback progress = nullptr);
    std::string resolve_cache_path(const std::string& repo_id,
                                   const std::string& revision) const;
};

} // namespace mlx_lm
