// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/safetensors.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mlx_lm {

std::unordered_map<std::string, mlx::core::array>
load_safetensors(const std::string& path) {
    // MLX C++ has built-in safetensors loading
    return mlx::core::load_safetensors(path).first;
}

std::unordered_map<std::string, mlx::core::array>
load_safetensors_from_directory(const std::string& directory) {
    std::unordered_map<std::string, mlx::core::array> all_weights;

    // Check for single model.safetensors
    auto single_path = fs::path(directory) / "model.safetensors";
    if (fs::exists(single_path)) {
        return load_safetensors(single_path.string());
    }

    // Check for sharded model
    auto index_path = fs::path(directory) / "model.safetensors.index.json";
    if (fs::exists(index_path)) {
        std::ifstream f(index_path);
        nlohmann::json index;
        f >> index;

        if (index.contains("weight_map")) {
            std::set<std::string> shard_files;
            for (auto& [key, val] : index["weight_map"].items()) {
                shard_files.insert(val.get<std::string>());
            }

            for (const auto& shard_name : shard_files) {
                auto shard_path = fs::path(directory) / shard_name;
                if (!fs::exists(shard_path)) {
                    throw std::runtime_error(
                        "Missing shard file: " + shard_path.string());
                }
                auto shard_weights = load_safetensors(shard_path.string());
                for (auto& [k, v] : shard_weights) {
                    all_weights.insert_or_assign(k, std::move(v));
                }
            }
            return all_weights;
        }
    }

    // Fallback: load all .safetensors files in the directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".safetensors") {
            auto weights = load_safetensors(entry.path().string());
            for (auto& [k, v] : weights) {
                all_weights.insert_or_assign(k, std::move(v));
            }
        }
    }

    if (all_weights.empty()) {
        throw std::runtime_error(
            "No .safetensors files found in " + directory);
    }

    return all_weights;
}

void load_weights(
    const std::string& model_directory,
    std::unordered_map<std::string, mlx::core::array*>& weight_map)
{
    auto all_weights = load_safetensors_from_directory(model_directory);

    for (auto& [name, target_ptr] : weight_map) {
        auto it = all_weights.find(name);
        if (it != all_weights.end()) {
            *target_ptr = std::move(it->second);
        }
    }
}

} // namespace mlx_lm
