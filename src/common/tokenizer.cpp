// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/tokenizer.h>
#include <tokenizers_cpp.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mlx_lm {

struct Tokenizer::Impl {
    std::unique_ptr<tokenizers::Tokenizer> tok;
};

Tokenizer::~Tokenizer() = default;

std::shared_ptr<Tokenizer> Tokenizer::from_directory(const std::string& model_dir) {
    auto json_path = fs::path(model_dir) / "tokenizer.json";
    if (!fs::exists(json_path)) {
        throw std::runtime_error("tokenizer.json not found in " + model_dir);
    }

    std::ifstream f(json_path);
    if (!f) {
        throw std::runtime_error("Failed to open " + json_path.string());
    }

    std::ostringstream ss;
    ss << f.rdbuf();
    return from_json_blob(ss.str());
}

std::shared_ptr<Tokenizer> Tokenizer::from_json_blob(const std::string& json_blob) {
    auto tokenizer = std::shared_ptr<Tokenizer>(new Tokenizer());
    tokenizer->impl_ = std::make_unique<Impl>();
    tokenizer->impl_->tok = tokenizers::Tokenizer::FromBlobJSON(json_blob);
    if (!tokenizer->impl_->tok) {
        throw std::runtime_error("Failed to create tokenizer from JSON blob");
    }
    return tokenizer;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    auto ids = impl_->tok->Encode(text);
    return std::vector<int>(ids.begin(), ids.end());
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::vector<int32_t> ids(token_ids.begin(), token_ids.end());
    return impl_->tok->Decode(ids);
}

size_t Tokenizer::vocab_size() const {
    return impl_->tok->GetVocabSize();
}

std::string Tokenizer::id_to_token(int token_id) const {
    return impl_->tok->IdToToken(static_cast<int32_t>(token_id));
}

int Tokenizer::token_to_id(const std::string& token) const {
    return static_cast<int>(impl_->tok->TokenToId(token));
}

} // namespace mlx_lm
