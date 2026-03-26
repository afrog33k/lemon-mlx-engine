// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/vlm/qwen_vl_utils.h>
#include <algorithm>
#include <cmath>

namespace mx = mlx::core;

namespace mlx_lm {
namespace qwen_vl {

mx::array rotate_half(const mx::array& x) {
    // Split along last axis: x1 = x[..., :half], x2 = x[..., half:]
    int half = x.shape(-1) / 2;

    mx::Shape start_1(x.ndim(), 0);
    mx::Shape stop_1(x.shape().begin(), x.shape().end());
    stop_1.back() = half;

    mx::Shape start_2(x.ndim(), 0);
    start_2.back() = half;
    mx::Shape stop_2(x.shape().begin(), x.shape().end());

    auto x1 = mx::slice(x, start_1, stop_1);
    auto x2 = mx::slice(x, start_2, stop_2);

    return mx::concatenate({mx::negative(x2), x1}, -1);
}

mx::array merge_input_ids_with_image_features(
    const mx::array& input_ids,
    const mx::array& input_embeds,
    const mx::array& image_features,
    int image_token_id,
    int video_token_id)
{
    // Flatten input ids and find image/video token positions
    auto ids_flat = mx::flatten(input_ids);
    mx::eval(ids_flat);

    std::vector<int> image_indices;
    auto ids_data = ids_flat.data<int>();
    for (int i = 0; i < ids_flat.shape(0); ++i) {
        int v = ids_data[i];
        if (v == image_token_id || v == video_token_id) {
            image_indices.push_back(i);
        }
    }

    // Ensure result is 3D [1, L, D]
    auto result = input_embeds;
    if (result.ndim() == 2) {
        result = mx::reshape(result, {1, result.shape(0), result.shape(1)});
    }

    if (image_indices.empty()) {
        return result;
    }

    // Ensure image features are 3D [1, N, D]
    auto features = image_features;
    if (features.ndim() == 2) {
        features = mx::reshape(features, {1, features.shape(0), features.shape(1)});
    }

    // Scatter image features into the result at the image/video token positions.
    // Equivalent to Swift: result[0..., MLXArray(imageIndices), 0...] = features
    auto idx_arr = mx::array(image_indices.data(),
                              {static_cast<int>(image_indices.size())},
                              mx::int32);

    // Use take/put pattern: gather rows from features, scatter into result
    // along the sequence dimension (axis 1).
    for (size_t i = 0; i < image_indices.size(); ++i) {
        int idx = image_indices[i];
        auto feat_slice = mx::slice(features,
                                     {0, static_cast<int>(i), 0},
                                     {1, static_cast<int>(i) + 1, features.shape(-1)});
        result = mx::slice_update(result, feat_slice,
                                   mx::Shape{0, idx, 0},
                                   mx::Shape{1, idx + 1, result.shape(-1)});
    }

    return result;
}

VisionRotaryEmbedding::VisionRotaryEmbedding(int dimensions, float theta)
    : dimensions_(dimensions), theta_(theta),
      inverse_freq_(mx::divide(
          mx::array(1.0f),
          mx::power(mx::array(theta),
                    mx::divide(
                        mx::astype(mx::arange(0, dimensions, 2), mx::float32),
                        mx::array(static_cast<float>(dimensions))))))
{
}

mx::array VisionRotaryEmbedding::operator()(int sequence_length) const {
    auto seq = mx::astype(mx::arange(0, sequence_length), inverse_freq_.dtype());
    // Outer product: seq [S] x inverse_freq [D/2] -> [S, D/2]
    auto freqs = mx::matmul(
        mx::reshape(seq, {sequence_length, 1}),
        mx::reshape(inverse_freq_, {1, dimensions_ / 2}));
    return freqs;
}

std::pair<int, int> target_size(
    int height, int width, int factor, int min_pixels, int max_pixels)
{
    if (height < factor) {
        throw std::runtime_error(
            "Height " + std::to_string(height) +
            " must be >= factor " + std::to_string(factor));
    }
    if (width < factor) {
        throw std::runtime_error(
            "Width " + std::to_string(width) +
            " must be >= factor " + std::to_string(factor));
    }
    if (std::max(height, width) / std::min(height, width) > 200) {
        throw std::runtime_error(
            "Absolute aspect ratio must be smaller than 200: "
            + std::to_string(width) + " x " + std::to_string(height));
    }

    int h_bar = std::max(factor,
        static_cast<int>(std::round(static_cast<float>(height) / factor)) * factor);
    int w_bar = std::max(factor,
        static_cast<int>(std::round(static_cast<float>(width) / factor)) * factor);

    // Scale based on total pixel count
    if (h_bar * w_bar > max_pixels) {
        float beta = std::sqrt(
            static_cast<float>(height * width) / static_cast<float>(max_pixels));
        h_bar = static_cast<int>(
            std::floor(static_cast<float>(height) / beta / factor)) * factor;
        w_bar = static_cast<int>(
            std::floor(static_cast<float>(width) / beta / factor)) * factor;
    } else if (h_bar * w_bar < min_pixels) {
        float beta = std::sqrt(
            static_cast<float>(min_pixels) / static_cast<float>(height * width));
        h_bar = static_cast<int>(
            std::ceil(static_cast<float>(height) * beta / factor)) * factor;
        w_bar = static_cast<int>(
            std::ceil(static_cast<float>(width) * beta / factor)) * factor;
    }

    // Ensure dimensions are divisible by the factor
    h_bar = (h_bar / factor) * factor;
    w_bar = (w_bar / factor) * factor;

    if (h_bar <= 0 || w_bar <= 0) {
        throw std::runtime_error(
            "Invalid target dimensions: "
            + std::to_string(w_bar) + " x " + std::to_string(h_bar));
    }

    return {h_bar, w_bar};
}

std::pair<mx::array, THW> patchify(
    const std::vector<mx::array>& images,
    int merge_size, int patch_size, int temporal_patch_size)
{
    if (images.empty()) {
        throw std::runtime_error("No images in video sequence");
    }

    int resized_height = images[0].shape(-2);
    int resized_width = images[0].shape(-1);

    // Concatenate all frames along axis 0
    auto patches = mx::concatenate(images, 0);

    // Pad to match temporal patch size if needed
    int mod = patches.shape(0) % temporal_patch_size;
    if (mod != 0) {
        // Repeat last frame to pad — equivalent to Swift tiled(lastPatch, ...)
        int last_idx = patches.shape(0) - 1;
        auto last = mx::slice(patches, {last_idx, 0, 0, 0},
                               {last_idx + 1, patches.shape(1),
                                patches.shape(2), patches.shape(3)});
        int pad_count = temporal_patch_size - mod;
        auto padding = mx::repeat(last, pad_count, 0);
        patches = mx::concatenate({patches, padding}, 0);
    }

    int channel = patches.shape(1);
    int grid_t = patches.shape(0) / temporal_patch_size;
    int grid_h = resized_height / patch_size;
    int grid_w = resized_width / patch_size;

    // Reshape: [T*tps, C, H, W] -> [gridT, tps, C, gridH/ms, ms, ps, gridW/ms, ms, ps]
    patches = mx::reshape(patches, {
        grid_t, temporal_patch_size, channel,
        grid_h / merge_size, merge_size, patch_size,
        grid_w / merge_size, merge_size, patch_size
    });

    // Transpose to group spatial merge dims together
    patches = mx::transpose(patches, {0, 3, 6, 4, 7, 2, 1, 5, 8});

    // Flatten to [gridT * gridH * gridW, C * tps * ps * ps]
    auto flattened = mx::reshape(patches, {
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size
    });

    return {flattened, THW(grid_t, grid_h, grid_w)};
}

} // namespace qwen_vl
} // namespace mlx_lm
