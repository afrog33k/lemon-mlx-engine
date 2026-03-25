// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/vlm/media_processing.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Note: stb_image_resize2 would be ideal but we implement simple bilinear resize
// using MLX operations for GPU acceleration.

#include <cmath>
#include <stdexcept>

namespace mlx_lm {
namespace media_processing {

namespace mx = mlx::core;

mlx::core::array load_image(const std::string& path) {
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    // Convert to float32 array [H, W, 3]
    auto image = mx::array(
        reinterpret_cast<const uint8_t*>(data),
        {height, width, 3},
        mx::uint8);
    stbi_image_free(data);

    // Convert to float and scale to [0, 1]
    image = mx::astype(image, mx::float32);
    image = mx::multiply(image, mx::array(1.0f / 255.0f));

    // Reshape to [1, C, H, W] (NCHW format)
    image = mx::transpose(mx::reshape(image, {1, height, width, 3}), {0, 3, 1, 2});

    return image;
}

mlx::core::array resize_image(const mlx::core::array& image, int target_height, int target_width) {
    // image: [1, C, H, W] or [C, H, W]
    bool had_batch = (image.ndim() == 4);
    auto img = had_batch ? image : mx::reshape(image, {1, image.shape(0), image.shape(1), image.shape(2)});

    int C = img.shape(1);
    int H = img.shape(2);
    int W = img.shape(3);

    if (H == target_height && W == target_width) {
        return image;
    }

    // Convert to [1, H, W, C] for easier spatial operations
    img = mx::transpose(img, {0, 2, 3, 1});

    // Create coordinate grids for bilinear interpolation
    auto y_coords = mx::arange(0, target_height, mx::float32);
    y_coords = mx::multiply(y_coords, mx::array(static_cast<float>(H - 1) / static_cast<float>(target_height - 1)));

    auto x_coords = mx::arange(0, target_width, mx::float32);
    x_coords = mx::multiply(x_coords, mx::array(static_cast<float>(W - 1) / static_cast<float>(target_width - 1)));

    // Compute floor and ceil indices
    auto y0 = mx::astype(mx::floor(y_coords), mx::int32);
    auto y1 = mx::minimum(mx::add(y0, mx::array(1)), mx::array(H - 1));
    auto x0 = mx::astype(mx::floor(x_coords), mx::int32);
    auto x1 = mx::minimum(mx::add(x0, mx::array(1)), mx::array(W - 1));

    // Compute interpolation weights
    auto wy = mx::subtract(y_coords, mx::astype(y0, mx::float32));
    auto wx = mx::subtract(x_coords, mx::astype(x0, mx::float32));

    // Flatten image for gather
    img = mx::reshape(img, {H * W, C});

    // For each output pixel, compute bilinear interpolation
    // This is a simplified approach; for production, consider using MLX's built-in resize if available

    // Compute linear indices for the 4 corner pixels
    auto idx_00 = mx::add(mx::multiply(y0, mx::array(W)), mx::reshape(x0, {1, target_width}));
    auto idx_01 = mx::add(mx::multiply(y0, mx::array(W)), mx::reshape(x1, {1, target_width}));
    auto idx_10 = mx::add(mx::multiply(y1, mx::array(W)), mx::reshape(x0, {1, target_width}));
    auto idx_11 = mx::add(mx::multiply(y1, mx::array(W)), mx::reshape(x1, {1, target_width}));

    // Reshape indices for broadcasting
    idx_00 = mx::reshape(mx::flatten(idx_00), {-1});
    idx_01 = mx::reshape(mx::flatten(idx_01), {-1});
    idx_10 = mx::reshape(mx::flatten(idx_10), {-1});
    idx_11 = mx::reshape(mx::flatten(idx_11), {-1});

    // Gather corner values
    auto v00 = mx::take(img, idx_00, 0);
    auto v01 = mx::take(img, idx_01, 0);
    auto v10 = mx::take(img, idx_10, 0);
    auto v11 = mx::take(img, idx_11, 0);

    // Reshape for broadcasting
    v00 = mx::reshape(v00, {target_height, target_width, C});
    v01 = mx::reshape(v01, {target_height, target_width, C});
    v10 = mx::reshape(v10, {target_height, target_width, C});
    v11 = mx::reshape(v11, {target_height, target_width, C});

    // Compute weights for broadcasting
    auto wy_r = mx::reshape(wy, {target_height, 1, 1});
    auto wx_r = mx::reshape(wx, {1, target_width, 1});
    auto one_minus_wy = mx::subtract(mx::array(1.0f), wy_r);
    auto one_minus_wx = mx::subtract(mx::array(1.0f), wx_r);

    // Bilinear interpolation
    auto result = mx::add(
        mx::add(
            mx::multiply(mx::multiply(one_minus_wy, one_minus_wx), v00),
            mx::multiply(mx::multiply(one_minus_wy, wx_r), v01)),
        mx::add(
            mx::multiply(mx::multiply(wy_r, one_minus_wx), v10),
            mx::multiply(mx::multiply(wy_r, wx_r), v11)));

    // Back to [1, C, H, W]
    result = mx::transpose(mx::reshape(result, {1, target_height, target_width, C}), {0, 3, 1, 2});

    return had_batch ? result : mx::reshape(result, {C, target_height, target_width});
}

mlx::core::array normalize(
    const mlx::core::array& image,
    const std::vector<float>& mean,
    const std::vector<float>& std_dev)
{
    // image: [1, C, H, W] or [C, H, W]
    bool has_batch = (image.ndim() == 4);
    int C = has_batch ? image.shape(1) : image.shape(0);

    mx::Shape mean_shape = has_batch ? mx::Shape{1, C, 1, 1} : mx::Shape{C, 1, 1};
    mx::Shape std_shape = has_batch ? mx::Shape{1, C, 1, 1} : mx::Shape{C, 1, 1};
    auto mean_arr = mx::reshape(mx::array(mean.data(), {C}, mx::float32), mean_shape);
    auto std_arr = mx::reshape(mx::array(std_dev.data(), {C}, mx::float32), std_shape);

    return mx::divide(mx::subtract(image, mean_arr), std_arr);
}

mlx::core::array center_crop(const mlx::core::array& image, int target_height, int target_width) {
    // image: [1, C, H, W]
    int H = image.shape(2);
    int W = image.shape(3);

    if (H <= target_height && W <= target_width) {
        return image;
    }

    int crop_h = std::min(H, target_height);
    int crop_w = std::min(W, target_width);
    int y_start = (H - crop_h) / 2;
    int x_start = (W - crop_w) / 2;

    return mx::slice(image,
        {0, 0, y_start, x_start},
        {image.shape(0), image.shape(1), y_start + crop_h, x_start + crop_w});
}

mlx::core::array pad_to_square(const mlx::core::array& image) {
    // image: [1, C, H, W]
    int H = image.shape(2);
    int W = image.shape(3);
    int side = std::max(H, W);

    if (H == W) return image;

    int pad_h = (side - H) / 2;
    int pad_w = (side - W) / 2;

    auto result = mx::zeros({image.shape(0), image.shape(1), side, side}, image.dtype());
    // Would need scatter or pad operation — for now use mx::pad
    return mx::pad(image, {{0, 0}, {0, 0}, {pad_h, side - H - pad_h}, {pad_w, side - W - pad_w}});
}

mlx::core::array rescale(const mlx::core::array& image, float scale) {
    return mx::multiply(mx::astype(image, mx::float32), mx::array(scale));
}

mlx::core::array from_raw_pixels(
    const uint8_t* data, int width, int height, int channels)
{
    auto image = mx::array(data, {height, width, channels}, mx::uint8);
    image = mx::astype(image, mx::float32);
    image = mx::multiply(image, mx::array(1.0f / 255.0f));
    image = mx::transpose(mx::reshape(image, {1, height, width, channels}), {0, 3, 1, 2});
    return image;
}

} // namespace media_processing
} // namespace mlx_lm
