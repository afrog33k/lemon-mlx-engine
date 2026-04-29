// Compare the compiled MLX GDN recurrence with the opt-in HIP GDN kernel.

#include <mlx-lm/common/gated_delta.h>
#include <mlx/mlx.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace mx = mlx::core;

static float max_abs_diff(const mx::array& a, const mx::array& b) {
    auto d = mx::max(mx::abs(mx::subtract(
        mx::astype(mx::reshape(a, {-1}), mx::float32),
        mx::astype(mx::reshape(b, {-1}), mx::float32))));
    mx::eval(d);
    return d.item<float>();
}

static mx::array bounded_normal(const mx::Shape& shape, float scale = 0.05f) {
    return mx::astype(mx::multiply(mx::random::normal(shape), mx::array(scale)), mx::bfloat16);
}

static std::pair<mx::array, mx::array> run_common(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    bool hip)
{
    if (hip) {
        setenv("LEMON_MLX_GDN_ENABLE_HIP", "1", 1);
        unsetenv("LEMON_MLX_GDN_DISABLE_HIP");
    } else {
        unsetenv("LEMON_MLX_GDN_ENABLE_HIP");
        setenv("LEMON_MLX_GDN_DISABLE_HIP", "1", 1);
    }
    auto out = mlx_lm::gated_delta_ops(q, k, v, g, beta, state, std::nullopt);
    mx::eval({out.first, out.second});
    mx::synchronize();
    return out;
}

static bool run_case(
    const std::string& name,
    int B,
    int T,
    int Hk,
    int Hv,
    int Dk,
    int Dv,
    float tolerance)
{
    auto q = bounded_normal({B, T, Hk, Dk});
    auto k = bounded_normal({B, T, Hk, Dk});
    auto v = bounded_normal({B, T, Hv, Dv});
    auto g = mx::astype(mx::add(mx::array(0.98f), mx::multiply(mx::random::normal({B, T, Hv}), mx::array(0.005f))), mx::bfloat16);
    auto beta = mx::astype(mx::add(mx::array(0.50f), mx::multiply(mx::random::normal({B, T, Hv}), mx::array(0.01f))), mx::bfloat16);
    auto state = bounded_normal({B, Hv, Dv, Dk});
    mx::eval({q, k, v, g, beta, state});
    mx::synchronize();

    auto ref = run_common(q, k, v, g, beta, state, false);
    auto hip = run_common(q, k, v, g, beta, state, true);

    float y_diff = max_abs_diff(ref.first, hip.first);
    float s_diff = max_abs_diff(ref.second, hip.second);
    bool ok = y_diff <= tolerance && s_diff <= tolerance;
    std::cout << name
              << "\tB=" << B
              << "\tT=" << T
              << "\tHk=" << Hk
              << "\tHv=" << Hv
              << "\tDk=" << Dk
              << "\tDv=" << Dv
              << "\ty_diff=" << std::scientific << y_diff
              << "\tstate_diff=" << s_diff
              << "\tstatus=" << (ok ? "ok" : "fail")
              << '\n';
    return ok;
}

int main() {
    bool ok = true;
    const float tolerance = 5e-2f;
    ok &= run_case("qwen35_0p8b_decode", 1, 1, 16, 16, 128, 128, tolerance);
    ok &= run_case("qwen35_large_decode", 1, 1, 16, 64, 192, 128, tolerance);
    ok &= run_case("qwen35_0p8b_prefill4", 1, 4, 16, 16, 128, 128, tolerance);
    return ok ? 0 : 2;
}
