// The environment map's tables (environment_data.h), built on the host in double while the scene is uploaded: the
// rows' cosines, every texel's density per steradian, and the map's total. A translation unit of its own with no
// device code, as display.cu and quality.cu are TUs of their own: the kernels' code doesn't depend on it.
//
// Double, because the sums run over half a million texels (a 1k map) whose luminance spans eight orders of magnitude
// (a sun of 61,184 over a sky of 10⁻³): float would lose the dim texels' share of S entirely. The work is linear in the
// texels, a few milliseconds for a 1k map, and needs no GPU, so the tables are exact and testable on the host.
#include "environment_data.h"

#include <cmath>
#include <cstddef>
#include <format>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace gpu
{
namespace
{
// Rec. 709's luminance of a texel, the weight its cell gets: any weight > 0 wherever a channel is > 0 would make the
// sampler unbiased; this one makes the luminance of radiance over density the same number for every sample (S)
double luminance(const float* rgb)
{
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

// What the renderer refuses (validate in renderer.cu says it first, in its own words): RGBE files can hold neither
// NaN nor negative values, but a map made in code could
void check(const environment_desc& env)
{
    const image_desc& map = env.image;
    if (map.width < 1 || map.height < 1)
    {
        throw std::invalid_argument("an environment map needs at least one texel");
    }
    if (map.width > max_environment_width || map.height > max_environment_height)
    {
        throw std::invalid_argument(std::format("an environment map is at most {}×{} texels, not {}×{}",
                                                max_environment_width, max_environment_height, map.width,
                                                map.height));
    }
    if (map.rgb.size() != 3 * static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.height))
    {
        throw std::invalid_argument("an environment map's pixels don't match its size");
    }
    for (const float value : map.rgb)
    {
        if (!(std::isfinite(value) && value >= 0)) // NaN fails both comparisons
        {
            throw std::invalid_argument("an environment map's radiance must be a finite number ≥ 0");
        }
    }
    if (!std::isfinite(env.yaw_degrees))
    {
        throw std::invalid_argument("an environment map's yaw must be a finite number");
    }
    if (!(std::isfinite(env.scale) && env.scale >= 0))
    {
        throw std::invalid_argument("an environment map's scale must be a finite number ≥ 0");
    }
}
}

environment_tables build_environment(const environment_desc& env)
{
    check(env);
    const int width = env.image.width;
    const int height = env.image.height;
    const std::size_t texel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const float* const rgb = env.image.rgb.data();

    // The rows' borders: cos θ_j = cos(πj/H) from the top (+y, θ = 0) down. The ends are exactly 1 and -1, so the
    // cells' solid angles add up to 4π: Σ_j (2π/W)·W·(cos θ_j − cos θ_{j+1}) = 2π·(cos θ_0 − cos θ_H).
    std::vector<double> cosines(static_cast<std::size_t>(height) + 1);
    for (int j = 0; j <= height; ++j)
    {
        cosines[static_cast<std::size_t>(j)] = std::cos(std::numbers::pi * j / height);
    }
    cosines.front() = 1;
    cosines.back() = -1;
    const auto solid_angle = [&](int j) {
        const auto row = static_cast<std::size_t>(j);
        return 2 * std::numbers::pi / width * (cosines[row] - cosines[row + 1]);
    };

    // S = Σ_ij Y_ij·Ω_j = ∫ Y dω of the map as its nearest texel shows it: the luminance the sampler distributes. Each
    // row's luminance first, then times its cells' one solid angle.
    double total = 0;
    for (int j = 0; j < height; ++j)
    {
        double row = 0;
        for (int i = 0; i < width; ++i)
        {
            row += luminance(rgb + 3 * (static_cast<std::size_t>(j) * width + i));
        }
        total += row * solid_angle(j);
    }

    environment_tables tables;
    tables.width = width;
    tables.height = height;
    tables.total = total;
    tables.row_cos.reserve(cosines.size());
    for (const double c : cosines)
    {
        tables.row_cos.push_back(static_cast<float>(c));
    }
    // The density per steradian of a cell's directions, (Y·Ω/S)/Ω = Y/S, rounded once. A black map (S = 0) has
    // nothing to sample: 0 everywhere, and its rays still see its texels.
    tables.texels.resize(texel_count);
    for (std::size_t k = 0; k < texel_count; ++k)
    {
        const float* const texel = rgb + 3 * k;
        const float pdf = total > 0 ? static_cast<float>(luminance(texel) / total) : 0.0f;
        tables.texels[k] = {texel[0], texel[1], texel[2], pdf};
    }
    // The alias table over the cells (weights Y·Ω) comes with the map's importance sampling: until then the map is
    // seen, not sampled (bins empty, log2_bins 0)
    return tables;
}
}
