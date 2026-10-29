// The environment map's tables (environment_data.h), built on the host in double while the scene is uploaded: the
// rows' cosines, every texel's density per steradian, the map's total, and the alias table the light sampler picks a
// cell with. A translation unit of its own with no device code, as display.cu and quality.cu are TUs of their own: the
// kernels' code doesn't depend on it.
//
// Double, because the sums run over half a million texels (a 1k map) whose luminance spans eight orders of magnitude
// (a sun of 61,184 over a sky of 10⁻³): float would lose the dim texels' share of S entirely. The work is linear in the
// texels, a few milliseconds for a 1k map, and needs no GPU, so the tables are exact and testable on the host.
#include "environment_data.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <numbers>
#include <stdexcept>
#include <utility>
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

// The steps of the sampler's coin: random_real<float> gives k·2^-24 with k uniform in [0, 2^24), and ξ < q holds for
// ⌈q·2^24⌉ of those k. A q on that grid is then the coin's chance exactly; any other q would be rounded up by the
// comparison, unseen by the table.
constexpr double coin_steps = 0x1p24;

// The coin of a bin whose cell keeps the share s < 1 of it (the rest goes to its alias), on the coin's grid. Rounded to
// the nearer grid point, unless the error the coins before it left (drift, in steps) is better paid back by the
// farther one: error diffusion, which keeps the drift within half a step. Rounded independently, the coins' errors
// would add up like a random walk, √P·2^-24/√12 ≈ 10⁻⁵ of a bin for a 1k map's 2^19 bins, and Vose's construction puts
// the whole sum on the last bins it fills. A cell with light keeps at least one step, so every direction the map lights
// can be drawn (-p nee counts the light samples alone where they could have found a direction: a cell that has a
// density but is never drawn would lose its light); a bin of weight 0 keeps none and never gives itself.
double coin(double share, double& drift)
{
    if (share == 0)
    {
        return 0;
    }
    const double x = share * coin_steps; // exact: a power of two
    double k = std::floor(x);
    if (drift + (k - x) < -0.5 || k == 0)
    {
        k += 1;
    }
    drift += k - x;
    return k / coin_steps;
}

// Vose's alias table (Vose 1991, in the stable form Schwarz describes in "Darts, Dice, and Coins"; pbrt-v4's
// AliasTable is the same method) for the bins' shares p̂_c = P·w_c/S, which add up to the P bins: a bin of share below 1
// (Small) keeps its own cell with its share as the coin's chance and gives the rest to a bin of share 1 or more
// (Large), which then has that much less to place, and goes back to Small or Large. Every bin is filled once, so the
// work is linear; the lists are stacks, so a large bin keeps giving to the next small ones until it is small itself.
//
// The large bin gives up exactly 1 − q, q the stored coin: on the 2^-24 grid 1 − q is exact in double, so p̂_g − (1 − q)
// rounds once (Schwarz's (p̂_g + p̂_l) − 1 is there to avoid the inexact 1 − p̂_l of an unrounded share). Then the chance
// of every cell, P'_c = (q_c + Σ_{b: alias_b = c} (1 − q_b))/P, is its share up to the rounding of its own coin, a step
// at most: what the coins before it rounded is in the share it had left. The sun of a sky map, a share of 50,000, is
// the alias of some 50,000 bins, whose roundings would otherwise add up in it.
//
// When one list runs out, what is left is round-off: the remaining shares add up to their count up to the drift (half
// a step, plus a step for each forced coin of a dim cell, less than 1/8 in all for 2^21 bins), so each of them is 1
// within that, and gets q = 1. A bin of weight 0 is never among them: its share 0 would leave the others' sum short by
// more than 1. So every alias is a cell with light, and a bin that keeps itself with q = 1 is a cell too.
std::vector<alias_bin> alias_table(std::vector<double> share)
{
    const std::size_t count = share.size();
    std::vector<alias_bin> bins(count);
    std::vector<std::uint32_t> small;
    std::vector<std::uint32_t> large;
    small.reserve(count);
    large.reserve(count);
    for (std::uint32_t c = 0; c < count; ++c)
    {
        (share[c] < 1 ? small : large).push_back(c);
    }
    double drift = 0;
    while (!small.empty() && !large.empty())
    {
        const std::uint32_t l = small.back();
        small.pop_back();
        const std::uint32_t g = large.back();
        large.pop_back();
        const double q = coin(share[l], drift);
        bins[l] = {static_cast<float>(q), static_cast<int>(g)}; // q = k·2^-24, k ≤ 2^24: exact in float
        share[g] -= 1 - q;                                       // ≥ 0: share[g] ≥ 1 ≥ 1 − q
        (share[g] < 1 ? small : large).push_back(g);
    }
    for (const std::uint32_t c : large)
    {
        bins[c] = {1.0f, static_cast<int>(c)};
    }
    for (const std::uint32_t c : small)
    {
        bins[c] = {1.0f, static_cast<int>(c)};
    }
    return bins;
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
    if (total == 0)
    {
        return tables; // no light to sample: no alias table (bins empty, log2_bins 0)
    }

    // The alias table over the cells, cell c's weight w_c = Y_c·Ω_j. Padded with bins of weight 0 to P = 2^k bins,
    // k = max(1, ⌈log2 N⌉): the device takes a bin's index from the top k bits of one random number, exactly uniform
    // for any N (⌊ξ·N⌋ from a 24-bit ξ is uniform only when N divides 2^24), and k ≥ 1 keeps its shift 32 − k below 32
    // (a shift by 32 is undefined), so a 1×1 map gets one padded bin. A 1k map's 2^19 cells need no padding.
    tables.log2_bins = std::max(1, static_cast<int>(std::bit_width(texel_count - 1)));
    const std::size_t bin_count = std::size_t{1} << tables.log2_bins;
    std::vector<double> share(bin_count, 0.0); // p̂_c = P·w_c/S; the padded bins keep 0
    for (int j = 0; j < height; ++j)
    {
        const double omega = solid_angle(j);
        for (int i = 0; i < width; ++i)
        {
            const std::size_t c = static_cast<std::size_t>(j) * width + i;
            share[c] = luminance(rgb + 3 * c) * omega * static_cast<double>(bin_count) / total;
        }
    }
    tables.bins = alias_table(std::move(share));
    return tables;
}
}
