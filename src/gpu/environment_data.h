// The environment map's tables: what environment_table.cu builds on the host from an equirectangular HDR image
// (environment_desc in renderer.h), and what the quality kernel reads on the GPU through quality_view::environment
// (environment.cuh, background.cuh). Plain types, for host and device code alike.
//
// The map is cut into cells of exact solid angle, one per texel: texel (i, j) covers φ' ∈ [2πi/W − π, 2π(i+1)/W − π)
// and θ from πj/H to π(j+1)/H, so in (φ', cos θ) it is a rectangle, of solid angle
// Ω_j = (2π/W)·(cos θ_j − cos θ_{j+1}), the cosines computed in double. With the texel's luminance
// Y = 0.2126R + 0.7152G + 0.0722B (Rec. 709) and the map's whole S = Σ Y·Ω, a sampler that picks cell c with the
// chance Y_c·Ω_c/S, then φ' and cos θ uniformly inside it, draws directions with the density Y/S per steradian,
// constant over the cell: dω = dφ'·d(cos θ), so no 1/sin θ and no pole to guard (pbrt-v3's (u, v) cells need both).
// That density is stored with the texel's radiance: one 16-byte load gives a ray that escaped both what it sees and
// how likely the light sampler was to draw it. A map of one radiance everywhere gets 1/(4π) exactly. (pbrt-v4's
// equal-area map gets the same constant density by resampling the image, which this avoids.)
#pragma once

#include <vector>

#include "renderer.h"

namespace gpu
{
// The largest map the renderer takes, per side: a 2k map. Its tables are 48 MiB on the GPU (texels 32, alias table
// 16); a 4k map would be 192.
inline constexpr int max_environment_width = 2048;
inline constexpr int max_environment_height = 1024;

// A texel of the map: its radiance as the image holds it (the scale multiplies it when it is looked up) and the
// density per steradian with which the light sampler draws the directions of its cell, Y/S. 16 bytes, one load.
struct alignas(16) env_texel
{
    float r, g, b, pdf;
};

// A bin of the map's alias table (Vose 1991): a bin drawn uniformly keeps its own cell with the chance q, else gives
// its alias, a cell with light. The table is padded to a power of two with bins of weight 0 (q = 0: they never give
// themselves), so the bin's index can be the top bits of one random number: exactly uniform whatever the map's size.
// q is a multiple of 2^-24, the step of the coin random_real<float> < q, so q is the coin's chance exactly.
struct alignas(8) alias_bin
{
    float q;
    int alias;
};

struct environment_tables
{
    int width = 0, height = 0, log2_bins = 0;
    double total = 0;              // S = Σ Y·Ω, in double: the map's luminance over the sphere, as the sampler sees it
    std::vector<env_texel> texels; // width·height, row 0 the image's top (+y); pdf = float(Y/S), 0 for a black map
    std::vector<alias_bin> bins;   // 1 << log2_bins ≥ 2 of them, the cells first, then the padding; empty for a black
                                   // map (S = 0, log2_bins 0), which no light sample can come from
    std::vector<float> row_cos;    // height + 1: float(cos(π·j/height)), from double; 1 first, -1 last
};

// The tables of a map, on the host. std::invalid_argument for a map the renderer refuses: a side below 1 or above
// 2048×1024, pixels that don't match the size, a value that isn't a finite number ≥ 0, a yaw that isn't finite, a
// scale that isn't a finite number ≥ 0.
environment_tables build_environment(const environment_desc& env);
}
