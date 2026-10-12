// The GPU's textures (see texture.ixx): plain data and a loop instead of objects with virtual functions.
#pragma once

#include "perlin.cuh"
#include "renderer.h"
#include "vec3.cuh"

namespace gpu
{
// A texture as the GPU stores it: 32 bytes in float, two 16-byte loads. Only the fields of its kind mean something.
template <real_number Real>
struct alignas(16) texture
{
    texture_kind kind;
    int even;        // checker: the texture of the even cubes, an index smaller than this texture's own
    int odd;         // checker: the texture of the odd cubes
    int table;       // image: its index in texture_view::images; noise: in texture_view::noises
    color<Real> rgb; // solid: the color. (Not named `color`: a member named like the type color<Real> used above it
                     // would change what the name means inside the struct, which C++ forbids.)
    Real inv_scale;  // checker: 1 / the side of its cubes, the number the CPU multiplies by
};

// Where an image's pixels are in texture_view::texels. 16 bytes: the alignment lets one load read all of it (from a
// struct of ints the compiler would know only 4-byte alignment and load them one by one).
struct alignas(16) image_info
{
    int offset; // texels[offset + j·width + i] is pixel (i, j), rows from the top
    int width;
    int height;
    int unused; // padding, named
};

// One pixel of an image: 16 bytes, one load. The fourth float is padding: three floats would take three loads. Floats
// keep stb_image's values exactly, so the --fp64 kernels read the same colors.
struct alignas(16) texel
{
    float r;
    float g;
    float b;
    float unused;
};

// A marble texture's numbers (the CPU's noise_texture): the tables of its own perlin object, and the frequency of its
// stripes
template <real_number Real>
struct noise_texture
{
    perlin_table<Real> perlin;
    Real scale;
};

// The texture arrays as the kernel sees them
template <real_number Real>
struct texture_view
{
    const texture<Real>* textures;
    const image_info* images;
    const texel* texels; // the pixels of all images, one after another
    const noise_texture<Real>* noises;
};

// The texture's color at the hit (texture::value): (u, v) are the hit's texture coordinates, p the hit point in world
// coordinates.
//
// The CPU's checker_texture asks one of its two textures for the color, and that one may be a checker too. Device code
// can call itself recursively, but each level would need room on the thread's stack, so the GPU walks down the chain in
// a loop: each step replaces the checker by the texture of p's cube. The textures a checker refers to always have
// smaller indices than its own (renderer.cu checks it), so the loop ends.
template <real_number Real>
__device__ color<Real> texture_value(const texture_view<Real>& view, int index, Real u, Real v, const point3<Real>& p)
{
    texture<Real> t = view.textures[index];
    while (t.kind == texture_kind::checker)
    {
        // checker_texture::value: floor before the cast, as int(-0.5) is 0, which would make the cube at 0 twice as
        // wide. For a negative sum C++'s % gives 0 or -1 (the sign of the dividend), so "== 0" still means even.
        const int x = static_cast<int>(floor(t.inv_scale * p.x));
        const int y = static_cast<int>(floor(t.inv_scale * p.y));
        const int z = static_cast<int>(floor(t.inv_scale * p.z));
        t = view.textures[(x + y + z) % 2 == 0 ? t.even : t.odd];
    }
    switch (t.kind)
    {
    case texture_kind::image:
    {
        // image_texture::value: u and v clamped to [0, 1], v flipped (rows go from the top), and the pixel they fall in
        const image_info image = view.images[t.table];
        const Real clamped_u = fmin(fmax(u, Real(0)), Real(1));
        const Real flipped_v = Real(1) - fmin(fmax(v, Real(0)), Real(1));
        const int column = static_cast<int>(clamped_u * static_cast<Real>(image.width));
        const int row = static_cast<int>(flipped_v * static_cast<Real>(image.height));
        // u = 1 would give the column width, one past the edge
        const int i = column < image.width - 1 ? column : image.width - 1;
        const int j = row < image.height - 1 ? row : image.height - 1;
        const texel pixel = view.texels[image.offset + j * image.width + i];
        return {static_cast<Real>(pixel.r), static_cast<Real>(pixel.g), static_cast<Real>(pixel.b)};
    }
    case texture_kind::noise:
    {
        // noise_texture::value: 0.5·(1 + sin(scale·p.z + 10·turbulence(p))) in every channel. The sine makes stripes
        // along z, and the turbulence bends them into veins. The turbulence takes p itself: the scale multiplies only
        // p.z. A reference, not a copy: the struct holds the whole table, 4.8 KB in float and 9 KB in double.
        const noise_texture<Real>& marble = view.noises[t.table];
        const Real gray =
            Real(0.5) * (Real(1) + sin(marble.scale * p.z + Real(10) * turbulence(marble.perlin, p, 7)));
        return {gray, gray, gray};
    }
    default:
        return t.rgb; // solid
    }
}
}
