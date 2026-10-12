// The GPU's textures (see texture.ixx): plain data and a loop instead of objects with virtual functions.
#pragma once

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
    int unused;      // padding, named: the alignment makes the struct 32 bytes in float either way
    color<Real> rgb; // solid: the color. (Not named `color`: a member named like the type color<Real> used above it
                     // would change what the name means inside the struct, which C++ forbids.)
    Real inv_scale;  // checker: 1 / the side of its cubes, the number the CPU multiplies by
};

// The texture arrays as the kernel sees them
template <real_number Real>
struct texture_view
{
    const texture<Real>* textures;
};

// The texture's color at the point p (texture::value), where p is the hit point in world coordinates.
//
// The CPU's checker_texture asks one of its two textures for the color, and that one may be a checker too. Device code
// can call itself recursively, but each level would need room on the thread's stack, so the GPU walks down the chain in
// a loop: each step replaces the checker by the texture of p's cube. The textures a checker refers to always have
// smaller indices than its own (renderer.cu checks it), so the loop ends.
template <real_number Real>
__device__ color<Real> texture_value(const texture_view<Real>& view, int index, const point3<Real>& p)
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
    return t.rgb; // solid
}
}
