// The environment map on the GPU: from a direction to the map's texel and back, and the map's radiance along it. Device
// code; the quality kernel's only, included by background.cuh (and the tests).
//
// The map is equirectangular with +y up, seen from inside: a direction m of map space has the longitude
// φ' = atan2(m_z, m_x) ∈ [−π, π] and the polar angle θ = atan2(√(m_x² + m_z²), m_y) ∈ [0, π], and the map's coordinates
// u = 0.5 + φ'/(2π) (columns, left to right) and t = θ/π (rows, from the top: row 0 is +y). So the image's centre is
// +x, +z is a quarter of the width right of it (u = 0.75) and -z a quarter left (u = 0.25): looking along +x with +y
// up, +z is on the right, and the photograph appears as it was taken, not mirrored. This is Blender's convention
// (Cycles' direction_to_equirectangular, z up) turned to +y up by a rotation, which keeps handedness, and it is the
// opposite of the books' sphere_uv (u = (atan2(−z, x) + π)/(2π)), which maps a globe seen from outside: from inside it
// would mirror the photograph left to right.
//
// Both atan2 forms take a direction of any length: the books' rays aren't unit vectors (camera rays, metal), and the
// map's cell of a ray must not depend on it. Near the poles atan2 stays accurate where acos(m_y) would not.
#pragma once

#include "environment_data.h"
#include "pdf.cuh"
#include "quality_view.cuh"
#include "vec3.cuh"

namespace gpu
{
// A world direction in map space: turned about +y by -α, the scene's yaw (cos α and sin α from the host, in double).
// Then the map's column u = 0.5 + φ'/(2π) lies at the world longitude φ' + α. Turning keeps solid angles, so the
// densities don't change.
__device__ inline vec3<float> to_map(const environment_view& e, const vec3<float>& d)
{
    return {e.cos_yaw * d.x + e.sin_yaw * d.z, d.y, -e.sin_yaw * d.x + e.cos_yaw * d.z};
}

// The inverse: a direction of map space (a sample drawn in the map's cells) in the world
__device__ inline vec3<float> from_map(const environment_view& e, const vec3<float>& d)
{
    return {e.cos_yaw * d.x - e.sin_yaw * d.z, d.y, e.sin_yaw * d.x + e.cos_yaw * d.z};
}

// The map's coordinates of a direction of map space: u = 0.5 + φ'/(2π) and t = θ/π, both in [0, 1] up to rounding
__device__ inline void map_coordinates(const vec3<float>& m, float& u, float& t)
{
    const float phi = atan2(m.z, m.x);
    const float theta = atan2(sqrt(m.x * m.x + m.z * m.z), m.y);
    u = 0.5f + phi * (0.5f * inverse_pi<float>);
    t = theta * inverse_pi<float>;
}

// A column or row of the map: k clamped to [0, n − 1]
__device__ inline int clamp_index(int k, int n)
{
    return k < 0 ? 0 : (k < n ? k : n - 1);
}

// THE one map from a direction (any non-zero length, world space) to the map's cell j·W + i: i = ⌊u·W⌋, j = ⌊t·H⌋,
// both clamped. Every lookup that weighs light by a density goes through it, the escapes and the light samples alike:
// a ray's radiance and the density of its direction then come from the same texel, which multiple importance sampling
// needs (the density must be the light sampler's, of the very function the escapes see). The float-to-int conversion
// truncates toward zero, ⌊⌋ for every value from -1 up, which covers the range: u·W can fall below 0 only by rounding
// (φ' = -π), t·H passes H only by rounding (θ = π), and the clamps take both back (a NaN converts to 0).
__device__ inline int env_cell(const environment_view& e, const vec3<float>& world_dir)
{
    float u;
    float t;
    map_coordinates(to_map(e, world_dir), u, t);
    const int i = clamp_index(static_cast<int>(u * static_cast<float>(e.width)), e.width);
    const int j = clamp_index(static_cast<int>(t * static_cast<float>(e.height)), e.height);
    return j * e.width + i;
}

// The way back, for the light sampler: the world direction at the fractions s_θ and s_φ ∈ [0, 1) of cell (i, j)'s
// extent in cos θ and in φ', cos θ = cos θ_j − s_θ·(cos θ_j − cos θ_{j+1}) and φ' = 2π(i + s_φ)/W − π. The cell is a
// rectangle in (φ', cos θ), and dω = dφ'·d(cos θ), so uniform fractions give directions uniform over its solid angle.
// sin θ = √((1 − cos θ)(1 + cos θ)) keeps its digits near the poles, where 1 − cos²θ would cancel them, and sincospi
// takes φ'/π = 2(i + s_φ)/W − 1, with no rounded π. A unit vector, to rounding.
__device__ inline vec3<float> cell_direction(const environment_view& e, int i, int j, float s_theta, float s_phi)
{
    const float top = e.row_cos[j];
    const float bottom = e.row_cos[j + 1];
    const float cos_theta = top - s_theta * (top - bottom);
    const float sin_theta = sqrt(fmax(0.0f, (1.0f - cos_theta) * (1.0f + cos_theta)));
    float sin_phi;
    float cos_phi;
    sincospi(2.0f * (static_cast<float>(i) + s_phi) / static_cast<float>(e.width) - 1.0f, &sin_phi, &cos_phi);
    return from_map(e, {sin_theta * cos_phi, cos_theta, sin_theta * sin_phi});
}

// The map's radiance along dir, as its nearest texel holds it (times the scale), and the density with which the light
// sampler draws dir (per steradian): one 16-byte load. What the rays multiple importance sampling weighs see.
__device__ inline color<float> env_radiance(const environment_view& e, const vec3<float>& dir, float& pdf)
{
    const env_texel texel = e.texels[env_cell(e, dir)];
    pdf = texel.pdf;
    return e.scale * color<float>{texel.r, texel.g, texel.b};
}

// a + s·(b − a): exact when a == b (a constant map stays its constant), a at s = 0
__device__ inline color<float> env_lerp(const color<float>& a, const color<float>& b, float s)
{
    return a + s * (b - a);
}

__device__ inline color<float> texel_color(const env_texel& texel)
{
    return {texel.r, texel.g, texel.b};
}

// The map's radiance along dir between its four nearest texel centres, bilinearly (times the scale): what a camera ray
// and a ray after a mirror or glass see, so a background seen directly isn't a mosaic of the map's texels (a 1k map
// spans about 4 pixels per texel at 480 pixels across 40°). No density goes with it: only the material's own sample
// can draw those rays, so no other technique weighs them, and the blur of the sun into its neighbours (which have a
// lower density) never meets a light sample. Texel centres sit at x = u·W − ½, y = t·H − ½: the columns wrap around
// (the map is a full turn: column W − 1 neighbours column 0), the rows are clamped (above the first row's centre and
// below the last's, the row alone). In software, in float: the texture units' filter weighs with 8-bit fractions.
__device__ inline color<float> env_radiance_filtered(const environment_view& e, const vec3<float>& dir)
{
    float u;
    float t;
    map_coordinates(to_map(e, dir), u, t);
    const float x = u * static_cast<float>(e.width) - 0.5f;
    const float y = t * static_cast<float>(e.height) - 0.5f;
    const float x0 = floor(x);
    const float y0 = floor(y);
    const float fx = x - x0;
    const float fy = y - y0;
    // x0 is in [-1, W − 1], as u is in [0, 1] (atan2 returns at most float(π), whose u rounds to 1 at most): -1 wraps
    // to the last column, and the last column's neighbour is column 0. (A NaN direction converts to column 0.)
    const int i = static_cast<int>(x0);
    const int i0 = i < 0 ? i + e.width : i;
    const int i1 = i0 + 1 < e.width ? i0 + 1 : 0;
    const int j = static_cast<int>(y0);
    const int j0 = clamp_index(j, e.height) * e.width;
    const int j1 = clamp_index(j + 1, e.height) * e.width;
    const color<float> top = env_lerp(texel_color(e.texels[j0 + i0]), texel_color(e.texels[j0 + i1]), fx);
    const color<float> bottom = env_lerp(texel_color(e.texels[j1 + i0]), texel_color(e.texels[j1 + i1]), fx);
    return e.scale * env_lerp(top, bottom, fy);
}
}
