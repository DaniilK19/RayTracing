// The display transforms beyond the books (gpu::tone_map in renderer.h): device functions that turn a pixel's linear
// light, after the exposure, into display-encoded values in [0, 1], and those into bytes. Float only. Included by
// tonemap.cu, whose kernel is compiled with -fmad=false, and by GpuTests' tone_tests.cu, which checks every function
// against references in double written from the same equations. No BVH, no scene: the shading headers of the quality
// kernel stay apart from the display.
//
// Why a curve. The books' bytes (display.cu) take the square root of each channel and clamp it below 1: everything
// brighter than 1 is plain white, one channel at a time. A lamp-lit highlight of 4 in red and 1 in green loses its
// shape and turns from orange to yellow to white as it brightens. A tone curve compresses the whole range of the light
// into [0, 1] instead, smoothly: the shadows and the middle greys keep their steps, and the highlights roll off toward
// white. The curves here are the ones film and game engines settled on in the 2020s:
//   - none: no curve, the honest baseline: the light clipped to [0, 1] and encoded with the sRGB curve, as most image
//     viewers expect (Blender's "Standard", pbrt-v4's PNG output).
//   - agx: Troy Sobotka's AgX (github.com/sobotka/AgX), the default view of Blender 4: the light on a log2 scale of
//     16.5 stops around middle grey (0.18 sits 10 stops above the bottom), then an S-shaped curve. Its inset matrix
//     first pulls every primary a little toward grey, so that a very bright saturated color drifts toward white as a
//     photograph's does, instead of toward the nearest of the six pure hues a per-channel curve skews it to. In the
//     form of Benjamin Wrensch's "Minimal AgX" (iolite), which adds the inverse matrix after the curve (the "outset")
//     to give back some of the saturation. Its output is already encoded for the display: no sRGB curve after it.
//   - neutral: Khronos PBR Neutral (github.com/KhronosGroup/ToneMapping, PBR_Neutral/README.md), written from the
//     README's equations: colors up to 0.8 pass unchanged but for a constant 0.04 (a base color comes out as authored),
//     and only the brightest channel is compressed above that, all three by one factor, so the hue never changes.
//     Then the sRGB curve.
// The constants are tonemap.md's (out/m4/understand), each checked there against its primary source.
//
// AgX's matrices and structure are from iolite's Minimal AgX, Copyright (c) 2024 Missing Deadlines (Benjamin Wrensch),
// and the closed form of its curve from github.com/bWFuanVzYWth/AgX, Copyright (c) 2024 linlin, both under the MIT
// License:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include <cstdint>

#include "renderer.h"

namespace gpu::tone
{
// A pixel's three channels: linear light in, display-encoded values out
struct rgb
{
    float r, g, b;
};

// [0, 1]; NaN gives 0 (fmaxf returns its other argument when one is NaN)
__device__ inline float clamp01(float v)
{
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

// What a tone map may take: NaN, -0, negatives and -∞ become 0, +∞ a large finite number. (∞ would make PBR Neutral's
// ∞·0 a NaN, and 3·10^38 still leaves the AgX inset room below FLT_MAX ≈ 3.4·10^38: its rows add up to at most
// 1.000136.)
__device__ inline float sanitize(float v)
{
    return v > 0.0f ? fminf(v, 3.0e38f) : 0.0f;
}

// The sRGB encoding (its OETF, IEC 61966-2-1; registry.color.org/rgb-registry/srgb) of a linear value in [0, 1]: a
// straight line near black, then a power of 1/2.4. The two pieces meet at 0.0031308 (both 0.0404500 there), so the
// side a value just at the threshold falls on matters by less than 10^-7. 1 gives exactly 1, so a clipped white is 255
// and never dithered: in float, 1.055 − 0.055 rounds to 1 − 2^-24.
__device__ inline float srgb_encode(float linear)
{
    if (linear >= 1.0f)
    {
        return 1.0f;
    }
    return linear <= 0.0031308f ? 12.92f * linear : 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
}

// AgX's S-curve, from a position x in [0, 1] on its log2 scale to the display value: linlin's closed form,
//   f(x) = 0.5 + 2d·(1 + a·|d|^b)^c   with d = x − t, t = 10/16.5 (middle grey),
// (a, b, c) = (59.507875, 3, −1/3) below t and (69.86278913545539, 13/4, −4/13) above it. It matches the 4096 entries
// of Troy Sobotka's curve (AgX_Default_Contrast.spi1d) to 5·10^-8, and f(0) = 0, f(t) = 0.5, f(1) = 1. Both halves
// have the slope 2 at t, so the curve is smooth there. The powers are computed so as to round as little as possible:
// |d|³ by multiplying, |d|^(13/4) as |d|³·√√|d| (the square root is correctly rounded), (·)^(−1/3) as CUDA's
// reciprocal cube root (1 ulp at most). Near x = 0 and x = 1 the result is 0.5 plus a number close to −0.5 or +0.5,
// so the error of the last power shows in full there.
__device__ inline float agx_curve(float x)
{
    constexpr float t = 0.6060606060606061f; // 10/16.5
    const float d = x - t;
    const float m = fabsf(d);
    const float cube = m * m * m;
    if (x <= t)
    {
        return 0.5f + 2.0f * d * rcbrtf(1.0f + 59.507875f * cube);
    }
    return 0.5f + 2.0f * d * powf(1.0f + 69.86278913545539f * (cube * sqrtf(sqrtf(m))), -4.0f / 13.0f);
}

// A linear value as a position in [0, 1] on AgX's log2 scale: 16.5 stops from log2(0.18) − 10 = −12.47393 to
// log2(0.18) + 6.5 = 4.026069, linear 0.000176 to 16.29; darker is 0 (log2f(0) = −∞) and brighter 1
__device__ inline float agx_position(float linear)
{
    constexpr float min_ev = -12.47393f;
    constexpr float max_ev = 4.026069f;
    return clamp01((log2f(linear) - min_ev) / (max_ev - min_ev));
}

// AgX of a linear color (each channel at least 0): the inset, the log2 scale, the curve per channel, the outset. The
// matrices act on the color as a column vector, out_i = Σ_j M[i][j]·in_j, their rows as written here: the inset's from
// Troy Sobotka's config.ocio ("AgX Log (Kraken)"), whose rows add up to 1.000136, 0.999963 and 0.999952 (white stays
// nearly white), and the outset, its exact inverse, from iolite (GLSL's mat3 is column-major: these are its columns).
// The inset's entries are all positive, so a color of light stays one. Display-encoded for a display of gamma 2.2, as
// AgX's base is: iolite's last pow 2.2 is only for a render target that applies the sRGB curve itself, and a PNG
// isn't one.
__device__ inline rgb agx(rgb linear)
{
    const float r = 0.842479062253094f * linear.r + 0.0784335999999992f * linear.g + 0.0792237451477643f * linear.b;
    const float g = 0.0423282422610123f * linear.r + 0.878468636469772f * linear.g + 0.0791661274605434f * linear.b;
    const float b = 0.0423756549057051f * linear.r + 0.0784336f * linear.g + 0.879142973793104f * linear.b;
    const float cr = agx_curve(agx_position(r));
    const float cg = agx_curve(agx_position(g));
    const float cb = agx_curve(agx_position(b));
    return {clamp01(1.19687900512017f * cr - 0.0980208811401368f * cg - 0.0990297440797205f * cb),
            clamp01(-0.0528968517574562f * cr + 1.15190312990417f * cg - 0.0989611768448433f * cb),
            clamp01(-0.0529716355144438f * cr - 0.0980434501171241f * cg + 1.15107367264116f * cb)};
}

// Khronos PBR Neutral of a linear color (each channel at least 0), linear in [0, 1] out. With F90 = 0.04, K_s = 0.76
// and K_d = 0.15:
//   x = min(R, G, B); f = x − x²/(4·F90) = x − 6.25x² for x ≤ 2·F90 = 0.08, else F90; c′ = c − f; p = max(c′).
// A dielectric's specular reflection adds about 0.04 to every color: f takes it off (up to 0.08 the quadratic toe, so
// black stays black). Up to p = K_s the color is c′ as it is. Above it the brightest channel is compressed,
//   p_n = 1 − (1 − K_s)²/(p + 1 − 2K_s),   g = 1/(K_d·(p − p_n) + 1),   out = c′·(p_n/p)·g + p_n·(1 − g),
// one factor and one shift for all three channels: out − min(out) stays a multiple of c − min(c), the hue, and the
// color fades toward white by g. Both joints are continuous: f = 0.04 from both sides of x = 0.08, and p_n = p, g = 1
// at p = 0.76.
__device__ inline rgb pbr_neutral(rgb c)
{
    constexpr float f90 = 0.04f;
    constexpr float ks = 0.8f - f90;
    constexpr float kd = 0.15f;
    const float x = fminf(c.r, fminf(c.g, c.b));
    const float f = x <= 2.0f * f90 ? x - 6.25f * x * x : f90;
    const rgb shifted{c.r - f, c.g - f, c.b - f};
    const float p = fmaxf(shifted.r, fmaxf(shifted.g, shifted.b));
    if (p <= ks)
    {
        return shifted;
    }
    const float pn = 1.0f - (1.0f - ks) * (1.0f - ks) / (p + 1.0f - 2.0f * ks);
    const float g = 1.0f / (kd * (p - pn) + 1.0f);
    const float scale = pn / p * g;
    const float white = pn * (1.0f - g);
    return {shifted.r * scale + white, shifted.g * scale + white, shifted.b * scale + white};
}

// A tone map other than books of a linear color (after the exposure): display-encoded values, each finite and in
// [0, 1], whatever the input (NaN and ±∞ included). The books' bytes never come here: display.cu makes them.
__device__ inline rgb encode(rgb linear, tone_map op)
{
    const rgb c{sanitize(linear.r), sanitize(linear.g), sanitize(linear.b)};
    switch (op)
    {
    case tone_map::agx:
        return agx(c);
    case tone_map::neutral:
    {
        const rgb n = pbr_neutral(c);
        return {srgb_encode(clamp01(n.r)), srgb_encode(clamp01(n.g)), srgb_encode(clamp01(n.b))};
    }
    case tone_map::books: // (not used: see above)
    case tone_map::none:
        break;
    }
    return {srgb_encode(clamp01(c.r)), srgb_encode(clamp01(c.g)), srgb_encode(clamp01(c.b))};
}

// v − ⌊v⌋, in [0, 1) for the values below (exact: v and ⌊v⌋ are close floats)
__device__ inline float fract(float v)
{
    return v - floorf(v);
}

// Interleaved gradient noise (Jorge Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare",
// SIGGRAPH 2014) at pixel (x, y) of the image: a number in [0, 1) that differs from its neighbors' as much as it can,
// every 3×3 block of pixels holding a well spread set of values. No table, no random numbers, no frame number: the
// same pixel of the same image always gets the same value, so the window and RayTracingGPU's PNG agree byte for byte.
__device__ inline float ign(int x, int y)
{
    return fract(52.9829189f * fract(0.06711056f * static_cast<float>(x) + 0.00583715f * static_cast<float>(y)));
}

// An encoded value as a byte, round(255·v + dither): the dither (the pixel's ign − 0.5, in [−0.5, 0.5), one value for
// the three channels so a grey stays grey) moves the rounding up or down by up to half a level, so a smooth gradient
// that would show bands of 8-bit levels shows a fine grain instead, and over many pixels the bytes average 255·v
// exactly (pbrt-v4 dithers its PNGs the same way, with a table of blue noise). 0 and 1 are never dithered: black stays
// 0, an encoded 1 is 255. (AgX's brightest color is (0.99983, 1, 1), its outset's red row adding up to 0.999828: a
// blown-out red is dithered between 254 and 255 as any value below 1.) NaN gives 0.
__device__ inline std::uint8_t quantize(float encoded, float dither)
{
    if (!(encoded > 0.0f))
    {
        return 0;
    }
    if (encoded >= 1.0f)
    {
        return 255;
    }
    return static_cast<std::uint8_t>(fminf(fmaxf(floorf(255.0f * encoded + 0.5f + dither), 0.0f), 255.0f));
}
}
