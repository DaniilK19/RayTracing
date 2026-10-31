// guides.cuh: the denoiser's guides at one sample of pixel (i, j): the albedo and normal of the first surface that
// isn't a mirror or glass, along the camera ray the beauty's pass starts with. One template over the tracer concept:
// guides.cu instantiates it with bvh_tracer, OptiX's __raygen__guides with optix_tracer.
//
// What a guide is for. OptiX's AI denoiser (the AOV model) tells noise from detail with two images besides the noisy
// one: the albedo, the color a surface reflects with, and the normal, the way it faces (world space). Where they
// change, an edge or a texture is real and must stay sharp; where they don't, a change in the noisy image is noise to
// smooth away. So a guide is what the eye would call the surface, by Open Image Denoise's rules for its auxiliary
// images (its documentation, "Denoising with auxiliary features"; OptiX's guide asks for the same kind of images):
//   - a matte surface, a particle of smoke, a glossy coat: its color (a coat's base color: the coat itself is
//     ignored) and its shading normal (a particle has none: the reverse of the ray);
//   - a metal rough enough to blur what it reflects (GGX r ≥ 0.1, the books' metal with fuzz): its color F0;
//   - a mirror, glass, a smoother metal: the scene seen in it. The guide follows its one direction, as the path does,
//     and what it finds there counts times the weights of those bounces (at most 8 of them: then nothing, 0 and 0);
//   - a lamp: its color scaled into [0, 1], and its normal;
//   - the sky or the map: its color clamped to [0, 1], and the normal 0 (no surface).
// One sample is noise-free in the denoiser's sense, but a pixel covers several surfaces at an edge, behind a lens out
// of focus or across a moving sphere's blur: 16 samples on the 4×4 grid of the beauty's own (camera_for_guides),
// averaged, see the same box filter, lens and shutter as the beauty.
#pragma once

#include <cstdint>
#include <vector_types.h>

#include "quality_path.cuh"

namespace gpu
{
struct guide_texel
{
    color<float> albedo; // in [0, 1]
    vec3<float> normal;  // world space, unit, or 0 where the ray escaped
};

// The samples of a pixel's guides: passes 0-15 of camera_for_guides' 4×4 grid
inline constexpr int guide_passes = 16;
// The mirror and glass vertices a guide's path goes through before it gives up
inline constexpr int guide_delta_vertices = 8;
// A GGX metal at least this rough blurs what it reflects into its own color, and the guide stops there (OIDN's
// "sufficiently rough" metal); below it the metal is a mirror to the eye
inline constexpr float guide_rough_metal = 0.1f;

// A color clamped to [0, 1], channel by channel; NaN gives 0 (fmaxf returns its other argument when one is NaN)
__device__ inline color<float> clamp_unit(const color<float>& c)
{
    return {fminf(fmaxf(c.x, 0.0f), 1.0f), fminf(fmaxf(c.y, 0.0f), 1.0f), fminf(fmaxf(c.z, 0.0f), 1.0f)};
}

// Whether the guide stops at a vertex of this lobe (what the eye takes for the surface), or follows its one direction
// (a mirror, glass, a metal too smooth to show its own color)
__device__ inline bool guide_stops_at(const bsdf& b)
{
    switch (b.kind)
    {
    case lobe::lambertian:
    case lobe::isotropic:
    case lobe::gloss:
        return true;
    case lobe::conductor:
        return b.roughness >= guide_rough_metal;
    case lobe::books_delta:
        return b.books.kind == material_kind::metal && b.books.fuzz > 0.0f; // the books' brushed metal, not glass
    default:
        return false; // smooth_conductor: a mirror
    }
}

// The guides of one sample: the camera ray of pass `pass` of pixel (i, j) (the integrator's own start, so its lens
// point and time are a beauty sample's), then the rules above vertex by vertex. cam is camera_for_guides' copy of the
// view's. The numbers come from rng as the integrator would draw them along the same rays (the media's distances, the
// choices of glass and of a rough mirror), from a seed of the guides' own.
template <tracer T>
__device__ guide_texel guide_sample(const T& tracer, const quality_view& view, const camera<float>& cam, int i, int j,
                                    int pass, pcg32& rng)
{
    quality_path q = start_quality_sample(cam, view, i, j, pass, rng);
    path<float>& p = q.base;
    color<float> w{1.0f, 1.0f, 1.0f}; // the weights of the delta bounces so far
    for (int deltas = 0; deltas < guide_delta_vertices; ++deltas)
    {
        const bvh_hit<float> hit = tracer.closest(p.r, spawn{p.from, p.inward}, rng);
        if (hit.index < 0)
        {
            // The sky, or the map as a camera ray and a ray after a delta vertex see it: filtered
            return {.albedo = w * clamp_unit(background(view, cam, p.r.direction, true)), .normal = {}};
        }
        const surface<float> s = tracer.surface_of(p.r, hit);
        const material<float> mat = view.base.materials[s.material];
        // The material's color as trace_quality_ray computes it: a matte surface, a particle, a lamp's front
        const bool colored = mat.kind == material_kind::lambertian || mat.kind == material_kind::isotropic ||
                             (mat.kind == material_kind::diffuse_light && s.rec.front_face);
        const color<float> c = colored ? surface_color(view.base, mat, s) : color<float>{};
        if (mat.kind == material_kind::diffuse_light)
        {
            // A lamp's radiance is its color times its power: scaled so that its largest channel is at most 1, its hue
            // stays. Its normal is the surface's (the absorber, meshes.cu's black lamp behind a triangle's face: the
            // reverse of the ray).
            return {.albedo = (1.0f / fmaxf(1.0f, largest_channel(c))) * (w * c), .normal = s.rec.normal};
        }
        const microfacet* side = view.microfacets ? &view.microfacets[s.material] : nullptr;
        const bsdf b = make_bsdf(mat, side, c, s.rec, p.r.direction);
        if (guide_stops_at(b))
        {
            // A conductor's albedo is its F0 (make_bsdf), the books' metal's its albedo, a coat's its base color c
            const color<float> albedo = b.kind == lobe::books_delta ? b.books.albedo : b.albedo;
            const vec3<float> normal = b.kind == lobe::isotropic ? -unit_vector(p.r.direction) : s.rec.normal;
            return {.albedo = w * albedo, .normal = normal};
        }
        // A mirror, glass or a smooth metal: what it shows lies along its one direction, times its weight
        bsdf_sample next;
        if (!sample_bsdf(b, p.r, s.rec, rng, next))
        {
            return {}; // absorbed: nothing found
        }
        w *= next.weight;
        p.r = {s.rec.p, next.wi, p.r.time};
        p.from = s.from;
        p.inward = dot(next.wi, s.outward) < 0.0f; // the integrator's self-intersection rule for the next ray
    }
    return {}; // 8 delta vertices: no surface found
}

// A copy of cam whose grid is 4×4 (sqrt_spp 4), so passes 0–15 cover each pixel as the beauty's samples do
__host__ __device__ inline camera<float> camera_for_guides(camera<float> cam)
{
    cam.sqrt_spp = 4;
    cam.recip_sqrt_spp = 0.25f;
    return cam;
}

// The sum of a pixel's 16 samples, added pairwise as a binary counter carries: sample `index` (0-15) waits in the
// lowest slot that is free, and two sums of 1, 2, 4 or 8 samples are added as soon as both are complete. Every
// addition then adds two sums of as many samples, so 16 equal samples sum to exactly 16 times one (each addition
// doubles a float) and their average, times 1/16, is exactly that sample: a pixel whose samples agree, the inside of a
// matte surface of one color, gets that color itself. (A running sum would round at 3 times, 5 times...) Named slots,
// not an array, so that they stay in registers.
struct pairwise_sum
{
    vec3<float> one;   // a sample waiting for its twin
    vec3<float> two;   // the sum of 2 samples waiting for the next 2
    vec3<float> four;  // of 4
    vec3<float> eight; // of 8
    vec3<float> total; // the 16, once sample 15 is in
};
static_assert(guide_passes == 16, "pairwise_sum adds 16 samples");

__device__ inline void add_pairwise(pairwise_sum& s, int index, vec3<float> x)
{
    if ((index & 1) == 0)
    {
        s.one = x;
        return;
    }
    x = s.one + x;
    if ((index & 2) == 0)
    {
        s.two = x;
        return;
    }
    x = s.two + x;
    if ((index & 4) == 0)
    {
        s.four = x;
        return;
    }
    x = s.four + x;
    if ((index & 8) == 0)
    {
        s.eight = x;
        return;
    }
    s.total = s.eight + x;
}

// Pixel (i, j)'s guides from its 16 samples, into albedo and normal (float4 per pixel, rows from the top): the albedo
// averaged and clamped to [0, 1], the normal averaged and made unit again unless it is 0 (every sample escaped, or
// opposite normals cancelled out). cam is the view's camera; sample `pass` draws from (seed, pixel, pass).
template <tracer T>
__device__ void write_guides(const T& tracer, const quality_view& view, const camera<float>& cam, int i, int j,
                             std::uint64_t seed, float4* albedo, float4* normal)
{
    const camera<float> grid = camera_for_guides(cam);
    const std::uint64_t pixel =
        static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(cam.image_width) + static_cast<std::uint64_t>(i);
    pairwise_sum albedo_sum{};
    pairwise_sum normal_sum{};
    for (int pass = 0; pass < guide_passes; ++pass)
    {
        pcg32 rng = sample_generator(seed, pixel, static_cast<std::uint64_t>(pass));
        const guide_texel t = guide_sample(tracer, view, grid, i, j, pass, rng);
        add_pairwise(albedo_sum, pass, t.albedo);
        add_pairwise(normal_sum, pass, t.normal);
    }
    const color<float> a = clamp_unit((1.0f / static_cast<float>(guide_passes)) * albedo_sum.total);
    const float length_squared = normal_sum.total.length_squared();
    const vec3<float> n = length_squared > 0.0f ? rsqrt(length_squared) * normal_sum.total : vec3<float>{};
    albedo[pixel] = make_float4(a.x, a.y, a.z, 1.0f);
    normal[pixel] = make_float4(n.x, n.y, n.z, 0.0f);
}
}
