// The quality kernel's path: one sample's rays, as the kernel for all three books traces them (trace_ray in
// camera.cuh), plus what light sampling needs. A path of the books keeps only its throughput, because it finds light
// once, where it ends (a lamp or the sky). A path that also samples the lights at its bounces adds their light as it
// goes, so it keeps a running sum, and for each ray the density the material drew it with, which weighs the light that
// ray finds against the light samples (mis.cuh).
//
// Under -p material (the base mode) every weight is 1 and the radiance stays 0 until the path ends, so the result is
// the books' throughput times the light found, the same numbers in the same order as trace_ray's.
#pragma once

#include "background.cuh"
#include "bsdf.cuh"
#include "camera.cuh"
#include "light_sampler.cuh"
#include "mis.cuh"
#include "quality_view.cuh"
#include "tracer.cuh"

namespace gpu
{
struct quality_path
{
    path<float> base;      // r, throughput, rays_left, from, inward (camera.cuh): embedded, unchanged
    color<float> radiance; // the light found so far: light samples and weighted hits of lamps and the sky
    float bsdf_pdf;        // the density base.r.direction was drawn with at a vertex that also took a light sample;
                           // 0 after the camera, a delta vertex or a vertex without one: its next light counts in full
    bool specular;         // base.r left the camera or a vertex without a density: the rays that see the map filtered
};

// The path of pass `pass` of pixel (i, j): the books' camera ray (and its time, in a scene where something moves)
__device__ inline quality_path start_quality_sample(const camera<float>& cam, const quality_view& view, int i, int j,
                                                    int pass, pcg32& rng)
{
    return {.base = start_sample<float, kernel_kind::all_books>(cam, view.base, i, j, pass, rng),
            .radiance = {0.0f, 0.0f, 0.0f},
            .bsdf_pdf = 0.0f,
            .specular = true};
}

// One ray of the path: true when the sample is complete, its light in `sample`; otherwise the path holds the ray to
// trace next. shadow_rays counts the light samples' rays the step traced.
template <tracer T>
__device__ bool trace_quality_ray(quality_path& q, const T& tracer, const quality_view& view, const camera<float>& cam,
                                  pcg32& rng, color<float>& sample, unsigned int& shadow_rays)
{
    (void)shadow_rays; // no light samples yet
    path<float>& p = q.base;
    const bvh_hit<float> hit = tracer.closest(p.r, spawn{p.from, p.inward}, rng);
    if (hit.index < 0)
    {
        // The sky or the environment, weighted against the light samples that could have drawn this direction. Both
        // look up the very same vector, the ray's direction as it is (of any length): the map's cell of the radiance is
        // then the cell of the density.
        const float w = bsdf_weight(view.combination, q.bsdf_pdf, environment_hit_pdf(view, p.r.direction));
        sample = q.radiance + w * (p.throughput * background(view, cam, p.r.direction, q.specular));
        return true;
    }
    const surface<float> s = tracer.surface_of(p.r, hit);
    const material<float> mat = view.base.materials[s.material];
    // The material's color, as trace_ray computes it: a matte surface, a particle of smoke, or a lamp seen from its
    // front (textures draw no random numbers)
    const bool colored = mat.kind == material_kind::lambertian || mat.kind == material_kind::isotropic ||
                         (mat.kind == material_kind::diffuse_light && s.rec.front_face);
    const color<float> c = colored ? surface_color(view.base, mat, s) : color<float>{};

    if (mat.kind == material_kind::diffuse_light)
    {
        // A lamp: its light, weighted against the light sample of this very lamp from the previous vertex
        // (r.origin), and the path ends (a lamp reflects nothing; its back emits nothing)
        const float distance = hit.t * sqrt(p.r.direction.length_squared());
        const float w = bsdf_weight(view.combination, q.bsdf_pdf,
                                    lamp_hit_pdf(view, hit.index, p.r.origin, unit_vector(p.r.direction), distance));
        sample = q.radiance + w * (p.throughput * c);
        return true;
    }

    const microfacet* side = view.microfacets ? &view.microfacets[s.material] : nullptr;
    const bsdf b = make_bsdf(mat, side, c, s.rec, p.r.direction);
    // A light sample stands for the next ray, so it is taken only where the path may still trace one (the books'
    // counting: rays_left before the decrement below). Its code comes with the light samples.
    const bool nee =
        view.combination != light_combination::bsdf_only && p.rays_left > 1 && light_count(view) > 0 && !is_delta(b);

    bsdf_sample next;
    if (!sample_bsdf(b, p.r, s.rec, rng, next))
    {
        sample = q.radiance; // absorbed: only the light found so far
        return true;
    }
    p.throughput *= next.weight;
    q.bsdf_pdf = nee ? next.pdf : 0.0f;
    q.specular = next.delta;
    if (--p.rays_left == 0)
    {
        sample = q.radiance; // the path may trace no more rays (the CPU's depth reached 0)
        return true;
    }
    // A bounce keeps the ray's time (material.ixx for metal and glass, ray_color in camera.ixx for a diffuse bounce)
    p.r = {s.rec.p, next.wi, p.r.time};
    p.from = s.from;
    p.inward = dot(next.wi, s.outward) < 0.0f; // read only when the ray leaves a sphere (see closest_hit)
    return false;
}
}
