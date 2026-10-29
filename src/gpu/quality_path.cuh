// The quality kernel's path: one sample's rays, as the kernel for all three books traces them (trace_ray in
// camera.cuh), plus light sampling. A path of the books keeps only its throughput, because it finds light once, where
// it ends (a lamp or the sky). A path that also samples the lights at its bounces adds their light as it goes, so it
// keeps a running sum, and for each ray the density the material drew it with, which weighs the light that ray finds
// against the light samples (mis.cuh).
//
// The estimator is Veach's multi-sample model (mis.cuh): at every vertex with a density one light sample, a shadow ray
// to a point of a light (next-event estimation), and one sample of the material, which is also the path's next ray.
// Their two weights depend on the combination: -p material takes no light sample (the base mode), -p nee counts the
// light samples in full and the light the material's rays find only where no light sample could have found it, -p mis
// weighs both by the power heuristic. All three have the same expected image, the books' -p material's: the path
// integral over paths of at most max_depth rays. Under -p material every weight is 1 and the radiance stays 0 until
// the path ends, so the result is the books' throughput times the light found, the same numbers in the same order as
// trace_ray's.
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

// The largest of a color's channels: above 0 when any of them is
__device__ inline float largest_channel(const color<float>& c)
{
    return fmax(fmax(c.x, c.y), c.z);
}

// The light sample of a vertex x with throughput β before it (next-event estimation): a direction toward a light of the
// set, and what that light brings if nothing blocks it, w_L·f·Le/p_L ⊙ β, its weight against the material's own sample
// of the same direction included. True when it brings something: then `contribution` is ready, and `ls` holds the
// shadow ray's direction and length and the lamp it mustn't count as a blocker.
//
// Computed whole before the shadow ray is traced: the light's radiance, both densities and the weight die here, and
// only the shadow ray and three numbers wait through its traversal, in registers. A contribution that isn't finite (a
// density that overflowed, a NaN) is dropped here, where it costs one light sample: once added, it would spoil the
// path's whole sum.
__device__ inline bool light_contribution(const quality_view& view, const bsdf& b, const color<float>& throughput,
                                          const point3<float>& x, pcg32& rng, light_sample& ls,
                                          color<float>& contribution)
{
    if (!sample_light(view, x, rng, ls))
    {
        return false; // the back of a quad, the inside of a sphere: no light from there, and f·Le = 0 (W1 holds)
    }
    float p_bsdf;
    const color<float> f = eval_bsdf(b, ls.wi, p_bsdf); // f·|cos θ|, and the density the material draws ls.wi with
    if (!(ls.pdf > 0.0f) || !(largest_channel(f) > 0.0f))
    {
        return false; // below the surface: the material reflects none of it
    }
    contribution = (light_weight(view.combination, ls.pdf, p_bsdf) / ls.pdf) * (throughput * (f * ls.le));
    // The channels are ≥ 0, so their sum is finite and above 0 exactly when each one is finite and one is above 0
    const float sum = contribution.x + contribution.y + contribution.z;
    return sum > 0.0f && sum < infinity<float>;
}

// One ray of the path: true when the sample is complete, its light in `sample`; otherwise the path holds the ray to
// trace next. shadow_rays counts the light samples' rays the step traced.
template <tracer T>
__device__ bool trace_quality_ray(quality_path& q, const T& tracer, const quality_view& view, const camera<float>& cam,
                                  pcg32& rng, color<float>& sample, unsigned int& shadow_rays)
{
    path<float>& p = q.base;
    const bvh_hit<float> hit = tracer.closest(p.r, spawn{p.from, p.inward}, rng);
    if (hit.index < 0)
    {
        // The sky or the environment, weighted against the light samples that could have drawn this direction (only
        // after a vertex that took one: bsdf_pdf is 0 after the camera and under -p material). The radiance and the
        // map's density come from one lookup of the ray's direction as it is (of any length), the same texel.
        float map_pdf;
        const color<float> le = background(view, cam, p.r.direction, q.specular, map_pdf);
        const float w =
            q.bsdf_pdf > 0.0f ? bsdf_weight(view.combination, q.bsdf_pdf, environment_hit_pdf(view, map_pdf)) : 1.0f;
        sample = q.radiance + w * (p.throughput * le);
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
        // A lamp: its light, weighted against the light sample that could have made this very path, the one of this
        // lamp from the previous vertex, r.origin: the very float point that sample was drawn from, so the light
        // sampler's density comes out as it was there (path-space MIS, lamps.cuh). Full weight after a vertex that took
        // no light sample (bsdf_pdf 0), and for a lamp outside the set (lamp_hit_pdf 0). The path ends: a lamp reflects
        // nothing, and its back emits nothing (c is 0 there).
        float w = 1.0f;
        if (q.bsdf_pdf > 0.0f)
        {
            const float distance = hit.t * sqrt(p.r.direction.length_squared());
            w = bsdf_weight(view.combination, q.bsdf_pdf,
                            lamp_hit_pdf(view, hit.index, p.r.origin, unit_vector(p.r.direction), distance));
        }
        sample = q.radiance + w * (p.throughput * c);
        return true;
    }

    const microfacet* side = view.microfacets ? &view.microfacets[s.material] : nullptr;
    const bsdf b = make_bsdf(mat, side, c, s.rec, p.r.direction);
    // A light sample here? Not under -p material, and not where the material has no density (a delta vertex: no light
    // sample could ever find the one direction it reflects into). A particle of smoke takes one too, with its phase
    // function albedo/(4π) for f, where the books' mixture keeps a particle's own samples (isotropic::mixes_in_lights
    // is false): a choice of noise, the same expected image, and thin haze gets its light from the lamps at every
    // particle. The depth rule: a light sample stands for the next ray, a path one ray longer, so it is taken only
    // where the path may still trace one, rays_left > 1 before the decrement below (the books' counting). A light
    // sample at the last vertex would add paths of max_depth + 1 rays, which -p material never traces, and the
    // strategies would no longer converge to the same image.
    const bool nee = view.combination != light_combination::bsdf_only && p.rays_left > 1 && !is_delta(b) &&
                     light_count(view) > 0 && largest_channel(p.throughput) > 0.0f;
    light_sample ls;
    color<float> contribution{};
    const bool lit = nee && light_contribution(view, b, p.throughput, s.rec.p, rng, ls, contribution);

    // The material's sample, the path's next ray, drawn with the numbers after the light sample's. The shadow ray is
    // traced after it: it draws no numbers, so the order of the numbers stays the light's, then the material's, and
    // the material's state (the lobe, the hit record) dies before the second traversal instead of waiting through it.
    bsdf_sample next;
    const bool scattered = sample_bsdf(b, p.r, s.rec, rng, next);
    if (lit)
    {
        // From the vertex as the path's next ray leaves it (the self-intersection rules of closest_hit: the primitive
        // it leaves, and whether it goes into that sphere; a particle leaves none, and its outward normal is 0), at
        // the path's time, so a moving sphere or boundary is where the path saw it; up to the sampled point of the
        // light, the lamp itself skipped; through the smoke and fog on the way (bvh_tracer::visibility)
        ++shadow_rays;
        const ray<float> shadow{s.rec.p, ls.wi, p.r.time};
        const float visible =
            tracer.visibility(shadow, spawn{s.from, dot(ls.wi, s.outward) < 0.0f}, ls.distance, ls.skip);
        q.radiance = q.radiance + visible * contribution;
    }
    if (!scattered)
    {
        sample = q.radiance; // absorbed: only the light found so far
        return true;
    }
    p.throughput *= next.weight;
    q.bsdf_pdf = nee ? next.pdf : 0.0f;
    q.specular = next.delta;
    // A path whose throughput is exactly 0 can't find any more light: a particle of the smoke scene's black block, of
    // albedo 0. Under -p nee and -p mis it ends here, with the same image and fewer rays (the particle's light sample,
    // whose f is 0 too, traced no shadow ray). -p material traces on, as the books' kernels do, so its rays per sample
    // stay theirs.
    if (view.combination != light_combination::bsdf_only && p.throughput.x == 0.0f && p.throughput.y == 0.0f &&
        p.throughput.z == 0.0f)
    {
        sample = q.radiance;
        return true;
    }
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
