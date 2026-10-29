// The quality kernel's light sampling: the lamps (lamps.cuh) and the environment map (background.cuh) as one set, one
// of them picked uniformly per light sample, which draws a direction toward itself. The integrator (quality_path.cuh)
// traces a shadow ray along it and weighs the light against the material's own sample of the same direction (mis.cuh).
//
// Uniform selection, the chance light_pmf = 1/N for each of the N lights, computed on the host in double: the scenes
// have at most a few lamps, and a choice by power (pbrt-v4's PowerLightSampler) would starve some of them. In the light
// scene the sphere lamp emits 4π times as much as the quad (π·area·L for a lamp that glows on its outside: 64π² against
// 16π), so it would get 93% of the samples, while the quad gives a point 3 units in front of it an irradiance of 1.55,
// and the sphere one of 1.03 to a point 7 units below its center. A density counts the pmf on both sides of the
// weights: in a light sample's pdf here, and in the density of a lamp or of the environment that a ray of the material
// finds (lamp_hit_pdf, environment_hit_pdf). Forgotten on one side, the weights no longer add up to 1.
#pragma once

#include "background.cuh"
#include "lamps.cuh"
#include "quality_view.cuh"
#include "random.cuh"
#include "vec3.cuh"

namespace gpu
{
struct light_sample
{
    vec3<float> wi;  // unit, from x toward the light
    float distance;  // the shadow ray's t_max: to the sampled point; infinity for the environment
    color<float> le; // the radiance arriving along -wi, before the shadow ray and the media
    float pdf;       // the light technique's density of wi: light_pmf · the light's own density
    int skip;        // the lamp's global id (-1 for the environment)
};

// How many lights the set holds: the lamps, and the environment when its map is sampled
__device__ inline int light_count(const quality_view& v)
{
    return v.lamps.count + (environment_sampled(v.environment) ? 1 : 0);
}

// A light sample from x; false when the picked light can't light x (the back of a quad, the inside of a sphere), or
// the set is empty. One number picks the light, ⌊ξ·N⌋, the lamps first and the environment last, drawn also when the
// set holds one light, so the numbers that follow are always the same ones: 3 for a lamp (lamps.cuh), 5 for the map.
__device__ inline bool sample_light(const quality_view& v, const point3<float>& x, pcg32& rng, light_sample& out)
{
    const int n = light_count(v);
    if (n == 0)
    {
        return false;
    }
    const float xi = random_real<float>(rng);
    int k = static_cast<int>(xi * static_cast<float>(n));
    k = k < n ? k : n - 1; // ξ < 1, but ξ·N can round up to N
    if (k < v.lamps.count)
    {
        const emitter& e = v.lamps.lamps[k];
        lamp_sample s;
        if (!sample_lamp(e, x, rng, s))
        {
            return false;
        }
        out.wi = s.wi;
        out.distance = s.distance;
        out.le = e.radiance;
        out.pdf = v.light_pmf * s.pdf;
        out.skip = e.primitive;
        return true;
    }
    // The environment: infinitely far, behind every primitive, so the shadow ray goes all the way and skips none
    float pdf;
    if (!sample_environment(v.environment, rng, out.wi, out.le, pdf))
    {
        return false;
    }
    out.distance = infinity<float>;
    out.pdf = v.light_pmf * pdf;
    out.skip = -1;
    return true;
}

// The light technique's density (pmf included) of the direction from x along which a ray hit the lamp with this
// global id at this distance; 0: no lamp of the set (a textured or moving lamp, or a set without lamps, whose lamp_of
// isn't uploaded). x is the ray's origin, the very point the previous vertex's light sample was drawn from.
__device__ inline float lamp_hit_pdf(const quality_view& v, int global_id, const point3<float>& x,
                                     const vec3<float>& unit_dir, float distance)
{
    if (v.lamps.count == 0)
    {
        return 0.0f;
    }
    const int k = v.lamps.lamp_of[global_id];
    return k >= 0 ? v.light_pmf * lamp_pdf(v.lamps.lamps[k], x, unit_dir, distance) : 0.0f;
}

// The same for a ray that escaped toward the environment, from the map's density of the texel its radiance came from
// (background()'s, 0 when the map isn't sampled): the integrator's escapes take it so, from one lookup
__device__ inline float environment_hit_pdf(const quality_view& v, float map_pdf)
{
    return v.light_pmf * map_pdf;
}

// The same for a direction alone (of any length), through env_cell: what an escape along dir gets
__device__ inline float environment_hit_pdf(const quality_view& v, const vec3<float>& dir)
{
    return environment_hit_pdf(v, environment_sampled(v.environment) ? environment_pdf(v.environment, dir) : 0.0f);
}
}
