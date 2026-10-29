// What a ray of the quality kernel that hits nothing sees: the camera's sky, or the scene's environment map, and the
// environment as a light the light sampler can draw from (light_sampler.cuh).
//
// The map is seen, not yet sampled: no light sample comes from it (environment_sampled is false without its alias
// table), so a ray that escapes finds all of its light with weight 1.
#pragma once

#include "camera.cuh"
#include "environment.cuh"
#include "quality_view.cuh"
#include "random.cuh"

namespace gpu
{
// filtered: camera rays and rays after a delta or pdf-less vertex, which see the map bilinearly filtered; the rays MIS
// combines see its nearest texel, the one its density is constant over. The caller says which (quality_path::specular,
// a property of the material, never inferred from a density: under -p material no vertex stores one, and its escapes
// must see the map the light samples of -p mis see, or the two strategies would render two environments). Without a
// map, the camera's sky: the books' arithmetic exactly.
__device__ inline color<float> background(const quality_view& v, const camera<float>& cam, const vec3<float>& dir,
                                          bool filtered)
{
    const environment_view& e = v.environment;
    if (e.texels == nullptr)
    {
        return cam.sky(dir);
    }
    if (filtered)
    {
        return env_radiance_filtered(e, dir);
    }
    float pdf; // the light sampler's density of dir: the caller weighs the light with it (environment_hit_pdf)
    return env_radiance(e, dir, pdf);
}

// A map with light in it (S > 0) and its alias table: a light of the set (a black map gets no table)
__device__ inline bool environment_sampled(const environment_view& e)
{
    return e.texels != nullptr && e.bins != nullptr;
}

// A direction drawn from the map by its brightness, its radiance and its density per steradian (no pmf); 4 numbers.
// The map isn't sampled yet: false, and no number drawn.
__device__ inline bool sample_environment(const environment_view& /*e*/, pcg32& /*rng*/, vec3<float>& /*wi*/,
                                          color<float>& /*le*/, float& /*pdf*/)
{
    return false;
}

// The density sample_environment draws a direction with, per steradian, through env_cell (the cell the escapes see).
// The map isn't sampled yet: 0.
__device__ inline float environment_pdf(const environment_view& /*e*/, const vec3<float>& /*dir*/)
{
    return 0.0f;
}
}
