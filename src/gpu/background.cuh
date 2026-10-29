// What a ray of the quality kernel that hits nothing sees: the camera's sky, or the scene's environment map, and the
// environment as a light the light sampler can draw from (light_sampler.cuh).
//
// A stub until the environment comes: the camera's sky, and no map to sample.
#pragma once

#include "camera.cuh"
#include "quality_view.cuh"

namespace gpu
{
// filtered: camera rays and rays after a delta or pdf-less vertex, which see the map bilinearly filtered; the rays MIS
// combines see its nearest texel, the one its density is constant over
__device__ inline color<float> background(const quality_view& /*v*/, const camera<float>& cam, const vec3<float>& dir,
                                          bool /*filtered*/)
{
    return cam.sky(dir);
}

// A map with light in it (S > 0) and its alias table
__device__ inline bool environment_sampled(const environment_view& /*e*/)
{
    return false;
}

// A direction drawn from the map by its brightness, its radiance and its density per steradian (no pmf); 4 numbers
__device__ inline bool sample_environment(const environment_view& /*e*/, pcg32& /*rng*/, vec3<float>& /*wi*/,
                                          color<float>& /*le*/, float& /*pdf*/)
{
    return false;
}

// The density sample_environment draws a direction with, per steradian
__device__ inline float environment_pdf(const environment_view& /*e*/, const vec3<float>& /*dir*/)
{
    return 0.0f;
}
}
