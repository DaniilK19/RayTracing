// guides.cuh: the denoiser's guides at one sample of pixel (i, j): the albedo and normal of the first surface that
// isn't a mirror or glass, along the camera ray the beauty's pass starts with. One template over the tracer concept:
// guides.cu instantiates it with bvh_tracer, OptiX's __raygen__guides with optix_tracer. (Bodies: track C.)
#pragma once
#include "quality_path.cuh"

namespace gpu
{
struct guide_texel
{
    color<float> albedo; // in [0, 1]
    vec3<float> normal;  // world space, unit, or 0 where the ray escaped
};

template <tracer T>
__device__ guide_texel guide_sample(const T& tracer, const quality_view& view, const camera<float>& cam, int i, int j,
                                    int pass, pcg32& rng) // stub: returns zeros
{
    return {};
}

// A copy of cam whose grid is 4×4 (sqrt_spp 4), so passes 0–15 cover each pixel as the beauty's samples do
__host__ __device__ inline camera<float> camera_for_guides(camera<float> cam)
{
    cam.sqrt_spp = 4;
    cam.recip_sqrt_spp = 0.25f;
    return cam;
}
}
