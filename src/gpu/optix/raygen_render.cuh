// __raygen__render: the quality kernel's loop (quality.cu:58-108) with optix_tracer in place of bvh_tracer. One thread
// per pixel of the launch's band, its pixel from optixGetLaunchIndex() (row first_row + y: the launch is exact, so no
// bounds test), the same pixel index and sample_generator, the NaN rule, the double sums read and written once per
// launch, and two atomics per thread for the rays and shadow rays (no warp sums: a warp's threads aren't kept
// together).
//
// `params` is the module's: programs.cu defines it before it includes this file. Track A's file (Step 89).
#pragma once
#include <cstdint>
#include <optix.h>
#include "optix_tracer.cuh"
#include "quality_path.cuh"
#include "random.cuh"

namespace gpu
{
// quality.cu's quality_without_nan, for this program: a NaN channel of a sample counts as 0
__device__ inline double optix_without_nan(float x)
{
    return static_cast<double>(x == x ? x : 0.0f);
}
}

// One thread per pixel, as quality_kernel: the launch's passes of the pixel in one loop, one ray per iteration, the
// next sample starting as soon as one ends, the sums in double read and written once per launch. counts[0] gets the
// rays traced (closest-hit searches, as the books count them), counts[1] the shadow rays: each thread adds its own with
// one atomic each, integers, so the totals don't depend on the order. (The quality kernel first adds a warp's counts
// with shuffles; OptiX doesn't keep a warp's threads together, OptiX Programming Guide §6.2.)
extern "C" __global__ void __raygen__render()
{
    const uint3 index = optixGetLaunchIndex();
    const int i = static_cast<int>(index.x);
    const int j = params.first_row + static_cast<int>(index.y);
    const std::uint64_t pixel = static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(params.cam.image_width) +
                                static_cast<std::uint64_t>(i);
    double* const pixel_sum = params.sums + 3 * pixel;
    double red = pixel_sum[0];
    double green = pixel_sum[1];
    double blue = pixel_sum[2];

    // The tracer's side record lives in this frame; the pixel's scratch slot serves a glTF material (unused while
    // scratch_base is -1: no glTF material in the scene)
    gpu::triangle_memo memo;
    const gpu::optix_tracer tracer{params, memo, static_cast<int>(params.meshes.scratch_base + pixel)};
    unsigned long long rays = 0;
    unsigned int shadow_rays = 0;
    int pass = params.first_pass;
    const int end_pass = params.first_pass + params.pass_count;
    gpu::pcg32 rng = gpu::sample_generator(params.seed, pixel, static_cast<std::uint64_t>(pass));
    gpu::quality_path q = gpu::start_quality_sample(params.cam, params.view, i, j, pass, rng);
    while (true)
    {
        ++rays;
        gpu::color<float> sample;
        if (!gpu::trace_quality_ray(q, tracer, params.view, params.cam, rng, sample, shadow_rays))
        {
            continue; // the path goes on
        }
        red += gpu::optix_without_nan(sample.x);
        green += gpu::optix_without_nan(sample.y);
        blue += gpu::optix_without_nan(sample.z);
        if (++pass == end_pass)
        {
            break;
        }
        rng = gpu::sample_generator(params.seed, pixel, static_cast<std::uint64_t>(pass));
        q = gpu::start_quality_sample(params.cam, params.view, i, j, pass, rng);
    }
    pixel_sum[0] = red;
    pixel_sum[1] = green;
    pixel_sum[2] = blue;

    atomicAdd(&params.counts[0], rays);
    atomicAdd(&params.counts[1], static_cast<unsigned long long>(shadow_rays));
}
