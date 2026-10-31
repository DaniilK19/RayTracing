// __raygen__guides: the denoiser's albedo and normal guides of an OptiX session, guide_sample<optix_tracer>
// (guides.cuh) over passes 0-15 of camera_for_guides(params.cam) per pixel, seeded with params.guide_seed, written to
// params.albedo and params.normal: the CUDA sessions' guide_kernel (guides.cu) with OptiX's tracer, so a mesh scene's
// triangles and glTF materials (the pixel's scratch slot) are seen as the render sees them. The launch is banded as
// the render's (scene.cu): the pixel's row is first_row plus the launch index's y, and the launch is exact (no bounds
// test).
#pragma once
#include <cstdint>

#include <optix.h>

#include "guides.cuh"
#include "optix_tracer.cuh"

extern "C" __global__ void __raygen__guides()
{
    const uint3 index = optixGetLaunchIndex();
    const int i = static_cast<int>(index.x);
    const int j = params.first_row + static_cast<int>(index.y);
    const std::uint64_t pixel = static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(params.cam.image_width) +
                                static_cast<std::uint64_t>(i);
    // The tracer's side record lives in this frame; the slot is the pixel's (unused when scratch_base is -1)
    gpu::triangle_memo memo;
    const gpu::optix_tracer tracer{params, memo, static_cast<int>(params.meshes.scratch_base + pixel)};
    gpu::write_guides(tracer, params.view, params.cam, i, j, params.guide_seed, params.albedo, params.normal);
}
