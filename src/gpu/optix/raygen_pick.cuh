// __raygen__pick: the window's double-click on a scene with meshes. One thread: the surfaces' closest hit of
// params.pick_ray through optix_tracer's traversal, without the media, its t (in lengths of the ray's direction, ∞ for
// none) written to *params.pick_t.
//
// Track B's file (Step 91).
#pragma once
#include <optix.h>
#include "optix_tracer.cuh"

// The tracer of the image itself, so a click finds what the image shows: the same traversal, the same intersection
// programs, the same triangles. Its closest() runs on a copy of the parameters whose media are empty, as a click
// focuses on a surface, never on smoke (the CPU world's pick in the window does the same). The ray leaves nothing (from
// -1), so its traversal starts at its own origin with the books' tmin 0.001, and no random number is drawn but a glTF
// triangle's lobe number, which nothing reads here.
extern "C" __global__ void __raygen__pick()
{
    gpu::optix_params surfaces = params;
    surfaces.view.base.media.count = 0;
    gpu::triangle_memo memo;
    const gpu::optix_tracer tracer{surfaces, memo, -1};
    gpu::pcg32 rng{0, 0};
    const gpu::bvh_hit<float> hit = tracer.closest(params.pick_ray, gpu::spawn{-1, false}, rng);
    *params.pick_t = hit.index < 0 ? gpu::infinity<float> : hit.t;
}
