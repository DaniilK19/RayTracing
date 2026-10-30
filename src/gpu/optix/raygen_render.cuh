// __raygen__render: the quality kernel's loop (quality.cu:58-108) with optix_tracer in place of bvh_tracer. One thread
// per pixel of the launch's band, its pixel from optixGetLaunchIndex() (row first_row + y: the launch is exact, so no
// bounds test), the same pixel index and sample_generator, the NaN rule, the double sums read and written once per
// launch, and two atomics per thread for the rays and shadow rays (no warp sums: a warp's threads aren't kept
// together).
//
// Track A's file (Step 89). Until then an empty body: no scene is uploaded before Step 88, so nothing launches it.
#pragma once
#include <optix.h>
#include "optix_tracer.cuh"
#include "quality_path.cuh"

extern "C" __global__ void __raygen__render()
{
}
