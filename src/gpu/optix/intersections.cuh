// The books' primitives in OptiX: the intersection programs of the books' GAS, one per kind of primitive (its
// sbtIndexOffset 0, 1, 2), the leaf bodies of closest_hit (bvh.cuh) and occluded (shadow.cuh) on the ray OptiX hands
// them, with the integrator's skips in the payload (optix_tracer.cuh's: from, inward, skip, the ray's time, 1/|d|²).
//
// Track A's file (Step 88). Until then the programs exist, so the pipeline links, and report nothing: no scene is
// uploaded before Step 88 (optix_runtime::upload says so), so no ray reaches them.
#pragma once
#include <optix.h>
#include "optix_params.h"

extern "C" __global__ void __intersection__sphere()
{
}

extern "C" __global__ void __intersection__moving_sphere()
{
}

extern "C" __global__ void __intersection__quad()
{
}
