// __raygen__guides: the denoiser's albedo and normal guides of an OptiX session, guide_sample<optix_tracer>
// (guides.cuh) over passes 0-15 of camera_for_guides(params.cam) per pixel, seeded with params.guide_seed, written to
// params.albedo and params.normal. The launch is banded as the render's (scene.cu): the pixel's row is first_row plus
// the launch index's y.
//
// Track C's file (Step 94). Until then an empty body: no scene is uploaded before Step 88, so nothing launches it.
#pragma once
#include <optix.h>
#include "guides.cuh"
#include "optix_tracer.cuh"

extern "C" __global__ void __raygen__guides()
{
}
