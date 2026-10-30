// __raygen__pick: the window's double-click on a scene with meshes. One thread: the surfaces' closest hit of
// params.pick_ray through optix_tracer's traversal, without the media, its t (in lengths of the ray's direction, ∞ for
// none) written to *params.pick_t.
//
// Track B's file (Step 91). Until then an empty body: no scene is uploaded before Step 88, so nothing launches it.
#pragma once
#include <optix.h>
#include "optix_tracer.cuh"

extern "C" __global__ void __raygen__pick()
{
}
