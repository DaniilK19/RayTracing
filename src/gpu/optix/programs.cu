// The OptiX programs of RayTracingGPU and RayTracingViewer: one module, compiled to PTX by nvcc and again by the
// graphics driver at the first run (OptiX caches the result). Its raygen programs run the quality kernel's integrator
// (quality_path.cuh, unchanged) with OptiX's tracer; its intersection programs are the books' own tests.
#include <optix.h>
static_assert(OPTIX_VERSION >= 90000); // NVIDIA's optix.h, not a project header of that name
#include "optix_params.h"

// The SDK's braced form: a definition. The unbraced `extern "C" __constant__ …` is a declaration, which nvcc's
// whole-program mode (no -rdc) warns about (20044) and treats as a static definition. Value-initialized
// ({}): optix_params holds members with default initializers (vec3's zeros, mesh_tables::triangle_base...), and nvcc
// takes a __constant__ variable only with a constant initialization, never a default constructor that isn't empty.
// OptiX replaces the variable's contents at every launch.
extern "C"
{
__constant__ gpu::optix_params params{};
}

#include "intersections.cuh" // __intersection__sphere, __intersection__moving_sphere, __intersection__quad (A, 88)
#include "optix_tracer.cuh"
#include "raygen_render.cuh" // __raygen__render (A, 89): quality.cu:58-108 with optix_tracer
#include "raygen_guides.cuh" // __raygen__guides (C, 94): guide_sample<optix_tracer>, 16 passes per pixel
#include "raygen_pick.cuh"   // __raygen__pick (B, 91): the surfaces' closest hit of params.pick_ray
