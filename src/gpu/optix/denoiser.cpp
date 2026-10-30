// OptiX's AI denoiser (runtime_core.h's denoise): the AOV model with albedo and normal guides, made at the first call,
// its intensity and average color computed before each invoke, images above 2048 a side in tiles, destroyed before
// the context (core_deleter). Compiled by cl.exe with OptiX's host API (core.h).
//
// Track C's file (Step 94). Until then the state is empty and a denoise says when it comes.
#include "core.h"
static_assert(OPTIX_VERSION >= 90000); // NVIDIA's optix.h, not a project header of that name

#pragma warning(push, 0)
#include <optix_denoiser_tiling.h> // optixUtilDenoiserInvokeTiled (includes <optix.h>: no project header may shadow it)
#pragma warning(pop)

#include <stdexcept>

namespace gpu::optix_core
{
// The denoiser and what it keeps between calls: its handle, state and scratch, the sizes it was set up for (Step 94)
struct denoiser_state
{
};

void denoiser_deleter::operator()(denoiser_state* state) const noexcept
{
    delete state;
}

void denoise(core& /*c*/, const void* /*beauty*/, const void* /*albedo*/, const void* /*normal*/, void* /*out*/,
             int /*width*/, int /*height*/)
{
    throw std::logic_error("OptiX's denoiser comes in Step 94");
}

denoise_report last_denoise(const core& /*c*/)
{
    return {}; // valid false: no denoise yet (Step 94)
}
}
