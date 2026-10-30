// The denoiser's images for renderer.cu (see denoise.cuh): beauty_kernel, the session's sums as the float4 image
// OptiX's denoiser reads, and widen_kernel, its result back as sums for the display's existing kernels. A translation
// unit of its own, as quality.cu is, so nvcc compiles renderer.cu's kernels as if it didn't exist. Track C writes the
// kernels in Step 94; until then the launchers only say so.
#include "denoise.cuh"

#include <stdexcept>

namespace gpu::detail
{
void to_beauty(const double* /*sums*/, int /*width*/, int /*first_row*/, int /*row_count*/, double /*scale*/,
               float4* /*beauty*/)
{
    throw std::logic_error("the denoiser's beauty image comes in Step 94");
}

void to_sums(const float4* /*image*/, int /*width*/, int /*height*/, double* /*sums*/)
{
    throw std::logic_error("the denoised image's sums come in Step 94");
}
}
