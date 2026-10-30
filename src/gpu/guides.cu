// The denoiser's guides for the CUDA kernels' sessions (see denoise.cuh): guide_kernel<bvh_tracer>, guides.cuh's
// guide_sample over the renderer's own BVH, 16 samples per pixel. A translation unit of its own, as quality.cu is, so
// nvcc compiles renderer.cu's kernels as if it didn't exist. Track C writes the kernel in Step 94; until then the
// launcher only says so.
#include "denoise.cuh"

#include <stdexcept>

namespace gpu::detail
{
void launch_guides(const quality_scene* /*quality*/, const scene_view<float>& /*base*/, const camera<float>& /*cam*/,
                   std::uint64_t /*seed*/, float4* /*albedo*/, float4* /*normal*/)
{
    throw std::logic_error("the denoiser's guides come in Step 94");
}
}
