// The denoiser's guides for the CUDA kernels' sessions (see denoise.cuh): guide_kernel<bvh_tracer>, guides.cuh's
// guide_sample over the renderer's own BVH, 16 samples per pixel. A translation unit of its own, as quality.cu is, so
// nvcc compiles renderer.cu's kernels as if it didn't exist. (OptiX's sessions compute the same guides with OptiX's
// tracer: __raygen__guides, src/gpu/optix/raygen_guides.cuh.)
#include "denoise.cuh"

#include <cstdint>

#include <cuda_runtime.h>

#include "device_memory.h"
#include "guides.cuh"
#include "quality_access.cuh"
#include "quality_view.cuh"
#include "tracer.cuh"

namespace gpu::detail
{
// (Not in an unnamed namespace: see quality.cu.)

// The quality kernel's blocks of 16×8 pixels: the same integrator's code, the same shape
constexpr int guide_block_width = 16;
constexpr int guide_block_height = 8;

// One thread per pixel of the whole view: write_guides' 16 samples through the tracer T over the view
template <tracer T>
__global__ void __launch_bounds__(guide_block_width * guide_block_height)
    guide_kernel(const camera<float> cam, const quality_view view, const std::uint64_t seed,
                 float4* __restrict__ albedo, float4* __restrict__ normal)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= cam.image_width || j >= cam.image_height)
    {
        return; // no warp sums here: a thread outside the image has nothing to do
    }
    const T scene_tracer{view};
    write_guides(scene_tracer, view, cam, i, j, seed, albedo, normal);
}

void launch_guides(const quality_scene* quality, const scene_view<float>& base, const camera<float>& cam,
                   std::uint64_t seed, float4* albedo, float4* normal)
{
    // A quality session's view (side entries, lamps, the map the sky comes from); for the books' kernels their tables
    // alone, with no side entries, lamps or map, which is exactly how the quality kernel reads a books' scene: the
    // books' sky, the books' lobes
    const quality_view view = quality ? quality_view_of(*quality, base) : quality_view{.base = base};
    // Blocks 8 rows high: at most 65535 of them down, as CUDA allows, since validate_camera in renderer.cu allows at
    // most 65535 × 8 rows
    const dim3 block(guide_block_width, guide_block_height);
    const dim3 grid(static_cast<unsigned>((cam.image_width + guide_block_width - 1) / guide_block_width),
                    static_cast<unsigned>((cam.image_height + guide_block_height - 1) / guide_block_height));
    guide_kernel<bvh_tracer><<<grid, block>>>(cam, view, seed, albedo, normal);
    // A launch that couldn't start shows here, one that fails while it runs when the host waits for it
    check(cudaGetLastError(), "launching the denoiser's guides");
}
}
