// The denoiser's images for renderer.cu (see denoise.cuh): beauty_kernel, the session's sums as the float4 image
// OptiX's denoiser reads, and widen_kernel, its result back as sums for the display's existing kernels. A translation
// unit of its own, as quality.cu is, so nvcc compiles renderer.cu's kernels as if it didn't exist.
//
// The display of a denoised image adds no display code: the denoised floats, widened to doubles, are sums that
// to_rgba8 (display.cu) and to_rgba8_toned (tonemap.cu) convert at a scale of 1.0, and static_cast<float>(double(f)
// · 1.0) is f exactly, so their bytes are the bytes of the denoised floats themselves, through the very code that shows
// the undenoised image.
#include "denoise.cuh"

#include <cstdint>

#include <cuda_runtime.h>

#include "device_memory.h"

namespace gpu::detail
{
// (Not in an unnamed namespace: see display.cu.)

// The beauty's largest value, as Blender's Cycles clamps its own denoiser input: a sun seen directly (sunlit's) would
// otherwise pass 65504, the largest half float, which the network may compute in; 10^4 is far above the light of any
// lamp or sky the other scenes show
constexpr float beauty_ceiling = 1e4f;

// A channel's light as the denoiser takes it: rounded to float (for a complete pass exactly the float the PFM stores,
// as tone_map_kernel computes it), NaN as 0 (a sample's NaN counts as 0 in the sums already: this is for the product),
// and clamped to [0, 10^4]
__device__ inline float beauty_value(double light)
{
    const auto f = static_cast<float>(light);
    return f == f ? fminf(fmaxf(f, 0.0f), beauty_ceiling) : 0.0f;
}

// One thread per pixel of the band, laid out as rgba8_kernel's: the band's sums times its scale, alpha 1
__global__ void beauty_kernel(const double* sums, int width, int first_row, int row_count, double scale,
                              float4* beauty)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int j = first_row + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= width || j >= first_row + row_count)
    {
        return;
    }
    // In 64 bits: an image of more than 2^29 pixels has offsets beyond what an int holds
    const std::uint64_t pixel =
        static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(width) + static_cast<std::uint64_t>(i);
    const double* sum = sums + 3 * pixel;
    beauty[pixel] =
        make_float4(beauty_value(sum[0] * scale), beauty_value(sum[1] * scale), beauty_value(sum[2] * scale), 1.0f);
}

// One thread per pixel: a float4's three channels as doubles, exactly
__global__ void widen_kernel(const float4* image, int width, int height, double* sums)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= width || j >= height)
    {
        return;
    }
    const std::uint64_t pixel =
        static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(width) + static_cast<std::uint64_t>(i);
    const float4 v = image[pixel];
    double* sum = sums + 3 * pixel;
    sum[0] = static_cast<double>(v.x);
    sum[1] = static_cast<double>(v.y);
    sum[2] = static_cast<double>(v.z);
}

namespace
{
// display.cu's blocks: 32×8 pixels, at most 65535 blocks down (validate_camera in renderer.cu allows at most 65535 × 8
// rows)
dim3 image_grid(int width, int rows)
{
    return {static_cast<unsigned>((width + 31) / 32), static_cast<unsigned>((rows + 7) / 8)};
}
}

void to_beauty(const double* sums, int width, int first_row, int row_count, double scale, float4* beauty)
{
    if (row_count <= 0)
    {
        return; // a grid of no blocks is an error for CUDA
    }
    beauty_kernel<<<image_grid(width, row_count), dim3(32, 8)>>>(sums, width, first_row, row_count, scale, beauty);
    check(cudaGetLastError(), "making the denoiser's image");
}

void to_sums(const float4* image, int width, int height, double* sums)
{
    if (height <= 0)
    {
        return;
    }
    widen_kernel<<<image_grid(width, height), dim3(32, 8)>>>(image, width, height, sums);
    check(cudaGetLastError(), "widening the denoised image");
}
}
