// The image's bytes for the screen (see display.cuh). Compiled with -fmad=false (see CMakeLists.txt): no addition here
// follows a multiplication, so there is nothing to fuse, but the flag makes sure nvcc never rounds these numbers
// differently from the CPU.
#include "display.cuh"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace gpu::detail
{
// (Not in an unnamed namespace: for a kernel that isn't a template, nvcc's generated host code would then declare its
// launch function extern and define it static, which MSVC warns about at /W4.)

// image.ixx's to_byte: linear_to_gamma (the square root, 0 for anything not above 0, NaN included), then
// std::clamp to [0, 0.999], then 256 times that, rounded down by the conversion. CUDA's double sqrt is correctly
// rounded, as the CPU's is, and 256 × g is exact (a power of two), so every byte equals the PNG's.
__device__ inline std::uint8_t to_byte(double linear)
{
    const double g = linear > 0 ? sqrt(linear) : 0.0;
    const double clamped = g < 0.0 ? 0.0 : (0.999 < g ? 0.999 : g);
    return static_cast<std::uint8_t>(256 * clamped);
}

__global__ void rgba8_kernel(const double* sums, int width, int first_row, int row_count, double scale,
                             std::uint8_t* rgba)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int j = first_row + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= width || j >= first_row + row_count)
    {
        return;
    }
    // In 64 bits: an image of more than 2^29 pixels has byte offsets beyond what an int holds
    const std::uint64_t pixel =
        static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(width) + static_cast<std::uint64_t>(i);
    const double* sum = sums + 3 * pixel;
    std::uint8_t* out = rgba + 4 * pixel;
    out[0] = to_byte(sum[0] * scale);
    out[1] = to_byte(sum[1] * scale);
    out[2] = to_byte(sum[2] * scale);
    out[3] = 255;
}

void to_rgba8(const double* sums, int width, int first_row, int row_count, double scale, std::uint8_t* rgba)
{
    if (row_count <= 0)
    {
        return; // a grid of no blocks is an error for CUDA
    }
    // Blocks 8 rows high: at most 65535 of them down, as CUDA allows, since validate_camera in renderer.cu allows
    // at most 65535 × 8 rows
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((width + 31) / 32), static_cast<unsigned>((row_count + 7) / 8));
    rgba8_kernel<<<grid, block>>>(sums, width, first_row, row_count, scale, rgba);
    if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess)
    {
        throw std::runtime_error(std::string("converting the image for the screen: ") + cudaGetErrorString(result));
    }
}
}
