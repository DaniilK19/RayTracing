// The image's bytes through a tone map (see tonemap.cuh, and tone_curves.cuh for the curves), computed on the GPU for
// the window and for RayTracingGPU's PNG alike: the host's and the device's powf and log2f round differently by an
// ulp or two, so the bytes of a tone map have one implementation, this one, and every program shows the same bytes. A
// translation unit of its own, as display.cu is: the books' display kernel and the render kernels stay as they were.
// Compiled with -fmad=false, as display.cu (see CMakeLists.txt): nvcc rounds every product and every sum of the curves
// on its own instead of fusing pairs of them, so the bytes follow from the order the code is written in.
#include "tonemap.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "tone_curves.cuh"

namespace gpu::detail
{
// (Not in an unnamed namespace: see display.cu.)

// One thread per pixel of the band, laid out as rgba8_kernel's. The light is the sum times the band's scale in double,
// rounded to float: for a complete pass exactly the float the PFM stores (pixels() multiplies the sums by the same
// 1/passes in double, and the PFM rounds that to float). Then the exposure, one float multiplication: by exactly 1 at
// 0 EV.
__global__ void tone_map_kernel(const double* sums, int width, int first_row, int row_count, double scale, tone_map op,
                                float exposure_scale, std::uint8_t* rgba)
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
    const tone::rgb light{static_cast<float>(sum[0] * scale) * exposure_scale,
                          static_cast<float>(sum[1] * scale) * exposure_scale,
                          static_cast<float>(sum[2] * scale) * exposure_scale};
    const tone::rgb encoded = tone::encode(light, op);
    // The dither depends on the pixel's place in the image alone: a band split by the pass in progress doesn't change
    // it, nor does the frame
    const float dither = tone::ign(i, j) - 0.5f;
    std::uint8_t* out = rgba + 4 * pixel;
    out[0] = tone::quantize(encoded.r, dither);
    out[1] = tone::quantize(encoded.g, dither);
    out[2] = tone::quantize(encoded.b, dither);
    out[3] = 255;
}

void to_rgba8_toned(const double* sums, int width, int first_row, int row_count, double scale, tone_map op,
                    float exposure_scale, std::uint8_t* rgba)
{
    if (op == tone_map::books)
    {
        throw std::invalid_argument("the books' bytes come from to_rgba8 (display.cu), not from a tone map");
    }
    if (row_count <= 0)
    {
        return; // a grid of no blocks is an error for CUDA
    }
    // As to_rgba8: blocks 8 rows high, at most 65535 of them down (validate_camera in renderer.cu allows at most
    // 65535 × 8 rows)
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((width + 31) / 32), static_cast<unsigned>((row_count + 7) / 8));
    tone_map_kernel<<<grid, block>>>(sums, width, first_row, row_count, scale, op, exposure_scale, rgba);
    if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess)
    {
        throw std::runtime_error(std::string("converting the image for the screen through a tone map: ") +
                                 cudaGetErrorString(result));
    }
}
}
