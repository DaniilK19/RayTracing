// The image's bytes for the screen, computed on the GPU: an internal header of the renderer (renderer.cu's session
// calls it). display.cu is a translation unit of its own, so this code can never change how nvcc compiles the
// kernels in renderer.cu.
#pragma once

#include <cstdint>

namespace gpu::detail
{
// Rows [first_row, first_row + row_count) of an image `width` pixels wide: each channel's sum times `scale`, as the
// byte the CPU's image::save_png computes from it (gamma 2, clamped to [0, 0.999], times 256, rounded down), and
// alpha 255. sums holds 3 doubles per pixel and rgba 4 bytes per pixel, both row by row from the top.
// Queues the work on the default stream and returns; a later copy from rgba waits for it.
void to_rgba8(const double* sums, int width, int first_row, int row_count, double scale, std::uint8_t* rgba);
}
