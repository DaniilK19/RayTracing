// renderer.cu's view of tonemap.cu, as display.cuh is of display.cu: a host function only. No device code of the tone
// maps enters renderer.cu's translation unit, so nvcc compiles the books' kernels there exactly as before.
#pragma once

#include <cstdint>

#include "renderer.h"

namespace gpu::detail
{
// display()'s bytes through a tone map other than books (std::invalid_argument for books, whose bytes come from
// to_rgba8 in display.cu), for rows [first_row, first_row + row_count) of an image `width` pixels wide, with that
// band's own scale: each channel's sum times `scale`, rounded to float (a complete pass's value as the PFM holds it),
// times exposure_scale = float(2^EV), through the tone map (tone_curves.cuh), and quantized with the pixel's dither;
// alpha 255. sums holds 3 doubles per pixel and rgba 4 bytes per pixel, both row by row from the top, in the GPU's
// memory; the other rows of rgba aren't written. Queues the work on the default stream and returns; a later copy from
// rgba waits for it.
void to_rgba8_toned(const double* sums, int width, int first_row, int row_count, double scale, tone_map op,
                    float exposure_scale, std::uint8_t* rgba);
}
