// denoise.cuh: the denoiser's CUDA kernels for renderer.cu (denoise.cu, guides.cu). Host declarations only.
#pragma once
#include <cstdint>
#include <vector_types.h>
#include "camera.cuh"
#include "quality.cuh"
#include "scene.cuh"

namespace gpu::detail
{
// float4 per pixel of rows [first_row, first_row + row_count): float(sum · scale), NaN → 0, clamped to [0, 10^4],
// alpha 1 (display()'s two bands are two calls). Stream 0.
void to_beauty(const double* sums, int width, int first_row, int row_count, double scale, float4* beauty);
// 3 doubles per pixel from float4: to_rgba8 and to_rgba8_toned at scale 1 then give exactly these floats' bytes
void to_sums(const float4* image, int width, int height, double* sums);
// The guides of a CUDA session (guide_kernel<bvh_tracer>): quality is null for the books' kernels (no side entries,
// lamps or map); 16 samples per pixel; stream 0
void launch_guides(const quality_scene* quality, const scene_view<float>& base, const camera<float>& cam,
                   std::uint64_t seed, float4* albedo, float4* normal);
}
