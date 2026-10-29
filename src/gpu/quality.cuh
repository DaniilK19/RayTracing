// renderer.cu's view of quality.cu, as display.cuh is of display.cu: host functions only, and an opaque handle for the
// quality kernel's own arrays. No device type or function of the quality kernel enters renderer.cu's translation unit,
// so nvcc compiles the books' kernels there exactly as before. (camera.cuh and scene.cuh, for the launch's camera and
// the books' tables it reads, are renderer.cu's own headers already.)
#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "camera.cuh"
#include "renderer.h"
#include "scene.cuh"

namespace gpu
{
namespace detail
{
class quality_scene; // the quality kernel's own arrays on the GPU (quality.cu): lamps, side entries, map tables
struct quality_scene_deleter
{
    void operator()(quality_scene* scene) const noexcept;
};
using quality_scene_ptr = std::unique_ptr<quality_scene, quality_scene_deleter>;

// Copies scene.quality and the lamp table to the GPU. sphere_ids and quad_ids are the BVH's (bvh::sphere_ids/quad_ids),
// which give every lamp its global id. Throws std::invalid_argument for data validate() refuses.
quality_scene_ptr upload_quality(const scene_desc& scene, std::span<const int> sphere_ids,
                                 std::span<const int> quad_ids);

// One launch of the quality kernel: passes [first_pass, first_pass + pass_count) of rows [first_row, first_row +
// row_count), with the round's seed, as the books' kernels' launch_work. counts[0] gets the rays traced, counts[1] the
// shadow rays.
void launch_quality(const quality_scene& q, const camera<float>& cam, const scene_view<float>& base,
                    std::uint64_t seed, int first_pass, int pass_count, int first_row, int row_count, double* sums,
                    unsigned long long* counts);

kernel_info quality_kernel_info(); // "quality", registers, local bytes, blocks per multiprocessor
int quality_resident_blocks();     // blocks the whole GPU runs at once: for the launch planner's one-wave floor
}
}
