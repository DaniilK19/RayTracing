// The lamps the quality kernel samples, on the GPU (quality_view::lamps): a part of its arrays (quality_scene in
// quality.cu), in a file of its own so that light sampling fills it without touching the other parts. Made from the
// scene's diffuse_light primitives (build_lamps in lamps.cuh); sphere_ids and quad_ids are the BVH's global ids of
// scene_desc's spheres and quads, by which a lamp is found again when a ray hits it (lamps_view::lamp_of). Host code.
//
// A scene without a lamp uploads nothing: view() is all null and count() 0, and the kernel reads lamp_of only when the
// set holds a lamp (lamp_hit_pdf in light_sampler.cuh).
#pragma once

#include <span>
#include <vector>

#include "device_memory.h"
#include "lamps.cuh"
#include "quality_view.cuh"
#include "renderer.h"

namespace gpu
{
class lamps_on_device
{
public:
    lamps_on_device(const scene_desc& scene, std::span<const int> sphere_ids, std::span<const int> quad_ids)
        : lamps_on_device(build_lamps(scene, sphere_ids, quad_ids))
    {
    }

    lamps_view view() const { return {.lamps = lamps_.data(), .lamp_of = lamp_of_.data(), .count = count_}; }
    int count() const { return count_; } // the lamps of the light set

private:
    explicit lamps_on_device(const lamp_table& table)
        : count_(static_cast<int>(table.lamps.size())),
          lamps_(table.lamps),
          lamp_of_(count_ > 0 ? table.lamp_of : std::vector<int>{})
    {
    }

    int count_; // first: the arrays below are made after it (members are initialized in their order here)
    device_array<emitter> lamps_;
    device_array<int> lamp_of_;
};
}
