// The lamps the quality kernel samples, on the GPU (quality_view::lamps): a part of its arrays (quality_scene in
// quality.cu), in a file of its own so that light sampling fills it without touching the other parts. Made from the
// scene's diffuse_light primitives; sphere_ids and quad_ids are the BVH's global ids of scene_desc's spheres and quads,
// by which a lamp is found again when a ray hits it (lamps_view::lamp_of). Host code.
//
// Empty until light sampling comes: no lamp, view() all null.
#pragma once

#include <span>

#include "device_memory.h"
#include "quality_view.cuh"
#include "renderer.h"

namespace gpu
{
class lamps_on_device
{
public:
    lamps_on_device(const scene_desc& /*scene*/, std::span<const int> /*sphere_ids*/, std::span<const int> /*quad_ids*/)
    {
    }

    lamps_view view() const { return {}; }
    int count() const { return 0; } // the lamps of the light set
};
}
