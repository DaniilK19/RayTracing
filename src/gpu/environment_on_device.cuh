// The environment map on the GPU (quality_view::environment): its texels with their densities, its rows' cosines and
// its alias table, a part of the quality kernel's arrays (quality_scene in quality.cu) in a file of its own, so that
// the environment fills it without touching the other parts. Made from scene_desc::quality.environment. Host code.
//
// Empty until the environment map comes: view() is the camera's sky (texels nullptr), not sampled.
#pragma once

#include "device_memory.h"
#include "quality_view.cuh"
#include "renderer.h"

namespace gpu
{
class environment_on_device
{
public:
    explicit environment_on_device(const scene_desc& /*scene*/) {}

    environment_view view() const { return {}; }
    bool sampled() const { return false; } // a map with light in it, and its alias table: a light of the set
};
}
