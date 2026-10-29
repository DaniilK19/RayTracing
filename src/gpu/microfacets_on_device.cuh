// The materials' GGX side entries on the GPU (quality_view::microfacets), made from scene_desc::quality.microfacets:
// a part of the quality kernel's arrays (quality_scene in quality.cu), in a file of its own so that the GGX materials
// fill it without touching the other parts. Host code.
//
// Empty until the GGX materials come (validate accepts no side entry but `none` until then): data() is nullptr, and
// the kernel reads no side entry.
#pragma once

#include "device_memory.h"
#include "quality_view.cuh"
#include "renderer.h"

namespace gpu
{
class microfacets_on_device
{
public:
    explicit microfacets_on_device(const scene_desc& /*scene*/) {}

    // By material index; nullptr: no GGX in the scene
    const microfacet* data() const { return entries_.size() > 0 ? entries_.data() : nullptr; }

private:
    device_array<microfacet> entries_{0};
};
}
