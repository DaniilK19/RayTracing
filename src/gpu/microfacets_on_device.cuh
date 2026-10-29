// The materials' GGX side entries on the GPU (quality_view::microfacets), made from scene_desc::quality.microfacets:
// a part of the quality kernel's arrays (quality_scene in quality.cu), in a file of its own so that the GGX materials
// fill it without touching the other parts. Host code.
//
// One entry per material, by the material's index, or none at all: the books' scenes have no side entries, data() is
// then nullptr, and the kernel reads no side entry (see make_bsdf in bsdf.cuh).
#pragma once

#include <vector>

#include "device_memory.h"
#include "quality_view.cuh"
#include "renderer.h"

namespace gpu
{
class microfacets_on_device
{
public:
    explicit microfacets_on_device(const scene_desc& scene) : entries_(entries(scene.quality.microfacets)) {}

    // By material index; nullptr: no GGX in the scene
    const microfacet* data() const { return entries_.size() > 0 ? entries_.data() : nullptr; }

private:
    device_array<microfacet> entries_;

    // The width α = r² and r itself, from the double r: computed in double and rounded to float once, here, so that
    // the kernel compares the host's α with min_ggx_alpha (r = 0 gives α = 0: the mirror)
    static std::vector<microfacet> entries(const std::vector<microfacet_desc>& descs)
    {
        std::vector<microfacet> result;
        result.reserve(descs.size());
        for (const microfacet_desc& d : descs)
        {
            result.push_back({.kind = d.kind,
                              .alpha = static_cast<float>(d.roughness * d.roughness),
                              .roughness = static_cast<float>(d.roughness)});
        }
        return result;
    }
};
}
